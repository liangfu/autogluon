import logging

import numpy as np
import math
import psutil
import time
import pickle
import os

from autogluon.common.features.types import R_INT, R_FLOAT, S_BOOL
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.utils import get_cpu_count
from autogluon.core.utils.exceptions import NotEnoughMemoryError
from autogluon.core.models import AbstractModel
from autogluon.core.utils.utils import normalize_pred_probas
from autogluon.core.utils import try_import_torch, try_import_tvm

logger = logging.getLogger(__name__)


# FIXME: DOESNT WORK ON REGRESSION PREDICTIONS
class KNNOnnxPredictor:
    def __init__(self, obj, model, path):
        import onnxruntime as rt
        self.num_classes = obj.num_classes
        self.model = model

        ########################################################
        # HACK, HACK, HACK
        if model.graph.initializer[0].data_type == 2:
            model.graph.initializer[0].data_type = 6
        ########################################################

        self._sess = rt.InferenceSession(model.SerializeToString())

    def predict(self, X):
        input_name = self._sess.get_inputs()[0].name
        label_name = self._sess.get_outputs()[0].name
        return self._sess.run([label_name], {input_name: X})[0]

    def predict_proba(self, X):
        input_name = self._sess.get_inputs()[0].name
        label_name = self._sess.get_outputs()[1].name
        pred_proba = self._sess.run([label_name], {input_name: X})[0]
        pred_proba = np.array([[r[i] for i in range(self.num_classes)] for r in pred_proba])
        return pred_proba


class KNNTVMPredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, X, **kwargs):
        y_pred = []
        batch_size = self.model._batch_size
        # batching via groupby
        for batch_id, batch_np in enumerate(np.split(X, np.arange(0, len(X), batch_size) + batch_size)):
            if batch_np.shape[0] != batch_size:
                # padding
                pad_shape = list(batch_np.shape)
                pad_shape[0] = batch_size - batch_np.shape[0]
                X_pad = np.zeros(tuple(pad_shape))
                batch_np = np.concatenate([batch_np, X_pad])
            y_pred.append(self.model.predict(batch_np.astype(np.float64)))
        y_pred = np.concatenate(y_pred)
        # remove padding
        if y_pred.shape[0] != X.shape[0]:
           y_pred = y_pred[:X.shape[0]]
        return y_pred

    def predict_proba(self, X, **kwargs):
        y_pred = []
        batch_size = self.model._batch_size
        # batching via groupby
        for batch_id, batch_np in enumerate(np.split(X, np.arange(0, len(X), batch_size) + batch_size)):
            if batch_np.shape[0] != batch_size:
                # padding
                pad_shape = list(batch_np.shape)
                pad_shape[0] = batch_size - batch_np.shape[0]
                X_pad = np.zeros(tuple(pad_shape))
                batch_np = np.concatenate([batch_np, X_pad])
            y_pred.append(self.model.predict_proba(batch_np.astype(np.float64)))
        y_pred = np.concatenate(y_pred)
        # remove padding
        if y_pred.shape[0] != X.shape[0]:
           y_pred = y_pred[:X.shape[0]]
        return y_pred


class KNNNativeCompiler:
    name = 'native'
    save_in_pkl = True

    @staticmethod
    def can_compile():
        return True

    @staticmethod
    def compile(obj, path: str):
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        if isinstance(obj.model, (KNeighborsClassifier, KNeighborsRegressor)):
            with open(path + 'model_native.pkl', 'wb') as fp:
                fp.write(pickle.dumps(obj))
        elif os.path.exists(path + 'model_native.pkl'):
            pkl = None
            with open(path + 'model_native.pkl', 'rb') as fp:
                pkl = fp.read()
            return pickle.loads(pkl).model
        else:
            raise RuntimeError(f"file {path + 'model_native.pkl'} not found for {type(obj.model)}")
        return obj.model

    @staticmethod
    def load(obj, path: str):
        if not os.path.exists(path + 'model_native.pkl'):
            import pdb
            pdb.set_trace()
            raise RuntimeError(f"file {path + 'model_native.pkl'} not found")
        pkl = None
        with open(path + 'model_native.pkl', 'rb') as fp:
            pkl = fp.read()
        model = pickle.loads(pkl)
        return model.model


class KNNOnnxCompiler:
    name = 'onnx'
    save_in_pkl = False

    @staticmethod
    def can_compile():
        try:
            import skl2onnx
            return True
        except ImportError:
            return False

    @staticmethod
    def compile(obj, path: str):
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        if isinstance(obj.model, (KNeighborsClassifier, KNeighborsRegressor)):
            KNNNativeCompiler.compile(obj=obj, path=path)
        return KNNOnnxCompiler.load(obj=obj, path=path)

    @staticmethod
    def load(obj, path: str):
        model = KNNNativeCompiler.load(obj, path)
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        initial_type = [('float_input', FloatTensorType([None, obj._num_features_post_process]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        return KNNOnnxPredictor(obj=obj, model=onnx_model, path=path)


class KNNTVMCompiler:
    name = 'tvm'
    save_in_pkl = False
    batch_size = 1024

    @staticmethod
    def can_compile():
        try:
            try_import_tvm()
            return True
        except ImportError:
            return False

    @staticmethod
    def compile(obj, path: str):
        KNNNativeCompiler.compile(obj=obj, path=path)
        return KNNTVMCompiler.load(obj=obj, path=path)

    @staticmethod
    def load(obj, path: str):
        compiler = obj._compiler.name
        model = KNNNativeCompiler.load(obj, path)
        from hummingbird.ml import convert as hb_convert
        import os
        batch_size = obj._compiler.batch_size
        model_path = path + f"model_{compiler}_b{batch_size}"
        input_shape = (batch_size, obj._num_features_post_process)
        if os.path.exists(model_path + ".zip"):
            from hummingbird.ml import load as hb_load
            tvm_model = hb_load(model_path)
        else:
            if compiler == "tvm":
                print("Building TVM model, this may take a few minutes...")
            test_input = np.random.rand(*input_shape)
            tvm_model = hb_convert(model, compiler, test_input = test_input, extra_config={
                "batch_size": batch_size, "test_input": test_input, "tvm_max_fuse_depth": 8})
            tvm_model.save(model_path)
        model = KNNTVMPredictor(model=tvm_model)
        return model

class KNNPyTorchCompiler(KNNTVMCompiler):
    name = 'pytorch'

class KNNTorchScriptCompiler(KNNTVMCompiler):
    name = 'torchscript'


# TODO: Normalize data!
class KNNModel(AbstractModel):
    """
    KNearestNeighbors model (scikit-learn): https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._X_unused_index = None  # Keeps track of unused training data indices, necessary for LOO OOF generation
        self._num_features_post_process = None

    def _get_model_type(self):
        if self.params_aux.get('use_daal', True):
            try:
                # TODO: Add more granular switch, currently this affects all future KNN models even if they had `use_daal=False`
                from sklearnex.neighbors import KNeighborsClassifier, KNeighborsRegressor
                # sklearnex backend for KNN seems to be 20-40x+ faster than native sklearn with no downsides.
                logger.log(15, '\tUsing sklearnex KNN backend...')
            except:
                from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        else:
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        if self.problem_type == REGRESSION:
            return KNeighborsRegressor
        else:
            return KNeighborsClassifier

    def _preprocess(self, X, **kwargs):
        X = super()._preprocess(X, **kwargs)
        X = X.fillna(0).to_numpy(dtype=np.float32)
        return X

    def _set_default_params(self):
        default_params = {
            'weights': 'uniform',
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=[R_INT, R_FLOAT],  # TODO: Eventually use category features
            ignored_type_group_special=[S_BOOL],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    @classmethod
    def _get_default_ag_args(cls) -> dict:
        default_ag_args = super()._get_default_ag_args()
        extra_ag_args = {'valid_stacker': False}
        default_ag_args.update(extra_ag_args)
        return default_ag_args

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {'use_child_oof': True}
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    # TODO: Enable HPO for KNN
    def _get_default_searchspace(self):
        spaces = {}
        return spaces

    def _fit(self,
             X,
             y,
             num_cpus=-1,
             time_limit=None,
             sample_weight=None,
             **kwargs):
        time_start = time.time()
        X = self.preprocess(X)
        self._num_features_post_process = X.shape[1]
        params = self._get_model_params()
        if 'n_jobs' not in params:
            params['n_jobs'] = num_cpus
        if sample_weight is not None:  # TODO: support
            logger.log(15, "sample_weight not yet supported for KNNModel, this model will ignore them in training.")

        num_rows_max = len(X)
        # FIXME: v0.1 Must store final num rows for refit_full or else will use everything! Worst case refit_full could train far longer than the original model.
        if time_limit is None or num_rows_max <= 10000:
            self.model = self._get_model_type()(**params).fit(X, y)
        else:
            self.model = self._fit_with_samples(X=X, y=y, model_params=params, time_limit=time_limit - (time.time() - time_start))

    def _estimate_memory_usage(self, X, **kwargs):
        model_size_bytes = 4 * X.shape[0] * X.shape[1]  # Assuming float32 types
        expected_final_model_size_bytes = model_size_bytes * 3.6 # Roughly what can be expected of the final KNN model in memory size
        return expected_final_model_size_bytes

    def _validate_fit_memory_usage(self, **kwargs):
        max_memory_usage_ratio = self.params_aux['max_memory_usage_ratio']
        expected_final_model_size_bytes = self.estimate_memory_usage(**kwargs) 
        if expected_final_model_size_bytes > 10000000:  # Only worth checking if expected model size is >10MB
            available_mem = psutil.virtual_memory().available
            model_memory_ratio = expected_final_model_size_bytes / available_mem
            if model_memory_ratio > (0.15 * max_memory_usage_ratio):
                logger.warning(f'\tWarning: Model is expected to require {round(model_memory_ratio * 100, 2)}% of available memory...')
            if model_memory_ratio > (0.20 * max_memory_usage_ratio):
                raise NotEnoughMemoryError  # don't train full model to avoid OOM error

    def _valid_compilers(self):
        return [KNNNativeCompiler, KNNOnnxCompiler, KNNPyTorchCompiler, KNNTorchScriptCompiler, KNNTVMCompiler]

    def _get_compiler(self):
        compiler = self.params_aux.get('compiler', None)
        batch_size = self.params_aux.get('compiler.batch_size', 1024)
        compilers = self._valid_compilers()
        compiler_names = {c.name: c for c in compilers}
        if compiler is not None and compiler not in compiler_names:
            raise AssertionError(f'Unknown compiler: {compiler}. Valid compilers: {compiler_names}')
        if compiler is None:
            return KNNOnnxCompiler
        compiler_cls = compiler_names[compiler]
        compiler_cls.batch_size = batch_size
        if not compiler_cls.can_compile():
            compiler_cls = KNNNativeCompiler
        return compiler_cls

    def save(self, path: str = None, verbose=True) -> str:
        _model = self.model
        if self.model is not None:
            self._compiler = self._get_compiler()
            self._is_model_saved = True
            if self._compiler is not None:
                self.model = None  # Don't save model in pkl
        path = super().save(path=path, verbose=verbose)
        self.model = _model
        if _model is not None and self._compiler is not None:
            self.model = self._compiler.compile(obj=self, path=path)
        return path

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        model = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        if model._is_model_saved:
            model.model = model._compiler.load(obj=model, path=path)
        return model

    # TODO: Won't work for RAPIDS without modification
    # TODO: Technically isn't OOF, but can be used inplace of OOF. Perhaps rename to something more accurate?
    def get_oof_pred_proba(self, X, normalize=None, **kwargs):
        """X should be the same X passed to `.fit`"""
        y_oof_pred_proba = self._get_oof_pred_proba(X=X, **kwargs)
        if normalize is None:
            normalize = self.normalize_pred_probas
        if normalize:
            y_oof_pred_proba = normalize_pred_probas(y_oof_pred_proba, self.problem_type)
        y_oof_pred_proba = y_oof_pred_proba.astype(np.float32)
        return y_oof_pred_proba

    def _get_oof_pred_proba(self, X, **kwargs):
        from ._knn_loo_variants import KNeighborsClassifierLOOMixin, KNeighborsRegressorLOOMixin
        if self.problem_type in [BINARY, MULTICLASS]:
            y_oof_pred_proba = KNeighborsClassifierLOOMixin.predict_proba_loo(self.model)
        else:
            y_oof_pred_proba = KNeighborsRegressorLOOMixin.predict_loo(self.model)
        y_oof_pred_proba = self._convert_proba_to_unified_form(y_oof_pred_proba)
        if X is not None and self._X_unused_index:
            X_unused = X.iloc[self._X_unused_index]
            y_pred_proba_new = self.predict_proba(X_unused)
            X_unused_index = set(self._X_unused_index)
            num_rows = len(X)
            X_used_index = [i for i in range(num_rows) if i not in X_unused_index]
            oof_pred_shape = y_oof_pred_proba.shape
            if len(oof_pred_shape) == 1:
                y_oof_tmp = np.zeros(num_rows, dtype=np.float32)
                y_oof_tmp[X_used_index] = y_oof_pred_proba
                y_oof_tmp[self._X_unused_index] = y_pred_proba_new
            else:
                y_oof_tmp = np.zeros((num_rows, oof_pred_shape[1]), dtype=np.float32)
                y_oof_tmp[X_used_index, :] = y_oof_pred_proba
                y_oof_tmp[self._X_unused_index, :] = y_pred_proba_new
            y_oof_pred_proba = y_oof_tmp
        return y_oof_pred_proba

    # TODO: Consider making this fully generic and available to all models
    def _fit_with_samples(self,
                          X,
                          y,
                          model_params,
                          time_limit,
                          start_samples=10000,
                          max_samples=None,
                          sample_growth_factor=2,
                          sample_time_growth_factor=8):
        """
        Fit model with samples of the data repeatedly, gradually increasing the amount of data until time_limit is reached or all data is used.

        X and y must already be preprocessed.

        Parameters
        ----------
        X : np.ndarray
            The training data features (preprocessed).
        y : Series
            The training data ground truth labels.
        time_limit : float, default = None
            Time limit in seconds to adhere to when fitting model.
        start_samples : int, default = 10000
            Number of samples to start with. This will be multiplied by sample_growth_factor after each model fit to determine the next number of samples.
            For example, if start_samples=10000, sample_growth_factor=2, then the number of samples per model fit would be [10000, 20000, 40000, 80000, ...]
        max_samples : int, default = None
            The maximum number of samples to use.
            If None or greater than the number of rows in X, then it is set equal to the number of rows in X.
        sample_growth_factor : float, default = 2
            The rate of growth in sample size between each model fit. If 2, then the sample size doubles after each fit.
        sample_time_growth_factor : float, default = 8
            The multiplier to the expected fit time of the next model. If `sample_time_growth_factor=8` and a model took 10 seconds to train, the next model fit will be expected to take 80 seconds.
            If an expected time is greater than the remaining time in `time_limit`, the model will not be trained and the method will return early.
        """
        time_start = time.time()

        num_rows_samples = []
        if max_samples is None:
            num_rows_max = len(X)
        else:
            num_rows_max = min(len(X), max_samples)
        num_rows_cur = start_samples
        while True:
            num_rows_cur = min(num_rows_cur, num_rows_max)
            num_rows_samples.append(num_rows_cur)
            if num_rows_cur == num_rows_max:
                break
            num_rows_cur *= sample_growth_factor
            num_rows_cur = math.ceil(num_rows_cur)
            if num_rows_cur * 1.5 >= num_rows_max:
                num_rows_cur = num_rows_max

        def sample_func(chunk, frac):
            # Guarantee at least 1 sample (otherwise log_loss would crash or model would return different column counts in pred_proba)
            n = max(math.ceil(len(chunk) * frac), 1)
            return chunk.sample(n=n, replace=False, random_state=0)

        if self.problem_type != REGRESSION:
            y_df = y.to_frame(name='label').reset_index(drop=True)
        else:
            y_df = None

        time_start_sample_loop = time.time()
        time_limit_left = time_limit - (time_start_sample_loop - time_start)
        model_type = self._get_model_type()
        idx = None
        for i, samples in enumerate(num_rows_samples):
            if samples != num_rows_max:
                if self.problem_type == REGRESSION:
                    idx = np.random.choice(num_rows_max, size=samples, replace=False)
                else:
                    idx = y_df.groupby('label', group_keys=False).apply(sample_func, frac=samples/num_rows_max).index
                X_samp = X[idx, :]
                y_samp = y.iloc[idx]
            else:
                X_samp = X
                y_samp = y
                idx = None
            self.model = model_type(**model_params).fit(X_samp, y_samp)
            time_limit_left_prior = time_limit_left
            time_fit_end_sample = time.time()
            time_limit_left = time_limit - (time_fit_end_sample - time_start)
            time_fit_sample = time_limit_left_prior - time_limit_left
            time_required_for_next = time_fit_sample * sample_time_growth_factor
            logger.log(15, f'\t{round(time_fit_sample, 2)}s \t= Train Time (Using {samples}/{num_rows_max} rows) ({round(time_limit_left, 2)}s remaining time)')
            if time_required_for_next > time_limit_left and i != len(num_rows_samples) - 1:
                logger.log(20, f'\tNot enough time to train KNN model on all training rows. Fit {samples}/{num_rows_max} rows. (Training KNN model on {num_rows_samples[i+1]} rows is expected to take {round(time_required_for_next, 2)}s)')
                break
        if idx is not None:
            idx = set(idx)
            self._X_unused_index = [i for i in range(num_rows_max) if i not in idx]
        return self.model

    def _get_default_resources(self):
        # use at most 32 cpus to avoid OpenBLAS error: https://github.com/awslabs/autogluon/issues/1020
        num_cpus = min(32, get_cpu_count())
        num_gpus = 0
        return num_cpus, num_gpus

    def _more_tags(self):
        return {
            'valid_oof': True,
            'can_refit_full': True,
        }


class FAISSModel(KNNModel):
    def _get_model_type(self):
        from .knn_utils import FAISSNeighborsClassifier, FAISSNeighborsRegressor
        if self.problem_type == REGRESSION:
            return FAISSNeighborsRegressor
        else:
            return FAISSNeighborsClassifier

    def _set_default_params(self):
        default_params = {
            'index_factory_string': 'Flat',
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
        super()._set_default_params()

    @classmethod
    def _get_default_ag_args_ensemble(cls, **kwargs) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(**kwargs)
        extra_ag_args_ensemble = {'use_child_oof': False}
        default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    def _more_tags(self):
        return {'valid_oof': False}
