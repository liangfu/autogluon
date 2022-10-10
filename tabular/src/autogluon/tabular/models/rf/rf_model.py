import logging
import math
import pickle
import sys
import time

import numpy as np
import psutil

from autogluon.common.features.types import R_BOOL, R_INT, R_FLOAT, R_CATEGORY
from autogluon.core.constants import MULTICLASS, REGRESSION, SOFTCLASS, QUANTILE
from autogluon.core.utils.exceptions import NotEnoughMemoryError, TimeLimitExceeded
from autogluon.core.utils.utils import normalize_pred_probas
from autogluon.core.utils import try_import_torch, try_import_tvm

from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator

logger = logging.getLogger(__name__)


# FIXME: DOESNT WORK ON REGRESSION PREDICTIONS
class RFOnnxPredictor:
    def __init__(self, obj, model, path):
        import onnxruntime as rt
        self.num_classes = obj.num_classes
        self.model = model
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


class RFTVMPredictor:
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


class RFNativeCompiler:
    name = 'native'
    save_in_pkl = True

    @staticmethod
    def can_compile():
        return True

    @staticmethod
    def compile(obj, path: str):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
        if isinstance(obj.model, (RandomForestClassifier, RandomForestRegressor,
                                  ExtraTreesClassifier, ExtraTreesRegressor)):
            with open(path + 'model_native.pkl', 'wb') as fp:
                fp.write(pickle.dumps(obj))
        else:
            pkl = None
            with open(path + 'model_native.pkl', 'rb') as fp:
                pkl = fp.read()
            return pickle.loads(pkl).model
        return obj.model

    @staticmethod
    def load(obj, path: str):
        pkl = None
        with open(path + 'model_native.pkl', 'rb') as fp:
            pkl = fp.read()
        model = pickle.loads(pkl)
        return model.model


class RFOnnxCompiler:
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
        # RFNativeCompiler.compile(obj=obj, path=path)
        return RFOnnxCompiler.load(obj=obj, path=path)

    @staticmethod
    def load(obj, path: str):
        model = RFNativeCompiler.load(obj, path)
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        initial_type = [('float_input', FloatTensorType([None, obj._num_features_post_process]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        return RFOnnxPredictor(obj=obj, model=onnx_model, path=path)


class RFTVMCompiler:
    name = 'tvm'
    save_in_pkl = False

    @staticmethod
    def can_compile():
        try:
            try_import_tvm()
            return True
        except ImportError:
            return False

    @staticmethod
    def compile(obj, path: str):
        RFNativeCompiler.compile(obj=obj, path=path)
        return RFTVMCompiler.load(obj=obj, path=path)

    @staticmethod
    def load(obj, path: str):
        compiler = obj._compiler.name
        model = RFNativeCompiler.load(obj, path)
        from hummingbird.ml import convert as hb_convert
        import os
        batch_size = 5120
        input_shape = (batch_size, obj._num_features_post_process)
        if os.path.exists(path + f"model_{compiler}.zip"):
            from hummingbird.ml import load as hb_load
            tvm_model = hb_load(path + f"model_{compiler}")
        else:
            if compiler == "tvm":
                print("Building TVM model, this may take a few minutes...")
            test_input = np.random.rand(*input_shape)
            tvm_model = hb_convert(model, compiler, test_input = test_input, extra_config={
                "batch_size": batch_size, "test_input": test_input})
            tvm_model.save(path + f"model_{compiler}")
        model = RFTVMPredictor(model=tvm_model)
        return model

class RFPyTorchCompiler(RFTVMCompiler):
    name = 'pytorch'

class RFTorchScriptCompiler(RFTVMCompiler):
    name = 'torchscript'

class RFModel(AbstractModel):
    """
    Random Forest model (scikit-learn): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None
        self._daal = False  # Whether daal4py backend is being used
        self._num_features_post_process = None

    def _get_model_type(self):
        if self.problem_type == QUANTILE:
            from .rf_quantile import RandomForestQuantileRegressor
            return RandomForestQuantileRegressor
        if self.params_aux.get('use_daal', False):
            # Disabled by default because OOB score does not yet work properly
            try:
                # FIXME: sklearnex OOB score is broken, returns biased predictions. Without this optimization, can't compute Efficient OOF.
                #  Refer to https://github.com/intel/scikit-learn-intelex/issues/933
                #  Current workaround: Forcibly set oob_score=True during fit to compute OOB during train time.
                #  Downsides:
                #    1. Slows down training slightly by forcing computation of OOB even if OOB is not needed (such as in medium_quality)
                #    2. Makes computing the correct pred_time_val difficult, as the time is instead added to the fit_time,
                #       and we would need to waste extra time to compute the proper pred_time_val post-fit.
                #       Therefore with sklearnex enabled, pred_time_val is incorrect.
                from sklearnex.ensemble import RandomForestClassifier, RandomForestRegressor
                logger.log(15, '\tUsing sklearnex RF backend...')
                self._daal = True
            except:
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                self._daal = False
        else:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            self._daal = False
        if self.problem_type in [REGRESSION, SOFTCLASS]:
            return RandomForestRegressor
        else:
            return RandomForestClassifier

    # TODO: X.fillna -inf? Add extra is_missing column?
    def _preprocess(self, X, **kwargs):
        X = super()._preprocess(X, **kwargs)

        if self._feature_generator is None:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        X = X.fillna(0).to_numpy(dtype=np.float32)
        return X

    def _set_default_params(self):
        default_params = {
            # TODO: 600 is much better, but increases info leakage in stacking -> therefore 300 is ~equal in stack ensemble final quality.
            #  Consider adding targeted noise to OOF to avoid info leakage, or increase `min_samples_leaf`.
            'n_estimators': 300,
            # Cap leaf nodes to 15000 to avoid large datasets using unreasonable amounts of memory/disk for RF/XT.
            #  Ensures that memory and disk usage of RF model with 300 n_estimators is at most ~500 MB for binary/regression, ~200 MB per class for multiclass.
            #  This has no effect on datasets with <=15000 rows, and minimal to no impact on datasets with <50000 rows.
            #  For large datasets, will often make the model worse, but will significantly speed up inference speed and massively reduce memory and disk usage.
            #  For example, when left uncapped, RF can use 5 GB of disk for a regression dataset with 2M rows.
            #  Multiply by the 8 RF/XT models in config for best quality / high quality and this is 40 GB of tree models, which is unreasonable.
            #  This size scales linearly with number of rows.
            'max_leaf_nodes': 15000,
            'n_jobs': -1,
            'random_state': 0,
            'bootstrap': True,  # Required for OOB estimates, setting to False will raise exception if bagging.
            # TODO: min_samples_leaf=5 is too large on most problems, however on some datasets it helps a lot (airlines likes >40 min_samples_leaf, adult likes 2 much better than 1)
            #  This value would need to be tuned per dataset, likely very worthwhile.
            #  Higher values = less OOF info leak, default = 1, which maximizes info leak.
            # 'min_samples_leaf': 5,  # Significantly reduces info leakage to stacker models. Never use the default/1 when using as base model.
            # 'oob_score': True,  # Disabled by default as it is better to do it post-fit via custom logic.
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    # TODO: Add in documentation that Categorical default is the first index
    # TODO: enable HPO for RF models
    def _get_default_searchspace(self):
        spaces = {
            # 'n_estimators': Int(lower=10, upper=1000, default=300),
            # 'max_features': Categorical(['auto', 0.5, 0.25]),
            # 'criterion': Categorical(['gini', 'entropy']),
        }
        return spaces

    def _get_num_trees_per_estimator(self):
        # Very rough guess to size of a single tree before training
        if self.problem_type in [MULTICLASS, SOFTCLASS]:
            if self.num_classes is None:
                num_trees_per_estimator = 10  # Guess since it wasn't passed in, could also check y for a better value
            else:
                num_trees_per_estimator = self.num_classes
        else:
            num_trees_per_estimator = 1
        return num_trees_per_estimator

    def _estimate_memory_usage(self, X, **kwargs):
        params = self._get_model_params()
        n_estimators_final = params['n_estimators']
        n_estimators_minimum = min(40, n_estimators_final)
        num_trees_per_estimator = self._get_num_trees_per_estimator()
        bytes_per_estimator = num_trees_per_estimator * len(X) / 60000 * 1e6  # Underestimates by 3x on ExtraTrees
        expected_min_memory_usage = bytes_per_estimator * n_estimators_minimum
        return expected_min_memory_usage

    def _validate_fit_memory_usage(self, **kwargs):
        max_memory_usage_ratio = self.params_aux['max_memory_usage_ratio']
        available_mem = psutil.virtual_memory().available
        expected_min_memory_usage = self.estimate_memory_usage(**kwargs) / available_mem
        if expected_min_memory_usage > (0.5 * max_memory_usage_ratio):  # if minimum estimated size is greater than 50% memory
            logger.warning(f'\tWarning: Model is expected to require {round(expected_min_memory_usage * 100, 2)}% of available memory (Estimated before training)...')
            raise NotEnoughMemoryError

    def _fit(self,
             X,
             y,
             num_cpus=-1,
             time_limit=None,
             sample_weight=None,
             **kwargs):
        time_start = time.time()

        model_cls = self._get_model_type()

        max_memory_usage_ratio = self.params_aux['max_memory_usage_ratio']
        params = self._get_model_params()
        if 'n_jobs' not in params:
            params['n_jobs'] = num_cpus
        n_estimators_final = params['n_estimators']

        n_estimators_minimum = min(40, n_estimators_final)
        n_estimators_test = min(4, max(1, math.floor(n_estimators_minimum/5)))

        X = self.preprocess(X)
        self._num_features_post_process = X.shape[1]
        n_estimator_increments = [n_estimators_final]

        num_trees_per_estimator = self._get_num_trees_per_estimator()
        bytes_per_estimator = num_trees_per_estimator * len(X) / 60000 * 1e6  # Underestimates by 3x on ExtraTrees
        available_mem = psutil.virtual_memory().available
        expected_memory_usage = n_estimators_final * bytes_per_estimator / available_mem
        
        if n_estimators_final > n_estimators_test * 2:
            if self.problem_type == MULTICLASS:
                n_estimator_increments = [n_estimators_test, n_estimators_final]
                params['warm_start'] = True
            else:
                if expected_memory_usage > (0.05 * max_memory_usage_ratio):  # Somewhat arbitrary, consider finding a better value, should it scale by cores?
                    # Causes ~10% training slowdown, so try to avoid if memory is not an issue
                    n_estimator_increments = [n_estimators_test, n_estimators_final]
                    params['warm_start'] = True

        params['n_estimators'] = n_estimator_increments[0]
        if self._daal:
            if params.get('warm_start', False):
                params['warm_start'] = False
            # FIXME: This is inefficient but sklearnex doesn't support computing oob_score after training
            params['oob_score'] = True

        model = model_cls(**params)

        time_train_start = time.time()
        for i, n_estimators in enumerate(n_estimator_increments):
            if i != 0:
                if params.get('warm_start', False):
                    model.n_estimators = n_estimators
                else:
                    params['n_estimators'] = n_estimators
                    model = model_cls(**params)
            model = model.fit(X, y, sample_weight=sample_weight)
            if (i == 0) and (len(n_estimator_increments) > 1):
                time_elapsed = time.time() - time_train_start
                model_size_bytes = 0
                for estimator in model.estimators_:  # Uses far less memory than pickling the entire forest at once
                    model_size_bytes += sys.getsizeof(pickle.dumps(estimator))
                expected_final_model_size_bytes = model_size_bytes * (n_estimators_final / model.n_estimators)
                available_mem = psutil.virtual_memory().available
                model_memory_ratio = expected_final_model_size_bytes / available_mem

                ideal_memory_ratio = 0.15 * max_memory_usage_ratio
                n_estimators_ideal = min(n_estimators_final, math.floor(ideal_memory_ratio / model_memory_ratio * n_estimators_final))

                if n_estimators_final > n_estimators_ideal:
                    if n_estimators_ideal < n_estimators_minimum:
                        logger.warning(f'\tWarning: Model is expected to require {round(model_memory_ratio*100, 2)}% of available memory...')
                        raise NotEnoughMemoryError  # don't train full model to avoid OOM error
                    logger.warning(f'\tWarning: Reducing model \'n_estimators\' from {n_estimators_final} -> {n_estimators_ideal} due to low memory. Expected memory usage reduced from {round(model_memory_ratio*100, 2)}% -> {round(ideal_memory_ratio*100, 2)}% of available memory...')

                if time_limit is not None:
                    time_expected = time_train_start - time_start + (time_elapsed * n_estimators_ideal / n_estimators)
                    n_estimators_time = math.floor((time_limit - time_train_start + time_start) * n_estimators / time_elapsed)
                    if n_estimators_time < n_estimators_ideal:
                        if n_estimators_time < n_estimators_minimum:
                            logger.warning(f'\tWarning: Model is expected to require {round(time_expected, 1)}s to train, which exceeds the maximum time limit of {round(time_limit, 1)}s, skipping model...')
                            raise TimeLimitExceeded
                        logger.warning(f'\tWarning: Reducing model \'n_estimators\' from {n_estimators_ideal} -> {n_estimators_time} due to low time. Expected time usage reduced from {round(time_expected, 1)}s -> {round(time_limit, 1)}s...')
                        n_estimators_ideal = n_estimators_time

                for j in range(len(n_estimator_increments)):
                    if n_estimator_increments[j] > n_estimators_ideal:
                        n_estimator_increments[j] = n_estimators_ideal
        if self._daal and model.criterion != 'entropy':
            # TODO: entropy is not accelerated by sklearnex, need to not set estimators_ to None to avoid crash
            # This reduces memory usage / disk usage.
            model.estimators_ = None
        self.model = model
        self.params_trained['n_estimators'] = self.model.n_estimators
        # self.compile(path=self.path)

    # def compile(self, backend="onnx"):
    #     if backend == "onnx":
    #         self._onnx_compile()
    #     else:
    #         self._tvm_compile()

    # def _onnx_compile(self):
    #     path = self.path
    #     print('compiling')
    #     # Convert into ONNX format
    #     from skl2onnx import convert_sklearn
    #     from skl2onnx.common.data_types import FloatTensorType
    #     import onnxruntime as rt
    #     import os
    #     import numpy
    #     initial_type = [('float_input', FloatTensorType([None, self._num_features_post_process]))]
    #     onx = self.model.model
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    #     with open(path + "model.onnx", "wb") as f:
    #         f.write(onx.SerializeToString())

    #     # Compute the prediction with ONNX Runtime
    #     self._model_onnx = rt.InferenceSession(path + "model.onnx")

    # def _load_onnx(self, path: str):
    #     import onnxruntime as rt
    #     sess = rt.InferenceSession(path + "model.onnx")
    #     return sess

    # def _predict_onnx(self, X):
    #     input_name = self._model_onnx.get_inputs()[0].name
    #     label_name = self._model_onnx.get_outputs()[0].name
    #     return self._model_onnx.run([label_name], {input_name: X})[0]

    # def _predict_proba_onnx(self, X):
    #     input_name = self._model_onnx.get_inputs()[0].name
    #     label_name = self._model_onnx.get_outputs()[1].name
    #     pred_proba = self._model_onnx.run([label_name], {input_name: X})[0]
    #     pred_proba = np.array([[r[i] for i in range(self.num_classes)] for r in pred_proba])
    #     return pred_proba

    # TODO: Remove this after simplifying _predict_proba to reduce code duplication. This is only present for SOFTCLASS support.
    def _predict_proba(self, X, **kwargs):
        X = self.preprocess(X, **kwargs)

        if self.problem_type == REGRESSION:
            return self.model.predict(X)
        elif self.problem_type == SOFTCLASS:
            return self.model.predict(X)
        elif self.problem_type == QUANTILE:
            return self.model.predict(X, quantile_levels=self.quantile_levels)

        y_pred_proba = self.model.predict_proba(X)
        # y_pred_proba = self._predict_proba_onnx(X)
        return self._convert_proba_to_unified_form(y_pred_proba)

    def get_oof_pred_proba(self, X, normalize=None, **kwargs):
        """X should be the same X passed to `.fit`"""
        y_oof_pred_proba = self._get_oof_pred_proba(X=X, **kwargs)
        if normalize is None:
            normalize = self.normalize_pred_probas
        if normalize:
            y_oof_pred_proba = normalize_pred_probas(y_oof_pred_proba, self.problem_type)
        y_oof_pred_proba = y_oof_pred_proba.astype(np.float32)
        return y_oof_pred_proba

    def _is_sklearn_1(self) -> bool:
        """Returns True if the trained model is from sklearn>=1.0"""
        return callable(getattr(self.model, "_set_oob_score_and_attributes", None))

    def _model_supports_oob_pred_proba(self) -> bool:
        """Returns True if model supports computing out-of-bag prediction probabilities"""
        # TODO: Remove `_set_oob_score` after sklearn version requirement is >=1.0
        return callable(getattr(self.model, "_set_oob_score", None)) or self._is_sklearn_1()

    # FIXME: Unknown if this works with quantile regression
    def _get_oof_pred_proba(self, X, y, **kwargs):
        if not self.model.bootstrap:
            raise ValueError('Forest models must set `bootstrap=True` to compute out-of-fold predictions via out-of-bag predictions.')

        oob_is_not_set = getattr(self.model, "oob_decision_function_", None) is None and getattr(self.model, "oob_prediction_", None) is None

        if oob_is_not_set and self._daal:
            raise AssertionError('DAAL forest backend does not support out-of-bag predictions.')

        # TODO: This can also be done via setting `oob_score=True` in model params,
        #  but getting the correct `pred_time_val` that way is not easy, since we can't time the internal call.
        if oob_is_not_set and self._model_supports_oob_pred_proba():
            X = self.preprocess(X)

            if getattr(self.model, "n_classes_", None) is not None:
                if self.model.n_outputs_ == 1:
                    self.model.n_classes_ = [self.model.n_classes_]
            from sklearn.tree._tree import DTYPE, DOUBLE
            X, y = self.model._validate_data(X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE)
            if y.ndim == 1:
                # reshape is necessary to preserve the data contiguity against vs
                # [:, np.newaxis] that does not.
                y = np.reshape(y, (-1, 1))
            if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
                y = np.ascontiguousarray(y, dtype=DOUBLE)
            if self._is_sklearn_1():
                # sklearn >= 1.0
                # TODO: Can instead do `_compute_oob_predictions` but requires post-processing. Skips scoring func.
                self.model._set_oob_score_and_attributes(X, y)
            else:
                # sklearn < 1.0
                # TODO: Remove once sklearn < 1.0 support is dropped
                self.model._set_oob_score(X, y)
            if getattr(self.model, "n_classes_", None) is not None:
                if self.model.n_outputs_ == 1:
                    self.model.n_classes_ = self.model.n_classes_[0]

        if getattr(self.model, "oob_decision_function_", None) is not None:
            y_oof_pred_proba = self.model.oob_decision_function_
            self.model.oob_decision_function_ = None  # save memory
        elif getattr(self.model, "oob_prediction_", None) is not None:
            y_oof_pred_proba = self.model.oob_prediction_
            self.model.oob_prediction_ = None  # save memory
        else:
            raise AssertionError(f'Model class {type(self.model)} does not support out-of-fold prediction generation.')

        # TODO: Regression does not return NaN for missing rows, instead it sets them to 0. This makes life hard.
        #  The below code corrects the missing rows to NaN instead of 0.
        # Don't bother if >60 trees, near impossible to have missing
        # If using 68% of data for training, chance of missing for each row is 1 in 11 billion.
        if self.problem_type == REGRESSION and self.model.n_estimators <= 60:
            from sklearn.ensemble._forest import _get_n_samples_bootstrap, _generate_unsampled_indices
            n_samples = len(y)

            n_predictions = np.zeros(n_samples)
            n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, self.model.max_samples)
            for estimator in self.model.estimators_:
                unsampled_indices = _generate_unsampled_indices(estimator.random_state, n_samples, n_samples_bootstrap)
                n_predictions[unsampled_indices] += 1
            missing_row_mask = n_predictions == 0
            y_oof_pred_proba[missing_row_mask] = np.nan

        # fill missing prediction rows with average of non-missing rows
        if np.isnan(np.sum(y_oof_pred_proba)):
            if len(y_oof_pred_proba.shape) == 1:
                col_mean = np.nanmean(y_oof_pred_proba)
                y_oof_pred_proba[np.isnan(y_oof_pred_proba)] = col_mean
            else:
                col_mean = np.nanmean(y_oof_pred_proba, axis=0)
                inds = np.where(np.isnan(y_oof_pred_proba))
                y_oof_pred_proba[inds] = np.take(col_mean, inds[1])

        return self._convert_proba_to_unified_form(y_oof_pred_proba)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=[R_BOOL, R_INT, R_FLOAT, R_CATEGORY],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    @classmethod
    def _get_default_ag_args_ensemble(cls, problem_type=None, **kwargs) -> dict:
        default_ag_args_ensemble = super()._get_default_ag_args_ensemble(problem_type=problem_type, **kwargs)
        if problem_type != QUANTILE:  # use_child_oof not supported in quantile regression
            extra_ag_args_ensemble = {'use_child_oof': True}
            default_ag_args_ensemble.update(extra_ag_args_ensemble)
        return default_ag_args_ensemble

    def _more_tags(self):
        # `can_refit_full=True` because final n_estimators is communicated at end of `_fit`:
        #  `self.params_trained['n_estimators'] = self.model.n_estimators`
        tags = {'can_refit_full': True}
        if self.problem_type == QUANTILE:
            tags['valid_oof'] = False  # not supported in quantile regression
        else:
            tags['valid_oof'] = True
        return tags

    # def save(self, path: str = None, verbose=True) -> str:
    #     _model = self.model
    #     # self.model = None
    #     # _model_onnx = self._model_onnx
    #     # self._model_onnx = None
    #     if _model is not None:
    #         self._is_model_saved = True
    #     # if _model_onnx is not None:
    #     #     self._is_model_onnx_saved = True
    #     path = super().save(path=path, verbose=verbose)
    #     # if _model is not None:
    #     #     save_func_map = {
    #     #         'onnx': self._save_onnx,
    #     #         None: self._save_native,
    #     #     }
    #     #     kwargs = dict(path=path, model=_model)
    #     #     save_func_map[self._get_compiler()](**kwargs)
    #     self.model = _model
    #     # self._model_onnx = _model_onnx
    #     return path
    #
    # @classmethod
    # def load(cls, path: str, reset_paths=True, verbose=True):
    #     model = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
    #     # model._model_onnx = model._load_onnx(path=path)
    #     # if model._is_model_saved:
    #     #     load_func_map = {
    #     #         'onnx': model._load_onnx,
    #     #         None: model._load_native,
    #     #     }
    #     #     model.model = load_func_map[model._get_compiler()](path=path)
    #     return model
    #
    def _valid_compilers(self):
        return [RFNativeCompiler, RFOnnxCompiler, RFPyTorchCompiler, RFTorchScriptCompiler, RFTVMCompiler]

    def _get_compiler(self):
        compiler = self.params_aux.get('compiler', None)
        compilers = self._valid_compilers()
        compiler_names = {c.name: c for c in compilers}
        if compiler is not None and compiler not in compiler_names:
            raise AssertionError(f'Unknown compiler: {compiler}. Valid compilers: {compiler_names}')
        if compiler is None:
            return RFOnnxCompiler
        compiler_cls = compiler_names[compiler]
        if not compiler_cls.can_compile():
            compiler_cls = RFNativeCompiler
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
