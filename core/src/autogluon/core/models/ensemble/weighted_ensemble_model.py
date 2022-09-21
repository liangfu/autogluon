import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from .stacker_ensemble_model import StackerEnsembleModel
from ..greedy_ensemble.greedy_weighted_ensemble_model import GreedyWeightedEnsembleModel

logger = logging.getLogger(__name__)

class EnsembleOnnxCompiler:
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
        print('compiling')
        # Convert into ONNX format
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from sklearn.pipeline import Pipeline

        import pdb
        pdb.set_trace()
        for model in obj.models:
            name = model.base_model_names[0]
            base_model = obj.load_model(name)
            print(base_model)
                
        pipe = Pipeline(steps=[('model', naive_bayes_model)])
        initial_type = [('float_input', FloatTensorType([None, obj._num_features_post_process]))]
        onx = convert_sklearn(obj.model, initial_types=initial_type)

        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + "model.onnx", "wb") as f:
            f.write(onx.SerializeToString())
        return EnsembleOnnxCompiler.load(obj=obj, path=path)

    @staticmethod
    def load(obj, path: str):
        import onnxruntime as rt
        model = rt.InferenceSession(path + "model.onnx")
        return EnsembleOnnxPredictor(obj=obj, model=model)

    # @staticmethod
    # def predict(obj, X):
    #     input_name = obj._model_onnx.get_inputs()[0].name
    #     label_name = obj._model_onnx.get_outputs()[0].name
    #     return obj._model_onnx.run([label_name], {input_name: X})[0]
    #
    # @staticmethod
    # def predict_proba(obj, X):
    #     input_name = obj._model_onnx.get_inputs()[0].name
    #     label_name = obj._model_onnx.get_outputs()[1].name
    #     pred_proba = obj._model_onnx.run([label_name], {input_name: X})[0]
    #     pred_proba = np.array([[r[i] for i in range(obj.num_classes)] for r in pred_proba])
    #     return pred_proba


# TODO: v0.1 see if this can be removed and logic moved to greedy weighted ensemble model -> Use StackerEnsembleModel as stacker instead
# TODO: Optimize predict speed when fit on kfold, can simply sum weights
class WeightedEnsembleModel(StackerEnsembleModel):
    """
    Weighted ensemble meta-model that implements Ensemble Selection: https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf

    A :class:`autogluon.core.models.GreedyWeightedEnsembleModel` must be specified as the `model_base` to properly function.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.low_memory = False

    def _fit(self,
             X,
             y,
             **kwargs):
        super()._fit(X, y, **kwargs)
        stack_columns = []
        for model in self.models:
            model = self.load_child(model, verbose=False)
            stack_columns = stack_columns + [stack_column for stack_column in model.base_model_names if stack_column not in stack_columns]
        self.stack_column_prefix_lst = [stack_column for stack_column in self.stack_column_prefix_lst if stack_column in stack_columns]
        self.stack_columns, self.num_pred_cols_per_model = self.set_stack_columns(stack_column_prefix_lst=self.stack_column_prefix_lst)
        min_stack_column_prefix_to_model_map = {k: v for k, v in self.stack_column_prefix_to_model_map.items() if k in self.stack_column_prefix_lst}
        self.base_model_names = [base_model_name for base_model_name in self.base_model_names if base_model_name in min_stack_column_prefix_to_model_map.values()]
        self.stack_column_prefix_to_model_map = min_stack_column_prefix_to_model_map
        return self

    def _get_model_weights(self):
        weights_dict = defaultdict(int)
        num_models = len(self.models)
        for model in self.models:
            model: GreedyWeightedEnsembleModel = self.load_child(model, verbose=False)
            model_weight_dict = model._get_model_weights()
            for key in model_weight_dict.keys():
                weights_dict[key] += model_weight_dict[key]
        for key in weights_dict:
            weights_dict[key] = weights_dict[key] / num_models
        return weights_dict

    def compute_feature_importance(self, X, y, features=None, is_oof=True, **kwargs) -> pd.DataFrame:
        logger.warning('Warning: non-raw feature importance calculation is not valid for weighted ensemble since it does not have features, returning ensemble weights instead...')
        if is_oof:
            fi = pd.Series(self._get_model_weights()).sort_values(ascending=False)
        else:
            logger.warning('Warning: Feature importance calculation is not yet implemented for WeightedEnsembleModel on unseen data, returning generic feature importance...')
            fi = pd.Series(self._get_model_weights()).sort_values(ascending=False)

        fi_df = fi.to_frame(name='importance')
        fi_df['stddev'] = np.nan
        fi_df['p_score'] = np.nan
        fi_df['n'] = np.nan

        # TODO: Rewrite preprocess() in greedy_weighted_ensemble_model to enable
        # fi_df = super().compute_feature_importance(X=X, y=y, features_to_use=features_to_use, preprocess=preprocess, is_oof=is_oof, **kwargs)
        return fi_df

    def _set_default_params(self):
        default_params = {'use_orig_features': False}
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
        super()._set_default_params()

    # TODO: Pick up from here, rename to save, implement _get_compiler(), etc.
    def compile(self, path: str = None, verbose=True) -> str:
        if path is None:
            path = self.path
        file_path = path + self.model_file_name
        save_in_pkl = True
        if self.models is not None:
            self._compiler = self._get_compiler()
            if self._compiler is not None:
                self.models = self._compiler.compile(obj=self, path=path)
                save_in_pkl = self._compiler.save_in_pkl
        _models = self.models
        if not save_in_pkl:
            self.models = None
        save_pkl.save(path=file_path, object=self, verbose=verbose)
        self.models = _models
        return path

    def _valid_compilers(self):
        return [EnsembleOnnxCompiler]

    def _get_compiler(self):
        # import pdb
        # pdb.set_trace()
        compiler = self.params_aux.get('compiler', None)
        compilers = self._valid_compilers()
        compiler_names = {c.name: c for c in compilers}
        if compiler is not None and compiler not in compiler_names:
            raise AssertionError(f'Unknown compiler: {compiler}. Valid compilers: {compiler_names}')
        if compiler is None:
            return EnsembleOnnxCompiler
        compiler_cls = compiler_names[compiler]
        if not compiler_cls.can_compile():
            compiler_cls = EnsembleOnnxCompiler
        return compiler_cls

