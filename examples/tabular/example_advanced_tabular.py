""" Example script for predicting columns of tables, demonstrating more advanced usage of fit().
    Note that all settings demonstrated here are just chosen for demonstration purposes (to minimize runtime), and do not represent wise choices to use in practice.
    To maximize predictive accuracy, we recommend you do NOT specify `hyperparameters` or `hyperparameter_tune_kwargs`, and instead specify `TabularPredictor(..., eval_metric=YOUR_METRIC).fit(..., presets='best_quality')`
"""

import autogluon.core as ag
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
import time
from numpy.testing import assert_allclose
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def benchmark(hyperparameters):

    # Training time:
    train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')  # can be local CSV file as well, returns Pandas DataFrame
    train_data = train_data.head(100)  # subsample for faster demo
    print(train_data.head())
    label = 'class'  # specifies which column do we want to predict
    save_path = 'ag_hpo_models/'  # where to save trained models

    # hyperparameters.update(get_hyperparameter_config('default'))
    print('\n'.join(list(get_hyperparameter_config('default').keys())))

    predictor = TabularPredictor(label=label, path=save_path).fit(
        train_data, hyperparameters=hyperparameters, time_limit=60
        # hyperparameter_tune_kwargs='auto'
    )
    predictor.save()
    predictor = TabularPredictor.load(path=save_path)

    results = predictor.fit_summary()  # display detailed summary of fit() process
    print(results["leaderboard"])

    # Inference time:
    test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')  # another Pandas DataFrame
    # print(test_data.head())

    perf = predictor.evaluate(test_data)  # shorthand way to evaluate our predictor if test-labels are available

    trainer = predictor._learner.load_trainer()
    model_names = trainer.get_model_names()

    for name in model_names:
        model = trainer.load_model(name)

        from autogluon.core.models.ensemble.weighted_ensemble_model import WeightedEnsembleModel
        from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import TabularNeuralNetTorchModel
        from autogluon.tabular.models.knn.knn_model import KNNModel
        from autogluon.tabular.models.rf.rf_model import RFModel
        from autogluon.tabular.models.xt.xt_model import XTModel
        from autogluon.tabular.models.lgb.lgb_model import LGBModel
        from autogluon.tabular.models.xgboost.xgboost_model import XGBoostModel

        if isinstance(model, WeightedEnsembleModel):
            base_name = model.base_model_names[0]
            model = trainer.load_model(base_name)
        print(name, model)
        model.compile()

        tic = time.time()
        X = predictor.transform_features(test_data)
        X = model.preprocess(X)
        print(f"pre elapsed: {(time.time() - tic)*1000.0:.0f} ms")

        if isinstance(model, TabularNeuralNetTorchModel):
            tic = time.time()
            y_pred_tch = model._predict_tabular_data(X)
            print(f"tch elapsed: {(time.time() - tic)*1000.0:.0f} ms")

            tic = time.time()
            X = model._tvm_compile(X, enable_tuner=False)
            compile_time = (time.time() - tic)*1000.0
            tic = time.time()
            y_pred_tvm = model._predict_tabular_data_tvm(X)
            print(f"tvm elapsed: {(time.time() - tic)*1000.0:.0f} ms + {compile_time:.0f} ms (compile + tune)")

            assert_allclose(y_pred_tch, y_pred_tvm, rtol=1e-5, atol=1e-5)
        elif isinstance(model, (KNNModel,)):
            tic = time.time()
            y_pred_skl = model.model.predict(X)
            print(f"skl elapsed: {(time.time() - tic)*1000.0:.0f} ms")

            tic = time.time()
            y_pred_onx = model._predict_onnx(X)
            print(f"onx elapsed: {(time.time() - tic)*1000.0:.0f} ms")
            assert_allclose(y_pred_skl, y_pred_onx)

        elif isinstance(model, (RFModel, XTModel, WeightedEnsembleModel)):
            # tic = time.time()
            # y_pred_skl = model.model.predict(X)
            # print(f"skl elapsed: {(time.time() - tic)*1000.0:.0f} ms")

            # tic = time.time()
            # y_pred_onx = model._predict_onnx(X)
            # print(f"onx elapsed: {(time.time() - tic)*1000.0:.0f} ms")
            # assert_allclose(y_pred_skl, y_pred_onx)

            model.params_aux['compiler'] = "native"
            model.compile()
            tic = time.time()
            y_pred_skl = model.model.predict_proba(X)
            print(f"{model.params_aux['compiler']} elapsed: {(time.time() - tic)*1000.0:.0f} ms")

            for compiler in ["onnx", "pytorch", "tvm"]:
                model.params_aux['compiler'] = compiler
                model.compile()
                tic = time.time()
                y_pred_tvm = model.model.predict_proba(X)
                print(f"{compiler} elapsed: {(time.time() - tic)*1000.0:.0f} ms")
                tic = time.time()
                y_pred_tvm = model.model.predict_proba(X)
                print(f"{compiler} elapsed: {(time.time() - tic)*1000.0:.0f} ms")
                assert_allclose(y_pred_skl, y_pred_tvm, rtol=1e-5, atol=1e-5)

        elif isinstance(model, (LGBModel)):
            model.params_aux['compiler'] = "native"
            model.compile()
            tic = time.time()
            y_pred_lgb = model.model.predict(X)
            print(f"{model.params_aux['compiler']} elapsed: {(time.time() - tic)*1000.0:.0f} ms")

            model.params_aux['compiler'] = "lleaves"
            model.compile()
            tic = time.time()
            y_pred_lleaves = model.model.predict(X)
            print(f"{model.params_aux['compiler']} elapsed: {(time.time() - tic)*1000.0:.0f} ms")
            tic = time.time()
            y_pred_lleaves = model.model.predict(X)
            print(f"{model.params_aux['compiler']} elapsed: {(time.time() - tic)*1000.0:.0f} ms")
            assert_allclose(y_pred_lgb, y_pred_lleaves)

            model.params_aux['compiler'] = "pytorch"
            model.compile()
            tic = time.time()
            y_pred_tvm = model.model.predict(X)
            print(f"{model.params_aux['compiler']} elapsed: {(time.time() - tic)*1000.0:.0f} ms")
            tic = time.time()
            y_pred_tvm = model.model.predict(X)
            print(f"{model.params_aux['compiler']} elapsed: {(time.time() - tic)*1000.0:.0f} ms")
            # assert_allclose(y_pred_lgb, y_pred_tvm)

            model.params_aux['compiler'] = "tvm"
            model.compile()
            tic = time.time()
            y_pred_tvm = model.model.predict(X)
            print(f"{model.params_aux['compiler']} elapsed: {(time.time() - tic)*1000.0:.0f} ms")
            tic = time.time()
            y_pred_tvm = model.model.predict(X)
            print(f"{model.params_aux['compiler']} elapsed: {(time.time() - tic)*1000.0:.0f} ms")
            # assert_allclose(y_pred_lgb, y_pred_tvm)

        elif isinstance(model, (XGBoostModel)):
            model.params_aux['compiler'] = "native"
            model.compile()
            tic = time.time()
            y_pred_native = model.model.predict(X)
            print(f"{model.params_aux['compiler']} elapsed: {(time.time() - tic)*1000.0:.0f} ms")

        else:
            raise NotImplementedError(f"{type(model)} is not supported.")
        

    # Otherwise we make predictions and can evaluate them later:
    y_pred = predictor.predict_proba(test_data)
    perf = predictor.evaluate_predictions(y_true=test_data[label], y_pred=y_pred, auxiliary_metrics=True)

if __name__=="__main__":
    hyperparameters = {
        # 'NN_TORCH': {'ag_args_fit': {'compiler': 'tvm'}},
        'RF': {'ag_args_fit': {'compiler': 'onnx'}},
        'KNN': {'ag_args_fit': {'compiler': 'onnx'}},
        'XT': {'ag_args_fit': {'compiler': 'onnx'}},
        'GBM': {'ag_args_fit': {'compiler': 'native'}},
        # 'XGB': {'ag_args_fit': {'compiler': 'native'}},
        # 'CAT': {},
    }
    # benchmark(hyperparameters)

    hyperparameters = {
        # 'NN_TORCH': {'ag_args_fit': {'compiler': 'tvm'}},
        'RF': {'ag_args_fit': {'compiler': 'native'}},
        # 'KNN': {'ag_args_fit': {'compiler': 'native'}},
        'XT': {'ag_args_fit': {'compiler': 'native'}},
        # 'GBM': {'ag_args_fit': {'compiler': 'native'}},
        # 'XGB': {'ag_args_fit': {'compiler': 'tvm'}},
        # 'CAT': {},
    }
    benchmark(hyperparameters)

    hyperparameters = {
        # 'NN_TORCH': {'ag_args_fit': {'compiler': 'tvm'}},
        # 'RF': {'ag_args_fit': {'compiler': 'onnx'}},
        # 'KNN': {'ag_args_fit': {'compiler': 'onnx'}},
        # 'XT': {'ag_args_fit': {'compiler': 'onnx'}},
        'GBM': {'ag_args_fit': {'compiler': 'native'}},
        # 'XGB': {'ag_args_fit': {'compiler': 'tvm'}},
        # 'CAT': {},
    }
    # benchmark(hyperparameters)
