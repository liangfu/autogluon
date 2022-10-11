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

from autogluon.core.models.ensemble.weighted_ensemble_model import WeightedEnsembleModel
from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import TabularNeuralNetTorchModel
from autogluon.tabular.models.knn.knn_model import KNNModel
from autogluon.tabular.models.rf.rf_model import RFModel
from autogluon.tabular.models.xt.xt_model import XTModel
from autogluon.tabular.models.lgb.lgb_model import LGBModel
from autogluon.tabular.models.xgboost.xgboost_model import XGBoostModel

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def benchmark(hyperparameters):
    # Training time:
    train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')  # can be local CSV file as well, returns Pandas DataFrame
    train_data = train_data.head(100)  # subsample for faster demo
    # print(train_data.head())
    label = 'class'  # specifies which column do we want to predict
    save_path = 'ag_hpo_models/'  # where to save trained models

    # hyperparameters.update(get_hyperparameter_config('default'))
    # print('\n'.join(list(get_hyperparameter_config('default').keys())))

    # Delete previously trained models
    import shutil, os
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    predictor = TabularPredictor(label=label, path=save_path).fit(
        train_data, hyperparameters=hyperparameters, time_limit=60, verbosity=0
        # hyperparameter_tune_kwargs='auto'
    )
    predictor.save()
    predictor = TabularPredictor.load(path=save_path)

    # results = predictor.fit_summary()  # display detailed summary of fit() process
    # print(results["leaderboard"])

    # Inference time:
    # test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')  # another Pandas DataFrame
    import polars as pl
    import pandas as pd
    test_data = pd.read_csv("test.csv")
    # print(test_data.head())

    # perf = predictor.evaluate(test_data)  # shorthand way to evaluate our predictor if test-labels are available

    trainer = predictor._learner.load_trainer()
    model_names = trainer.get_model_names()

    # This would help us load *ALL MODELS* in memory
    # Warning: choose to load best model only in production environment
    predictor.persist_models(models='all')

    supported_compilers = {
        "KNeighbors":   ["onnx", "pytorch"],
        "RandomForest": ["onnx", "pytorch", "tvm"],
        "ExtraTrees":   ["onnx", "pytorch", "tvm"],
        "LightGBM":     ["lleaves", "pytorch", "tvm"],
        # "XGB": ["pytorch", "tvm"],
        # "NN_TORCH": ["pytorch", "tvm"],
    }
    # for compiler in ["onnx", "pytorch", "tvm"]:

    # Evaluate timing results on all models
    for name in model_names:
        model = trainer.load_model(name)
        if isinstance(model, WeightedEnsembleModel):
            name = model.base_model_names[0]
            assert name in supported_compilers, f"unsupported base model {name}"
            model = trainer.load_model(name)

        # Get feature generators
        # feature_generators = predictor._learner.feature_generators[0].generators

        # Timing results for native compiler
        compiler = "native"
        model.params_aux['compiler'] = compiler
        model.compile()
        tic = time.time()
        y_pred_native = predictor.predict_proba(test_data, as_pandas=False, model=name)
        print(f"{compiler} elapsed {(time.time() - tic)*1000.0:.0f} ms ({name})")
        
        # Timing results for all supported compilers (e.g. onnx, pytorch, tvm)
        for compiler in supported_compilers[name]:
            model.params_aux['compiler'] = compiler
            model.compile()

            # Otherwise we make predictions and can evaluate them later:
            # tic = time.time()
            # if compiler == "tvm" and name == "RandomForest":
            #     import pdb
            #     pdb.set_trace()
            # y_pred = predictor.predict_proba(test_data, model=name)
            # print(f"elapsed {(time.time() - tic)*1000.0:.0f} ms for {hyperparameters}")
            tic = time.time()
            y_pred = predictor.predict_proba(test_data, model=name)
            print(f"{compiler} elapsed {(time.time() - tic)*1000.0:.0f} ms ({name})")

            # lightgbm model is not well supported in hummingbird converter,
            # considier replace lgb.basic.Booster with lgb.LGBMClassifier.
            if name not in ["LightGBM"]:
                assert_allclose(y_pred_native, y_pred, atol=1e-5, rtol=1e-5)

        # separate different models
        print('-------------------------------------------------------')

def benchmark_all(hyperparameters):

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
            model.params_aux['compiler'] = "native"
            model.compile()
            tic = time.time()
            y_pred_skl = model.model.predict_proba(X)
            print(f"skl elapsed: {(time.time() - tic)*1000.0:.0f} ms")

            # tic = time.time()
            # y_pred_onx = model._predict_onnx(X)
            # print(f"onx elapsed: {(time.time() - tic)*1000.0:.0f} ms")
            # assert_allclose(y_pred_skl, y_pred_onx)

            # for compiler in ["onnx", "pytorch", "tvm"]:
            for compiler in ["onnx", "pytorch", "torchscript"]:
                model.params_aux['compiler'] = compiler
                model.compile()
                tic = time.time()
                y_pred_tvm = model.model.predict_proba(X)
                print(f"{compiler} elapsed: {(time.time() - tic)*1000.0:.0f} ms")
                tic = time.time()
                y_pred_tvm = model.model.predict_proba(X)
                print(f"{compiler} elapsed: {(time.time() - tic)*1000.0:.0f} ms")
                assert_allclose(y_pred_skl, y_pred_tvm, rtol=1e-5, atol=1e-5)

        elif isinstance(model, (RFModel, XTModel, WeightedEnsembleModel)):
            model.params_aux['compiler'] = "native"
            model.compile()
            tic = time.time()
            y_pred_skl = model.model.predict_proba(X)
            print(f"{model.params_aux['compiler']} elapsed: {(time.time() - tic)*1000.0:.0f} ms")

            for compiler in ["onnx", "pytorch", "torchscript", "tvm"]:
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

            for compiler in ["lleaves", "pytorch", "torchscript", "tvm"]:
                model.params_aux['compiler'] = compiler
                model.compile()
                tic = time.time()
                y_pred_tvm = model.model.predict(X)
                print(f"{compiler} elapsed: {(time.time() - tic)*1000.0:.0f} ms")
                tic = time.time()
                y_pred_tvm = model.model.predict(X)
                print(f"{compiler} elapsed: {(time.time() - tic)*1000.0:.0f} ms")
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
        'RF': {'ag_args_fit': {'compiler': 'native'}},
        'KNN': {'ag_args_fit': {'compiler': 'native'}},
        'XT': {'ag_args_fit': {'compiler': 'native'}},
        'GBM': {'ag_args_fit': {'compiler': 'native'}},
        # 'XGB': {'ag_args_fit': {'compiler': 'tvm'}},
        # 'CAT': {},
    }
    # benchmark_all(hyperparameters)

    hyperparameters = {
        # 'NN_TORCH': {'ag_args_fit': {'compiler': 'native'}},
        'RF': {'ag_args_fit': {'compiler': 'native'}},
        'KNN': {'ag_args_fit': {'compiler': 'native'}},
        'XT': {'ag_args_fit': {'compiler': 'native'}},
        'GBM': {'ag_args_fit': {'compiler': 'native'}},
        # 'XGB': {'ag_args_fit': {'compiler': 'native'}},
        # 'CAT': {},
    }
    benchmark(hyperparameters)

    # hyperparameters = {'KNN': {'ag_args_fit': {'compiler': "native"}},}
    # y_pred_ref = benchmark(hyperparameters, fit=True)
    # for compiler in ["onnx", "pytorch"]:
    #     hyperparameters = {'KNN': {'ag_args_fit': {'compiler': compiler}},}
    #     y_pred_new = benchmark(hyperparameters, fit=False)
    #     assert_allclose(y_pred_ref, y_pred_new, rtol=1e-5, atol=1e-5)
    # print("-------------------------------------------------------")

    # hyperparameters = {'RF': {'ag_args_fit': {'compiler': "native"}},}
    # y_pred_ref = benchmark(hyperparameters, fit=True)
    # for compiler in ["onnx", "pytorch", "tvm"]:
    #     hyperparameters = {'RF': {'ag_args_fit': {'compiler': compiler}},}
    #     y_pred_new = benchmark(hyperparameters, fit=False)
    #     assert_allclose(y_pred_ref, y_pred_new, rtol=1e-5, atol=1e-5)
    # print("-------------------------------------------------------")

    # hyperparameters = {'XT': {'ag_args_fit': {'compiler': "native"}},}
    # y_pred_ref = benchmark(hyperparameters, fit=True)
    # for compiler in ["onnx", "pytorch", "tvm"]:
    #     hyperparameters = {'XT': {'ag_args_fit': {'compiler': compiler}},}
    #     y_pred_new = benchmark(hyperparameters, fit=False)
    #     assert_allclose(y_pred_ref, y_pred_new, rtol=1e-5, atol=1e-5)
