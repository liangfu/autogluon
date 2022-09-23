""" Example script for predicting columns of tables, demonstrating more advanced usage of fit().
    Note that all settings demonstrated here are just chosen for demonstration purposes (to minimize runtime), and do not represent wise choices to use in practice.
    To maximize predictive accuracy, we recommend you do NOT specify `hyperparameters` or `hyperparameter_tune_kwargs`, and instead specify `TabularPredictor(..., eval_metric=YOUR_METRIC).fit(..., presets='best_quality')`
"""

import autogluon.core as ag
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
import time
from numpy.testing import assert_allclose

def main():

    # Training time:
    train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')  # can be local CSV file as well, returns Pandas DataFrame
    train_data = train_data.head(100)  # subsample for faster demo
    print(train_data.head())
    label = 'class'  # specifies which column do we want to predict
    save_path = 'ag_hpo_models/'  # where to save trained models

    hyperparameters = {
        # 'NN_TORCH': {'num_epochs': 10, 'activation': 'relu', 'dropout_prob': ag.Real(0.0, 0.5)},
        # 'GBM': {'num_boost_round': 1000, 'learning_rate': ag.Real(0.01, 0.1, log=True)},
        # 'XGB': {'n_estimators': 1000, 'learning_rate': ag.Real(0.01, 0.1, log=True)},
        'RF': {'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}, 'criterion': 'gini'},
        'KNN': {},
        # 'NN_TORCH': {},
        'XT': {},
        # 'GBM': {},
        # 'CAT': {},
        # 'XGB': {},
    }
    # hyperparameters.update(get_hyperparameter_config('default'))
    print('\n'.join(list(get_hyperparameter_config('default').keys())))

    predictor = TabularPredictor(label=label, path=save_path).fit(
        train_data, hyperparameters=hyperparameters, hyperparameter_tune_kwargs='auto', time_limit=60
    )

    results = predictor.fit_summary()  # display detailed summary of fit() process
    print(results["leaderboard"])

    # Inference time:
    test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')  # another Pandas DataFrame
    # print(test_data.head())

    perf = predictor.evaluate(test_data)  # shorthand way to evaluate our predictor if test-labels are available
    # import pdb
    # pdb.set_trace()

    trainer = predictor._learner.load_trainer()
    model_names = trainer.get_model_names()
    # best = trainer._get_best()
    # model = trainer.load_model(best)
    # models = model.models

    for name in model_names:
        model = trainer.load_model(name)
        from autogluon.core.models.ensemble.weighted_ensemble_model import WeightedEnsembleModel
        from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import TabularNeuralNetTorchModel
        if isinstance(model, WeightedEnsembleModel):
            base_name = model.base_model_names[0]
            model = trainer.load_model(base_name)
        print(name, model)
        model.compile()

        # import pdb
        # pdb.set_trace()

        tic = time.time()
        X = predictor.transform_features(test_data)
        X = model.preprocess(X)
        print(f"pre elapsed: {(time.time() - tic)*1000.0:.0f} ms")

        tic = time.time()
        if isinstance(model, TabularNeuralNetTorchModel):
            y_pred_pt = model._predict_tabular_data(X)
            print(f"pt elapsed: {(time.time() - tic)*1000.0:.0f} ms")
        else:
            # y_pred_skl = model.model.predict_proba(X)
            y_pred_skl = model.model.predict(X)
            print(f"skl elapsed: {(time.time() - tic)*1000.0:.0f} ms")

            tic = time.time()
            # y_pred_onx = model._predict_proba_onnx(X)
            y_pred_onx = model._predict_onnx(X)
            print(f"onx elapsed: {(time.time() - tic)*1000.0:.0f} ms")

        assert_allclose(y_pred_skl, y_pred_onx)
        

    # Otherwise we make predictions and can evaluate them later:
    y_pred = predictor.predict_proba(test_data)
    perf = predictor.evaluate_predictions(y_true=test_data[label], y_pred=y_pred, auxiliary_metrics=True)

if __name__=="__main__":
    main()
