""" Example script for predicting columns of tables, demonstrating more advanced usage of fit().
    Note that all settings demonstrated here are just chosen for demonstration purposes (to minimize runtime), and do not represent wise choices to use in practice.
    To maximize predictive accuracy, we recommend you do NOT specify `hyperparameters` or `hyperparameter_tune_kwargs`, and instead specify `TabularPredictor(..., eval_metric=YOUR_METRIC).fit(..., presets='best_quality')`
"""

import numpy as np
import autogluon.core as ag
from autogluon.tabular import TabularDataset, TabularPredictor


# Training time:
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')  # can be local CSV file as well, returns Pandas DataFrame
# train_data = train_data.head(100)  # subsample for faster demo
# print(train_data.head())
label = 'class'  # specifies which column do we want to predict
save_path = 'ag_hpo_models/'  # where to save trained models

hyperparameters = {
    'NN_TORCH': {'num_epochs': 10, 'activation': 'relu'}, # , 'dropout_prob': ag.Real(0.0, 0.5)},
    # 'GBM': {'num_boost_round': 1000, 'learning_rate': ag.Real(0.01, 0.1, log=True)},
    # 'XGB': {'n_estimators': 1000, 'learning_rate': ag.Real(0.01, 0.1, log=True)}
}

# predictor = TabularPredictor(label=label, path=save_path).fit(
#     train_data, hyperparameters=hyperparameters # , hyperparameter_tune_kwargs='auto', time_limit=60
# )
# predictor.save()
predictor = TabularPredictor.load(path=save_path)
predictor.persist_models()

# results = predictor.fit_summary()  # display detailed summary of fit() process
# print(results)

# Inference time:
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')  # another Pandas DataFrame
test_data = test_data.head(10)
# test_data['age'] = np.nan
print(test_data)

# perf = predictor.evaluate(test_data)  # shorthand way to evaluate our predictor if test-labels are available
# print(perf)

# Otherwise we make predictions and can evaluate them later:
import time
for _ in range(3):
    tic = time.time()
    y_pred = predictor.predict_proba(test_data)
    print(f"elapsed (total): {(time.time()-tic)*1000:.0f} ms")
    # print(f"y_pred=\n{y_pred}")
# perf = predictor.evaluate_predictions(y_true=test_data[label], y_pred=y_pred, auxiliary_metrics=True)
