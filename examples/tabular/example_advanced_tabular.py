""" Example script for predicting columns of tables, demonstrating more advanced usage of fit().
    Note that all settings demonstrated here are just chosen for demonstration purposes (to minimize runtime), and do not represent wise choices to use in practice.
    To maximize predictive accuracy, we recommend you do NOT specify `hyperparameters` or `hyperparameter_tune_kwargs`, and instead specify `TabularPredictor(..., eval_metric=YOUR_METRIC).fit(..., presets='best_quality')`
"""
import sys
sys.path.insert(0, "/local/home/liangfc/workspace/autogluon")

import os
import shutil
import pandas as pd
import autogluon.core as ag
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core.utils.infer_utils import get_model_true_infer_speed_per_row_batch

columns_to_drop = [] # "capital-gain", "capital-loss", 'fnlwgt']

# Training time:
train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')  # can be local CSV file as well, returns Pandas DataFrame
train_data = train_data.drop(columns=columns_to_drop)
# train_data = train_data.head(100)  # subsample for faster demo
# print(train_data.head())
label = 'class'  # specifies which column do we want to predict
deploy_path = 'ag_hpo_models_deploy/'  # where to save trained models
save_path = 'ag_hpo_models/'  # where to save trained models

hyperparameters = {
    'NN_TORCH': {'num_epochs': 10, 'activation': 'relu'}, # , 'dropout_prob': ag.Real(0.0, 0.5)},
    # 'GBM': {'num_boost_round': 1000, 'learning_rate': ag.Real(0.01, 0.1, log=True)},
    # 'XGB': {'n_estimators': 1000, 'learning_rate': ag.Real(0.01, 0.1, log=True)}
}

# predictor = TabularPredictor(label=label, path=save_path).fit(
#     train_data, hyperparameters=hyperparameters, hyperparameter_tune_kwargs='auto', time_limit=60,
#     fit_weighted_ensemble=False
# )
# predictor.save()

predictor = TabularPredictor.load(path=save_path)
if os.path.exists(deploy_path):
    shutil.rmtree(deploy_path)
predictor.clone(path=deploy_path)

# Inference time:
test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')  # another Pandas DataFrame
# print(test_data.head())
test_data = test_data.drop(columns=columns_to_drop)

models = 'best'
paths = [save_path, deploy_path]
compile_flags = [False, True]
suffices = ["native", "tvm"]

repeats = 3
batch_sizes = [
    1,
    10,
    100,
    1000,
    # 10000,
    # 100000,
    # 1000000,
]
datasets = [test_data.head(bs) for bs in batch_sizes]

for batch_size, dataset in zip(batch_sizes, datasets):
    for path, compile_flag, suffix in zip(paths, compile_flags, suffices):
        thispath = f"{path}{suffix}_b{batch_size}"
        print(f"Working on {thispath}")
        predictor = TabularPredictor.load(path=save_path)
        compiler_configs = {"NN_TORCH": {"compiler": "tvm", "batch_size": batch_size}}

        if compile_flag:
            if os.path.exists(thispath):
                shutil.rmtree(thispath)
            predictor.clone(path=thispath)
            predictor = TabularPredictor.load(path=thispath)
            predictor.compile_models(models=models, compiler_configs=compiler_configs)
        predictor.persist_models(models=models)

        # results = predictor.fit_summary()  # display detailed summary of fit() process
        # print(results)

        # perf = predictor.evaluate(test_data)  # shorthand way to evaluate our predictor if test-labels are available

        # Otherwise we make predictions and can evaluate them later:
        import time
        for _ in range(repeats):
            tic = time.time()
            y_pred = predictor.predict_proba(dataset)
            toc = time.time()
            total_time = toc - tic
            print(f"elapsed (total): {total_time*1000:.0f} ms")
        print("--")
        # perf = predictor.evaluate_predictions(y_true=test_data[label], y_pred=y_pred, auxiliary_metrics=True)

def get_model_true_infer_speed_per_row_batch_bulk(data, predictor, batch_sizes: list = None, repeats=1, models='all') -> (pd.DataFrame, pd.DataFrame):
    if batch_sizes is None:
        batch_sizes = [
            1,
            10,
            100,
            1000,
            10000,
            100000,
        ]
    infer_dfs = dict()
    infer_transform_dfs = dict()
    for batch_size in batch_sizes:
        infer_df, time_per_row_transform = get_model_true_infer_speed_per_row_batch(data=data, predictor=predictor, batch_size=batch_size, repeats=repeats, models=models)
        infer_dfs[batch_size] = infer_df
        infer_transform_dfs[batch_size] = time_per_row_transform
    for key in infer_dfs.keys():
        infer_dfs[key] = infer_dfs[key].reset_index()
        infer_dfs[key]['batch_size'] = key

    infer_df_full_transform = pd.Series(infer_transform_dfs, name='pred_time_test').to_frame().rename_axis('batch_size')
    infer_df_full_transform['pred_time_test_marginal'] = infer_df_full_transform['pred_time_test']
    infer_df_full_transform['pred_time_test_with_transform'] = infer_df_full_transform['pred_time_test']
    infer_df_full_transform = infer_df_full_transform.reset_index()

    infer_df_full = pd.concat([infer_dfs[key] for key in infer_dfs.keys()])

    return infer_df_full, infer_df_full_transform

for path, compile_flag, suffix in zip(paths, compile_flags, suffices):
    infer_dfs = dict()
    infer_transform_dfs = dict()

    for batch_size in batch_sizes:
        predictor = TabularPredictor.load(path=save_path)
        compiler_configs = {"NN_TORCH": {"compiler": "tvm", "batch_size": batch_size}}
        if compile_flag:
            if os.path.exists(path):
                shutil.rmtree(path)
            predictor.clone(path=path)
            predictor = TabularPredictor.load(path=path)
            predictor.compile_models(models=models, compiler_configs=compiler_configs)
        predictor.persist_models(models=models)

        infer_df, time_per_row_transform = get_model_true_infer_speed_per_row_batch(data=test_data, predictor=predictor, batch_size=batch_size, repeats=repeats, models=models)
        infer_dfs[batch_size] = infer_df
        infer_transform_dfs[batch_size] = time_per_row_transform
    for key in infer_dfs.keys():
        infer_dfs[key] = infer_dfs[key].reset_index()
        infer_dfs[key]['batch_size'] = key

    infer_df_full_transform = pd.Series(infer_transform_dfs, name='pred_time_test').to_frame().rename_axis('batch_size')
    infer_df_full_transform['pred_time_test_marginal'] = infer_df_full_transform['pred_time_test']
    infer_df_full_transform['pred_time_test_with_transform'] = infer_df_full_transform['pred_time_test']
    infer_df_full_transform = infer_df_full_transform.reset_index()

    infer_df_full = pd.concat([infer_dfs[key] for key in infer_dfs.keys()])

    infer_df_full['rows_per_second'] = 1 / infer_df_full['pred_time_test']
    infer_df_full_transform['rows_per_second'] = 1 / infer_df_full_transform['pred_time_test']
    infer_df_full.to_csv(f'infer_df_{suffix}.csv')
    infer_df_full_transform.to_csv(f'infer_df_full_transform_{suffix}.csv')
