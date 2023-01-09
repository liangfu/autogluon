""" Example script for predicting columns of tables, demonstrating simple use-case """

import os
import shutil
import time

from autogluon.tabular import TabularDataset, TabularPredictor

def main():
    # Training time:
    train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')  # can be local CSV file as well, returns Pandas DataFrame
    train_data = train_data.head(500)  # subsample for faster demo
    # print(train_data.head())
    label = 'class'  # specifies which column do we want to predict
    save_path = 'ag_models/'  # where to save trained models
    save_path_clone_opt = '~/Downloads/AdultIncomeBinaryClassificationModel_0_6_2'
    # save_path_clone_opt = 'ag_models_deploy/'  # where to save trained models

    # if os.path.exists(save_path):
    #     shutil.rmtree(save_path, ignore_errors=True)
    # if os.path.exists(save_path_clone_opt):
    #     shutil.rmtree(save_path_clone_opt, ignore_errors=True)

    # # NOTE: Default settings above are intended to ensure reasonable runtime at the cost of accuracy. To maximize predictive accuracy, do this instead:
    # predictor = TabularPredictor(label=label, path=save_path).fit(train_data, hyperparameters={
    #     "NN_TORCH": {}
    # }, presets='good_quality', time_limit=600)
    # predictor.save()

    # predictor = TabularPredictor.load(save_path)
    # results = predictor.fit_summary()
    # path_clone_opt = predictor.clone_for_deployment(path=save_path_clone_opt)

    predictor = TabularPredictor.load(save_path_clone_opt)
    results = predictor.fit_summary()

    # Inference time:
    test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')  # another Pandas DataFrame
    test_data = test_data.head()
    y_test = test_data[label]
    test_data = test_data.drop(labels=[label], axis=1)  # delete labels from test data since we wouldn't have them in practice
    # print(test_data.head())

    # predictor = TabularPredictor.load(save_path)

    predictor.persist_models()
    predictor.compile_models()

    for _ in range(3):
        tic = time.time()
        y_pred = predictor.predict(test_data)
        print(f"elapsed: {(time.time()-tic)*1000:.0f} ms")
    perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
    print(perf)

if __name__=="__main__":
    main()
