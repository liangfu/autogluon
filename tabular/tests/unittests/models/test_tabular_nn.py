
from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import TabularNeuralNetTorchModel


def test_tabular_nn_binary(fit_helper):
    fit_args = dict(
        hyperparameters={TabularNeuralNetTorchModel: {}},
    )
    dataset_name = 'adult'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_tabular_nn_multiclass(fit_helper):
    fit_args = dict(
        hyperparameters={TabularNeuralNetTorchModel: {}},
    )
    dataset_name = 'covertype_small'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_tabular_nn_regression(fit_helper):
    fit_args = dict(
        hyperparameters={TabularNeuralNetTorchModel: {}},
        time_limit=10,  # TabularNN trains for a long time on ames
    )
    dataset_name = 'ames'
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args)


def test_tabular_nn_binary_compile_tvm(fit_helper):
    fit_args = dict(
        hyperparameters={TabularNeuralNetTorchModel: {}},
    )
    dataset_name = 'adult'
    compiler_configs = {TabularNeuralNetTorchModel: {'compiler': 'tvm', 'batch_size': 512}}
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args,
                                        compile_models=True, compiler_configs=compiler_configs)


def test_tabular_nn_multiclass_compile_tvm(fit_helper):
    fit_args = dict(
        hyperparameters={TabularNeuralNetTorchModel: {}},
    )
    dataset_name = 'covertype_small'
    compiler_configs = {TabularNeuralNetTorchModel: {'compiler': 'tvm', 'batch_size': 512}}
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args,
                                        compile_models=True, compiler_configs=compiler_configs)


def test_tabular_nn_regression_compile_tvm(fit_helper):
    fit_args = dict(
        hyperparameters={TabularNeuralNetTorchModel: {}},
        time_limit=10,  # TabularNN trains for a long time on ames
    )
    dataset_name = 'ames'
    compiler_configs = {TabularNeuralNetTorchModel: {'compiler': 'tvm', 'batch_size': 512}}
    fit_helper.fit_and_validate_dataset(dataset_name=dataset_name, fit_args=fit_args,
                                        compile_models=True, compiler_configs=compiler_configs)
