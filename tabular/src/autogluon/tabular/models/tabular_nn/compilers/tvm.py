import os
import numpy as np
import torch
import torch.nn as nn

from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS, QUANTILE


class GraphModuleWrapper:
    """
    Wrap around GraphModule in tvm runtime, since it cannot be pickled.
    """
    def __init__(self, model):
        from tvm.contrib import graph_executor
        self.module = graph_executor.GraphModule(model)
        self.batch_size = self.module._get_input("input0").shape[0]

    def run(self, *args):
        return self.module.run(*args)

    def set_input(self, *args):
        return self.module.set_input(*args)

    def get_output(self, *args):
        return self.module.get_output(*args)

    def __getstate__(self):
        # No need to duplicate the model parameters here.
        return {}

    def __setstate__(self, values):
        pass


class TabularNeuralNetTorchTvmPredictor:
    def __init__(self, model, architecture_desc : dict = None):
        if architecture_desc is None:
            architecture_desc = {}
        import tvm
        dev = tvm.cpu()
        self.module = GraphModuleWrapper(model["default"](dev))
        self.batch_size = self.module.batch_size
        self.has_vector_features = architecture_desc.get("has_vector_features", False)
        self.has_embed_features = architecture_desc.get("has_embed_features", False)
        self.problem_type = architecture_desc.get("problem_type", None)
        self.architecture_desc = architecture_desc

        if self.has_embed_features:
            num_categs_per_feature = architecture_desc['num_categs_per_feature']
            embed_dims = architecture_desc['embed_dims']
            self.embed_blocks = nn.ModuleList()
            for i in range(len(num_categs_per_feature)):
                self.embed_blocks.append(nn.Embedding(num_embeddings=num_categs_per_feature[i],
                                                      embedding_dim=embed_dims[i]))

    def _get_input_vector(self, data_batch : dict) -> list:
        input_data = []
        if self.has_vector_features:
            input_data.append(data_batch['vector'].numpy())
        if self.has_embed_features:
            if len(self.embed_blocks) != len(data_batch['embed']):
                raise ValueError("input data size doesn't match number of embed blocks.")
            with torch.no_grad():
                for i in range(len(self.embed_blocks)):
                    input_data.append(self.embed_blocks[i](data_batch['embed'][i]).numpy())
        if len(input_data) > 1:
            input_data = np.concatenate(input_data, axis=1)
        else:
            input_data = input_data[0]
        return input_data

    def predict(self, X):
        """Run the model with the input and return the result."""
        import tvm
        import time
        tic = time.time()
        input_data = self._get_input_vector(X)
        num_net_outputs = self.architecture_desc['num_net_outputs']
        data_size = input_data.shape[0]
        batch_size = self.batch_size
        out_shape = (batch_size, num_net_outputs)

        # Pad the batch if input data is small
        if data_size < batch_size:
            pad_shape = list(input_data.shape)
            pad_shape[0] = batch_size - data_size
            pad_data = np.zeros(tuple(pad_shape))
            input_data = np.concatenate([input_data, pad_data], axis=0)
        if data_size > batch_size:
            raise RuntimeError("Input data size is larger than expected. "
                               f"Try use DataLoader with batch_size={batch_size}.")
        print(f"elapsed (embed): {(time.time()-tic)*1000:.0f} ms")

        # Run on graph executor
        tic = time.time()
        self.module.set_input(f"input0", input_data)
        self.module.run()
        predict_data = self.module.get_output(0, tvm.nd.empty(out_shape)).numpy()
        print(f"elapsed (run): {(time.time()-tic)*1000:.0f} ms")

        # Remove padded result from output
        if data_size < batch_size:
            predict_data = predict_data[:data_size]

        # Problem type specific post processing
        if self.problem_type == QUANTILE:
            predict_data = torch.sort(predict_data, -1)[0]  # sorting ensures monotonicity of quantile estimates
        elif self.problem_type in [BINARY, MULTICLASS, SOFTCLASS]:
            from scipy.special import softmax
            predict_data = softmax(predict_data)  # convert NN output to probability
        elif self.problem_type == REGRESSION:
            predict_data = predict_data.flatten()
        if self.problem_type == BINARY:
            predict_data = predict_data[:,1]

        return predict_data

    def predict_proba(self, X):
        """Run the model with the input, and return probabilities as result."""
        pass


class TabularNeuralNetTorchTvmCompiler:
    name = 'tvm'
    save_in_pkl = False

    @staticmethod
    def can_compile():
        """Verify whether the required package has been installed."""
        try:
            import tvm
            return True
        except ImportError:
            return False

    @staticmethod
    def compile(model, path: str, input_types=None):
        """
        Compile the trained model for faster inference.

        Parameters
        ----------
        model
            The native model that is expected to be compiled.
        path : str
            The path for saving the compiled model.
        input_types : list, default=None
            A list of tuples containing shape and element type info, e.g. [((1, 14), np.float32),].
            The list would be used as the input data for the model.
            The compiler would optimize the model to perform best with the given input type.
        """
        import torch
        import tvm
        from tvm import relay, auto_scheduler
        from tvm.topi.sparse.utils import sparse_sketch_rules
        if input_types is None or not isinstance(input_types[0], tuple):
            raise RuntimeError("input_types argument should contain at least one tuple"
                               ", e.g. [((1, 14), np.float32)]")
        if isinstance(model, TabularNeuralNetTorchTvmPredictor):
            return model

        input_data = []
        for shape, dtype in input_types:
            if dtype == np.float32:
                input_data.append(torch.randn(shape, dtype=torch.float32))
            elif dtype == np.int32:
                input_data.append(torch.randint(low=0, high=1, size=(shape[0],)))
            else:
                raise NotImplementedError(f"not implemented dtype={dtype} for tvm compile.")
            
        scripted_model = torch.jit.trace(model, input_data).eval()

        shape_list = [(f"input{i}", t[0] if t[1] == np.float32 else (t[0][0],)) for i, t in enumerate(input_types)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

        target = tvm.target.Target("llvm", host="llvm")
        dev = tvm.cpu(0)

        # log_file = path + "log.txt"
        # tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
        # tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        # tune_option = auto_scheduler.TuningOptions(
        #     num_measure_trials=len(tasks)*2,  # change this to 20000 to achieve the best performance
        #     runner=auto_scheduler.LocalRunner(repeat=6, enable_cpu_cache_flush=True),
        #     measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        #     verbose=0,
        # )
        # tuner.tune(tune_option)
        # with auto_scheduler.ApplyHistoryBest(log_file):
        #     with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        #         lib = relay.build(mod, target=target, params=params)
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)

        model.architecture_desc['problem_type'] = model.problem_type
        TabularNeuralNetTorchTvmCompiler.save(lib, path, model.architecture_desc)

    @staticmethod
    def save(model, path: str, architecture_desc : dict = None) -> str:
        """Save the compiled model into tvm file format."""
        import json
        if architecture_desc is None:
            architecture_desc = {}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        path_lib = path + "model_tvm.tar"
        model.export_library(path_lib)
        with open(path + "model_architecture_desc.json", "wt") as fp:
            json.dump(architecture_desc, fp)
        return path_lib

    @staticmethod
    def load(path: str) -> TabularNeuralNetTorchTvmPredictor:
        """Load from the path that contains an tvm file."""
        import tvm, json
        path_lib = path + "model_tvm.tar"
        loaded_lib = tvm.runtime.load_module(path_lib)
        with open(path + "model_architecture_desc.json", "rt") as fp:
            architecture_desc = json.load(fp)
        return TabularNeuralNetTorchTvmPredictor(model=loaded_lib,
                                                 architecture_desc=architecture_desc)
