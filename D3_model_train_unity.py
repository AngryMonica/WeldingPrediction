import deepxde as dde
import torch
import torch.nn as nn


class MultiTaskDeepONet(dde.nn.DeepONetCartesianProd):
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, activation,
                 kernel_initializer, tasks_config, is_output_activation=False,
                 output_activation="relu", multi_output_strategy="split_branch",
                 isDropout=False, dropout_rate=0.1, is_bias=True, init_bias=0.0):

        self.tasks_config = tasks_config
        self.task_names = list(tasks_config.keys())
        self.num_outputs = sum([config['output_dim'] for config in tasks_config.values()])

        super().__init__(
            layer_sizes_branch, layer_sizes_trunk, activation, kernel_initializer,
            self.num_outputs, multi_output_strategy
        )

        self.output_activation = self._get_activation(output_activation)
        self.is_output_activation = is_output_activation
        self.isDropout = isDropout
        self.branch_dropout = nn.Dropout(dropout_rate)
        self.trunk_dropout = nn.Dropout(dropout_rate)
        self.is_bias = is_bias

        if is_bias:
            self.output_bias = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32))

        self._setup_task_indices()

    def _get_activation(self, name):
        activations = {
            "relu": nn.ReLU(), "softplus": nn.Softplus(), "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(), "elu": nn.ELU(), "gelu": nn.GELU(), "linear": lambda x: x
        }
        return activations.get(name.lower(), nn.ReLU())

    def _setup_task_indices(self):
        self.task_output_indices = {}
        start_idx = 0
        for task_name, config in self.tasks_config.items():
            output_dim = config['output_dim']
            self.task_output_indices[task_name] = slice(start_idx, start_idx + output_dim)
            start_idx += output_dim

    def forward(self, inputs):
        x_func, x_loc = inputs[0], inputs[1]

        if self.isDropout:
            x_func = self.branch_dropout(x_func)
            x_loc = self.trunk_dropout(x_loc)

        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)

        x = self.multi_output_strategy.call(x_func, x_loc)

        if self.is_output_activation:
            x = self.output_activation(x)
        if self.is_bias:
            x = x + self.output_bias

        # 分割输出
        outputs = {}
        for task_name, output_slice in self.task_output_indices.items():
            outputs[task_name] = x[..., output_slice]

        if self._output_transform is not None:
            outputs = self._output_transform(inputs, outputs)

        return outputs