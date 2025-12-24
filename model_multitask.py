import os
import json
import yaml
import deepxde as dde
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
import shutil
import warnings

# 设置后端和设备
dde.backend.set_default_backend("pytorch")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================
# 配置管理
# ===========================================================

def create_experiment_dir(base_dir="runs", exp_name=None):
    """创建实验目录结构"""
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

    if exp_name is None:
        exp_name = timestamp
    else:
        exp_name = f"{timestamp}_{exp_name}"

    exp_dir = os.path.join(base_dir, exp_name)
    log_dir = os.path.join(exp_dir, "logs")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    output_dir = os.path.join(exp_dir, "outputs")

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"创建实验目录: {exp_dir}")
    return exp_dir, log_dir, ckpt_dir, output_dir


def save_config(config, exp_dir, scale_params=None):
    """保存配置"""
    config_path = os.path.join(exp_dir, "config.json")
    # 转换 scale_params 中的 numpy 数组为列表
    if scale_params is not None:
        config['scale_params'] = {k: float(v) for k, v in scale_params.items()}
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"配置已保存到: {config_path}")


def load_config(config_file):
    """从YAML文件加载配置"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_args(config_yaml_path):
    """解析命令行参数"""
    # 第一步：解析配置文件路径
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=config_yaml_path)
    config_args, _ = config_parser.parse_known_args()

    # 默认配置
    default_config = {
        'experiment': {
            'base_dir': 'runs', 'exp_name': None
        },
        'data': {
            'branch_file': 'branch_net_K_sampled.csv',
            'trunk_file': 'trunk_net_K_sampled.csv',
            'temp_file': 'all_temperature.csv',
            'stress_file': 'all_stress.csv',
            'isPCA': False, 'pca_dim': 512, 'test_size': 0.2,
            'random_state': 42, 'scaling_method': "standard"
        },
        'model': {
            'branch_layers': [256, 256, 128], 'trunk_layers': [64, 64, 128],
            'activation': 'tanh', 'initializer': 'Glorot normal',
            'tasks_config': {
                'temperature': {'output_dim': 1, 'loss_weight': 1.0, 'loss_type': 'mse'},
                'stress': {'output_dim': 1, 'loss_weight': 1.0, 'loss_type': 'mse'}
            },
            'multi_output_strategy': "split_branch",
            'is_output_activation': False, 'output_activation': 'relu',
            'isDropout': False, 'dropout_rate': 0.1, 'is_bias': True
        },
        'training': {
            'optimizer': 'adamw', 'lr': 0.0001, 'weight_decay': 1e-3,
            'decay': None, 'epochs': 10000, 'batch_size': None, 'display_every': 1
        },
        'callbacks': {
            'log_freq': 1, 'save_freq': 1000,
            'metrics': ["mean l2 relative error", "MAPE", "MAE", "RMSE"]
        },
        'early_stopping': {
            'isES': False, 'monitor': 'loss_test', 'min_delta': 1e-5,
            'patience': 3000, 'baseline': 1e-5, 'start_from_epoch': 1000
        }
    }

    # 加载用户配置
    if config_args.config and os.path.exists(config_args.config):
        user_config = load_config(config_args.config)

        # 深度合并配置
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    update_dict(d[k], v)
                else:
                    d[k] = v

        update_dict(default_config, user_config)
        print("从配置文件加载参数")
    else:
        print(f"使用默认参数，配置文件 {config_args.config} 不存在")

    # 创建参数解析器
    parser = argparse.ArgumentParser(description="多任务DeepONet训练")

    # 实验配置
    exp_config = default_config['experiment']
    parser.add_argument("--base_dir", type=str, default=exp_config['base_dir'])
    parser.add_argument("--exp_name", type=str, default=exp_config['exp_name'])

    # 数据配置
    data_config = default_config['data']
    parser.add_argument("--branch_file", type=str, default=data_config['branch_file'])
    parser.add_argument("--trunk_file", type=str, default=data_config['trunk_file'])
    parser.add_argument("--temp_file", type=str, default=data_config['temp_file'])
    parser.add_argument("--stress_file", type=str, default=data_config['stress_file'])
    parser.add_argument("--isPCA", type=bool, default=data_config['isPCA'])
    parser.add_argument("--pca_dim", type=int, default=data_config['pca_dim'])
    parser.add_argument("--test_size", type=float, default=data_config['test_size'])
    parser.add_argument("--random_state", type=int, default=data_config['random_state'])
    parser.add_argument("--scaling_method", type=str, default=data_config['scaling_method'])

    # 模型配置
    model_config = default_config['model']
    parser.add_argument("--branch_layers", type=int, nargs="+", default=model_config['branch_layers'])
    parser.add_argument("--trunk_layers", type=int, nargs="+", default=model_config['trunk_layers'])
    parser.add_argument("--activation", type=str, default=model_config['activation'])
    parser.add_argument("--initializer", type=str, default=model_config['initializer'])
    parser.add_argument("--tasks_config", type=dict, default=model_config['tasks_config'])
    parser.add_argument("--multi_output_strategy", type=str, default=model_config['multi_output_strategy'])
    parser.add_argument("--is_output_activation", type=bool, default=model_config['is_output_activation'])
    parser.add_argument("--output_activation", type=str, default=model_config['output_activation'])
    parser.add_argument("--isDropout", type=bool, default=model_config['isDropout'])
    parser.add_argument("--dropout_rate", type=float, default=model_config['dropout_rate'])
    parser.add_argument("--is_bias", type=bool, default=model_config['is_bias'])

    # 训练配置
    training_config = default_config['training']
    parser.add_argument("--optimizer", type=str, default=training_config['optimizer'])
    parser.add_argument("--lr", type=float, default=training_config['lr'])
    parser.add_argument("--weight_decay", type=float, default=training_config['weight_decay'])
    parser.add_argument("--decay", type=str, nargs="+", default=training_config['decay'])
    parser.add_argument("--epochs", type=int, default=training_config['epochs'])
    parser.add_argument("--batch_size", type=int, default=training_config['batch_size'])
    parser.add_argument("--display_every", type=int, default=training_config['display_every'])

    # 回调配置
    callbacks_config = default_config['callbacks']
    parser.add_argument("--log_freq", type=int, default=callbacks_config['log_freq'])
    parser.add_argument("--save_freq", type=int, default=callbacks_config['save_freq'])
    parser.add_argument("--metrics", type=str, nargs="+", default=callbacks_config['metrics'])

    # 早停配置
    early_stopping_config = default_config['early_stopping']
    parser.add_argument("--isES", type=bool, default=early_stopping_config['isES'])
    parser.add_argument("--monitor", type=str, default=early_stopping_config['monitor'])
    parser.add_argument("--min_delta", type=float, default=early_stopping_config['min_delta'])
    parser.add_argument("--patience", type=int, default=early_stopping_config['patience'])
    parser.add_argument("--baseline", type=float, default=early_stopping_config['baseline'])
    parser.add_argument("--start_from_epoch", type=int, default=early_stopping_config['start_from_epoch'])

    args = parser.parse_args()

    # 处理学习率衰减
    if args.decay:
        if args.decay[0] == "exponential":
            args.decay = ("exponential", float(args.decay[1]))
        elif args.decay[0] == "cosine":
            args.decay = ("cosine", int(args.decay[1]), float(args.decay[2]))

    return args


# ===========================================================
# 数据加载和预处理
# ===========================================================

def load_data(branch_file, trunk_file, temp_file,stress_file):
    """加载数据文件"""
    trunk_df = pd.read_csv(trunk_file, index_col=0)
    print(f"[INFO] trunk_net 维度: {trunk_df.shape}")

    temp_df = pd.read_csv(temp_file)
    temp_df.drop(columns=['test', 'step', 'increment','step_time'], inplace=True)
    print(f"[INFO] 温度数据 {temp_file} 维度: {temp_df.shape}")

    stress_df=pd.read_csv(stress_file)
    stress_df.drop(columns=['test', 'step', 'increment','step_time'], inplace=True)
    print(f"[INFO] 应力数据 {stress_file} 维度: {stress_df.shape}")


    branch_df = pd.read_csv(branch_file, index_col=0)
    branch_df = branch_df.iloc[temp_df.index]
    print(f"[INFO] branch_net 维度: {branch_df.shape}")

    return branch_df, trunk_df, temp_df,stress_df


class MultiTaskOperatorData:
    """多任务数据类"""

    def __init__(self, branch_input, trunk_input, labels_dict,
                 test_size=0.2, random_state=42, scale='standard'):
        self.branch_input = branch_input
        self.trunk_input = trunk_input
        self.labels_dict = labels_dict
        self.task_names = list(labels_dict.keys())
        self.scale_method = scale

        if scale not in ['standard', 'minmax', 'none']:
            raise ValueError("scale参数必须是 'standard', 'minmax' 或 'none'")

        self._split_data(test_size, random_state)

    def _split_data(self, test_size, random_state):
        """划分训练/测试集并缩放"""
        n_samples = len(self.branch_input)
        indices = np.arange(n_samples)

        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)

        self.train_idx = train_idx
        self.test_idx = test_idx
        self.num_train = len(train_idx)
        self.num_test = len(test_idx)

        # 原始数据
        self.branch_train_raw = self.branch_input[train_idx]
        self.trunk_train_raw = self.trunk_input
        self.labels_train_raw = {task: labels[train_idx] for task, labels in self.labels_dict.items()}

        self.branch_test_raw = self.branch_input[test_idx]
        self.trunk_test_raw = self.trunk_input
        self.labels_test_raw = {task: labels[test_idx] for task, labels in self.labels_dict.items()}

        self.scale_params = {}

        if self.scale_method != 'none':
            self._scale_data()
        else:
            self.branch_train = self.branch_train_raw
            self.trunk_train = self.trunk_train_raw
            self.labels_train = self.labels_train_raw
            self.branch_test = self.branch_test_raw
            self.trunk_test = self.trunk_test_raw
            self.labels_test = self.labels_test_raw

    def _scale_data(self):
        """数据缩放 - 使用numpy计算"""
        eps = 1e-8  # 防止除零

        # Branch输入缩放
        if self.scale_method == 'standard':
            branch_mean = np.mean(self.branch_train_raw)
            branch_std = np.std(self.branch_train_raw) + eps
            self.branch_train = (self.branch_train_raw - branch_mean) / branch_std
            self.branch_test = (self.branch_test_raw - branch_mean) / branch_std
            self.scale_params['branch_mean'] = branch_mean
            self.scale_params['branch_std'] = branch_std
        elif self.scale_method == 'minmax':
            branch_min = np.min(self.branch_train_raw)
            branch_max = np.max(self.branch_train_raw)
            branch_range = branch_max - branch_min + eps
            self.branch_train = (self.branch_train_raw - branch_min) / branch_range
            self.branch_test = (self.branch_test_raw - branch_min) / branch_range
            self.scale_params['branch_min'] = branch_min
            self.scale_params['branch_max'] = branch_max

        # Trunk输入缩放
        if self.scale_method == 'standard':
            trunk_mean = np.mean(self.trunk_train_raw)
            trunk_std = np.std(self.trunk_train_raw) + eps
            self.trunk_train = (self.trunk_train_raw - trunk_mean) / trunk_std
            self.trunk_test = (self.trunk_test_raw - trunk_mean) / trunk_std
            self.scale_params['trunk_mean'] = trunk_mean
            self.scale_params['trunk_std'] = trunk_std

        elif self.scale_method == 'minmax':
            trunk_min = np.min(self.trunk_train_raw)
            trunk_max = np.max(self.trunk_train_raw)
            trunk_range = trunk_max - trunk_min + eps
            self.trunk_train = (self.trunk_train_raw - trunk_min) / trunk_range
            self.trunk_test = (self.trunk_test_raw - trunk_min) / trunk_range
            self.scale_params['trunk_min'] = trunk_min
            self.scale_params['trunk_max'] = trunk_max

        # 标签缩放
        self.labels_train = {}
        self.labels_test = {}

        for task_name in self.task_names:
            y_train_raw = self.labels_train_raw[task_name]
            y_test_raw = self.labels_test_raw[task_name]

            # 展平后计算统计量
            y_train_flat = y_train_raw.reshape(y_train_raw.shape[0], -1)
            y_test_flat = y_test_raw.reshape(y_test_raw.shape[0], -1)

            if self.scale_method == 'standard':
                y_mean = np.mean(y_train_flat)
                y_std = np.std(y_train_flat) + eps
                y_train_scaled = (y_train_flat - y_mean) / y_std
                y_test_scaled = (y_test_flat - y_mean) / y_std
                self.scale_params[f'{task_name}_mean'] = y_mean
                self.scale_params[f'{task_name}_std'] = y_std
            elif self.scale_method == 'minmax':
                y_min = np.min(y_train_flat)
                y_max = np.max(y_train_flat)
                y_range = y_max - y_min + eps
                y_train_scaled = (y_train_flat - y_min) / y_range
                y_test_scaled = (y_test_flat - y_min) / y_range
                self.scale_params[f'{task_name}_min'] =y_min
                self.scale_params[f'{task_name}_max'] =y_max

            # 恢复形状
            self.labels_train[task_name] = y_train_scaled.reshape(y_train_raw.shape)
            self.labels_test[task_name] = y_test_scaled.reshape(y_test_raw.shape)

    def get_train_data(self, scaled=True):
        if scaled:
            return self.branch_train, self.trunk_train, self.labels_train
        return self.branch_train_raw, self.trunk_train_raw, self.labels_train_raw

    def get_test_data(self, scaled=True):
        if scaled:
            return self.branch_test, self.trunk_test, self.labels_test
        return self.branch_test_raw, self.trunk_test_raw, self.labels_test_raw


def prepare_multitask_data(branch_df, trunk_df, temp_df, stress_df,
                           isPCA=False, pca_dim=512, test_size=0.2, random_state=42,
                           scale="standard"):
    """准备多任务数据"""
    # 转换数据
    X_branch = branch_df.to_numpy(dtype=np.float32)
    X_trunk = trunk_df.to_numpy(dtype=np.float32)
    y_temp = temp_df.to_numpy(dtype=np.float32)

    # PCA降维
    pca_obj = None
    if isPCA:
        pca_obj = PCA(n_components=pca_dim)
        X_branch = pca_obj.fit_transform(X_branch)
        print(f"[INFO] PCA后维度: {X_branch.shape}")

    # 准备标签字典
    labels_dict = {'temperature': y_temp}
    if stress_df is not None:
        y_stress = stress_df.to_numpy(dtype=np.float32)
        labels_dict['stress'] = y_stress

    # 确保trunk是3D格式
    # if X_trunk.ndim == 2:
    #     X_trunk = np.tile(X_trunk[np.newaxis, :, :], (X_branch.shape[0], 1, 1))

    # 创建数据对象
    data = MultiTaskOperatorData(
        branch_input=X_branch,
        trunk_input=X_trunk,
        labels_dict=labels_dict,
        test_size=test_size,
        random_state=random_state,
        scale=scale
    )

    print(f"[INFO] 训练样本: {data.num_train}, 测试样本: {data.num_test}")
    return data, pca_obj


# ===========================================================
# 模型定义
# ===========================================================

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


class MultiTaskLoss(nn.Module):
    def __init__(self, tasks_config):
        super().__init__()
        self.tasks_config = tasks_config
        self.loss_fns = {}

        for task_name, config in tasks_config.items():
            loss_type = config.get('loss_type', 'mse')
            if loss_type == 'mse':
                self.loss_fns[task_name] = nn.MSELoss()
            elif loss_type == 'l1':
                self.loss_fns[task_name] = nn.L1Loss()

    def forward(self, predictions, targets):
        total_loss = 0
        task_losses = {}

        for task_name in self.tasks_config.keys():
            if task_name in predictions and task_name in targets:
                loss = self.loss_fns[task_name](predictions[task_name], targets[task_name])
                weight = self.tasks_config[task_name].get('loss_weight', 1.0)
                total_loss += weight * loss
                task_losses[task_name] = loss.item()

        return total_loss, task_losses


# ===========================================================
# 训练组件
# ===========================================================

class MultiTaskTrainer:
    def __init__(self, model, data, lr=0.001):
        self.model = model
        self.data = data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        self.loss_fn = MultiTaskLoss(model.tasks_config)

        self.losshistory = type('LossHistory', (), {'steps': [], 'loss_train': [], 'loss_test': []})()
        self.train_state = type('TrainState', (), {
            'epoch': 0, 'loss': float('inf'), 'best_loss': float('inf'), 'best_step': 0
        })()

    def train(self, iterations, batch_size=32, callbacks=None, display_every=100):
        callbacks = callbacks or []

        # 绑定回调的 trainer 引用并触发 on_train_begin
        for cb in callbacks:
            try:
                if hasattr(cb, 'set_trainer'):
                    cb.set_trainer(self)
                if hasattr(cb, 'on_train_begin'):
                    cb.on_train_begin()
            except Exception:
                pass
        # 准备数据
        branch_train, trunk_train, labels_train = self.data.get_train_data(scaled=True)

        branch_tensor = torch.tensor(branch_train, dtype=torch.float32).to(self.device)
        trunk_tensor = torch.tensor(trunk_train, dtype=torch.float32).to(self.device)
        # 按 model.task_names 顺序构造标签张量列表和字典
        label_tensors_list = []
        labels_tensor_dict = {}
        for task in self.model.task_names:
            lab = labels_train[task]
            t = torch.tensor(lab, dtype=torch.float32).to(self.device)
            label_tensors_list.append(t)
            labels_tensor_dict[task] = t

        generator = torch.Generator(device=self.device)  # 确保生成器的设备与模型一致

        dataset = TensorDataset(branch_tensor, trunk_tensor, *label_tensors_list)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

        # 训练循环
        for epoch in range(iterations):
            self.model.train()
            epoch_total_loss = 0
            epoch_task_losses = {task: 0 for task in self.model.task_names}

            for batch in dataloader:
                branch_batch, trunk_batch, *label_batches = batch
                labels_batch = {
                    task: label_batches[i] for i, task in enumerate(self.model.task_names)
                }

                predictions = self.model([branch_batch, trunk_batch])
                total_loss, task_losses = self.loss_fn(predictions, labels_batch)

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                epoch_total_loss += total_loss.item()
                for task, loss in task_losses.items():
                    epoch_task_losses[task] += loss

            # 记录损失
            avg_loss = epoch_total_loss / max(1, len(dataloader))
            self.losshistory.steps.append(epoch)
            self.losshistory.loss_train.append(avg_loss)
            # 计算并记录测试集损失（如果存在测试样本）
            test_loss_value = None
            if getattr(self.data, "num_test", 0) > 0:
                self.model.eval()
                branch_test, trunk_test, labels_test = self.data.get_test_data(scaled=True)
                with torch.no_grad():
                    b_test_tensor = torch.tensor(branch_test, dtype=torch.float32).to(self.device)
                    t_test_tensor = torch.tensor(trunk_test, dtype=torch.float32).to(self.device)
                    # 构建测试标签字典与模型任务顺序一致
                    labels_test_tensors = {
                        task: torch.tensor(labels_test[task], dtype=torch.float32).to(self.device)
                        for task in self.model.task_names
                    }
                    preds_test = self.model([b_test_tensor, t_test_tensor])
                    test_total_loss, _ = self.loss_fn(preds_test, labels_test_tensors)
                    test_loss_value = test_total_loss.item()
                    self.losshistory.loss_test.append(test_loss_value)
            else:
                # 没有测试集时保持与训练长度一致的占位
                self.losshistory.loss_test.append(avg_loss)

            # 更新训练状态
            self.train_state.epoch = epoch
            self.train_state.loss = avg_loss
            if avg_loss < self.train_state.best_loss:
                self.train_state.best_loss = avg_loss
                self.train_state.best_step = epoch

            # 回调 on_epoch_end
            for cb in callbacks:
                try:
                    if hasattr(cb, 'on_epoch_end'):
                        cb.on_epoch_end()
                except Exception:
                    pass

            if epoch % display_every == 0:
                if test_loss_value is not None:
                    print(f'Epoch {epoch}: Train Loss: {avg_loss:.6f}, Test Loss: {test_loss_value:.6f}')
                else:
                    print(f'Epoch {epoch}: Train Loss: {avg_loss:.6f}')
                for task, loss in epoch_task_losses.items():
                    print(f'  {task}: {loss / max(1, len(dataloader)):.6f}')

            # 训练结束回调
        for cb in callbacks:
            try:
                if hasattr(cb, 'on_train_end'):
                    cb.on_train_end()
            except Exception:
                pass

        return self.losshistory, self.train_state


# python
class TensorBoardCallback(dde.callbacks.Callback):
    def __init__(self, writer, log_freq=1, ckpt_dir=None, save_freq=1, scale_params=None):
        super().__init__()
        self.writer = writer
        self.log_freq = log_freq
        self.ckpt_dir = ckpt_dir
        self.save_freq = save_freq
        self.scale_params = scale_params
        self.step = 0
        self.best_loss = float('inf')
        self.trainer = None
        self.model_ref = None
        self.data = None

    def set_trainer(self, trainer):
        self.trainer = trainer
        try:
            self.model_ref = trainer.model
        except Exception:
            self.model_ref = None
        try:
            self.data = trainer.data
        except Exception:
            self.data = None

    def on_train_begin(self):
        print("训练开始，TensorBoard记录已启用")
        try:
            if self.model_ref is not None and hasattr(self.model_ref, "net"):
                dummy_branch = torch.zeros(1, self.model_ref.net.branch.linears[0].in_features)
                dummy_trunk = torch.zeros(1, self.model_ref.net.trunk.linears[0].in_features)
                self.writer.add_graph(self.model_ref, (dummy_branch, dummy_trunk))
        except Exception as e:
            print(f"记录模型图结构时出错: {e}")

    def _unscale_array(self, arr, task_name):
        """更鲁棒的反缩放：先把 arr 转为 numpy，根据缩放参数重塑到 (-1, feature_dim) 再反缩放，最后恢复形状"""
        arr = np.asarray(arr)
        sp = None
        if self.data is not None and hasattr(self.data, "scale_params"):
            sp = self.data.scale_params.get(task_name, None)
        if sp is None and self.scale_params is not None:
            sp = self.scale_params.get(task_name, None)
        if sp is None:
            return arr

        stype = sp.get("type", "standard")

        def _apply_unscale(arr_np, param_a, param_b=None, mode="standard"):
            # param_a: mean or min, param_b: std or max
            mean_like = np.asarray(param_a)
            if mode == "standard":
                std_like = np.asarray(param_b)
                # if scalar mean/std, broadcast directly
                if mean_like.ndim == 0:
                    return arr_np * std_like + mean_like
                # 如果最后一维与 mean 长度匹配，直接按最后维度反缩放
                if arr_np.ndim >= 1 and arr_np.shape[-1] == mean_like.size:
                    return arr_np * std_like + mean_like
                # 如果 arr 为一维且能整除 mean 长度，reshape 后反缩放
                if arr_np.ndim == 1 and arr_np.size % mean_like.size == 0:
                    orig_shape = arr_np.shape
                    arr2 = arr_np.reshape(-1, mean_like.size)
                    res = arr2 * std_like + mean_like
                    return res.reshape(orig_shape)
                # 最后尝试广播（若失败返回原数组）
                try:
                    return arr_np * std_like + mean_like
                except Exception:
                    return arr_np
            else:  # minmax
                vmin = mean_like
                vmax = np.asarray(param_b)
                scale = (vmax - vmin)
                if vmin.ndim == 0:
                    return arr_np * scale + vmin
                if arr_np.ndim >= 1 and arr_np.shape[-1] == vmin.size:
                    return arr_np * scale + vmin
                if arr_np.ndim == 1 and arr_np.size % vmin.size == 0:
                    orig_shape = arr_np.shape
                    arr2 = arr_np.reshape(-1, vmin.size)
                    res = arr2 * scale + vmin
                    return res.reshape(orig_shape)
                try:
                    return arr_np * scale + vmin
                except Exception:
                    return arr_np

        if stype == "standard":
            mean = sp.get("mean", 0.0)
            std = sp.get("std", 1.0)
            return _apply_unscale(arr, mean, std, mode="standard")
        elif stype == "minmax":
            vmin = sp.get("min", 0.0)
            vmax = sp.get("max", 1.0)
            return _apply_unscale(arr, vmin, vmax, mode="minmax")
        else:
            return arr

    def on_epoch_end(self):
        if self.step % self.log_freq != 0:
            self.step += 1
            return

        try:
            if self.trainer is not None and hasattr(self.trainer, "losshistory"):
                if len(self.trainer.losshistory.loss_train) > 0:
                    loss_train = self.trainer.losshistory.loss_train[-1]
                    self.writer.add_scalar('Loss/Train', loss_train, self.step)
                if len(self.trainer.losshistory.loss_test) > 0:
                    loss_test = self.trainer.losshistory.loss_test[-1]
                    self.writer.add_scalar('Loss/Test', loss_test, self.step)

            if self.model_ref is not None and self.data is not None:
                branch_test, trunk_test, labels_test = self.data.get_test_data(scaled=True)
                self.model_ref.eval()
                with torch.no_grad():
                    b_tensor = torch.tensor(branch_test, dtype=torch.float32).to(
                        self.model_ref.parameters().__next__().device)
                    t_tensor = torch.tensor(trunk_test, dtype=torch.float32).to(
                        self.model_ref.parameters().__next__().device)
                    preds = self.model_ref([b_tensor, t_tensor])

                for task in self.model_ref.task_names:
                    if task not in preds or task not in labels_test:
                        continue
                    y_pred_scaled = preds[task].cpu().numpy()
                    y_true_scaled = np.asarray(labels_test[task])

                    # 如果最后一个维度是 1（单通道），去掉该维度
                    if y_pred_scaled.ndim >= 3 and y_pred_scaled.shape[-1] == 1:
                        y_pred_scaled = np.squeeze(y_pred_scaled, axis=-1)
                    if y_true_scaled.ndim >= 3 and y_true_scaled.shape[-1] == 1:
                        y_true_scaled = np.squeeze(y_true_scaled, axis=-1)

                    # 反缩放（函数内会自动处理 reshape）
                    y_pred = self._unscale_array(y_pred_scaled, task)
                    y_true = self._unscale_array(y_true_scaled, task)

                    y_true_flat = y_true.reshape(-1)
                    y_pred_flat = y_pred.reshape(-1)

                    mse = np.mean((y_true_flat - y_pred_flat) ** 2)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
                    denom = np.where(np.abs(y_true_flat) < 1e-8, 1e-8, np.abs(y_true_flat))
                    mape = np.mean(np.abs((y_true_flat - y_pred_flat) / denom)) * 100.0

                    self.writer.add_scalar(f"Unscaled/{task}/RMSE", rmse, self.step)
                    self.writer.add_scalar(f"Unscaled/{task}/MAE", mae, self.step)
                    self.writer.add_scalar(f"Unscaled/{task}/MAPE", mape, self.step)

                    try:
                        max_points = 10000
                        idx = np.random.choice(len(y_pred_flat), min(len(y_pred_flat), max_points), replace=False)
                        self.writer.add_histogram(f"{task}/pred", y_pred_flat[idx], self.step)
                        self.writer.add_histogram(f"{task}/true", y_true_flat[idx], self.step)
                    except Exception:
                        pass

            if self.trainer is not None and hasattr(self.trainer, "optimizer"):
                try:
                    lr = self.trainer.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar("LearningRate", lr, self.step)
                except Exception:
                    pass

        except Exception as e:
            print(f"记录TensorBoard数据时出错: {e}")

        self.step += 1

    def on_train_end(self):
        print("训练结束，关闭TensorBoard写入器")
        try:
            self.writer.close()
        except Exception:
            pass



# ===========================================================
# 主函数
# ===========================================================

def main():
    args = parse_args("D:/Users/MXY/PycharmProjects/data/config.yaml")

    # 创建实验目录
    exp_dir, log_dir, ckpt_dir, output_dir = create_experiment_dir(
        args.base_dir, args.exp_name
    )


    # 加载数据
    branch_df, trunk_df, temp_df, stress_df = load_data(
        args.branch_file, args.trunk_file, args.temp_file, args.stress_file
    )

    # 准备多任务数据
    data, pca_obj = prepare_multitask_data(
        branch_df, trunk_df, temp_df, stress_df,
        isPCA=args.isPCA, pca_dim=args.pca_dim, test_size=args.test_size,
        random_state=args.random_state, scale=args.scaling_method
    )

    # 创建模型
    branch_dim = data.branch_train.shape[1]
    trunk_dim = data.trunk_train.shape[-1]  # 坐标维度

    model = MultiTaskDeepONet(
        layer_sizes_branch=[branch_dim] + args.branch_layers,
        layer_sizes_trunk=[trunk_dim] + args.trunk_layers,
        activation=args.activation,
        kernel_initializer=args.initializer,
        tasks_config=args.tasks_config,
        is_output_activation=args.is_output_activation,
        output_activation=args.output_activation,
        multi_output_strategy=args.multi_output_strategy,
        isDropout=args.isDropout,
        dropout_rate=args.dropout_rate,
        is_bias=args.is_bias
    )

    # 创建回调
    # callbacks = [MultiTaskTensorBoardCallback(log_dir, args.log_freq)]

    writer=SummaryWriter(log_dir=log_dir)
    # 创建回调
    tensorboard_callback = TensorBoardCallback(
        writer,
        log_freq=args.log_freq,
        ckpt_dir=ckpt_dir,
        save_freq=args.save_freq,
        scale_params=data.scale_params
    )

    early_stopping = dde.callbacks.EarlyStopping(
        monitor=args.monitor,
        min_delta=args.min_delta,
        patience=args.patience,
        baseline=args.baseline,
        start_from_epoch=args.start_from_epoch
    )
    # 训练模型
    trainer = MultiTaskTrainer(model, data, lr=args.lr)
    losshistory, train_state = trainer.train(
        iterations=args.epochs,
        batch_size=args.batch_size or 32,
        callbacks=[tensorboard_callback,early_stopping],
        display_every=args.display_every
    )
    path = os.path.join(output_dir, "final_model.pt")
    torch.save(model.state_dict(), path)
    print(f"训练好的模型已保存到: {path}")

    # 保存配置（包含正确格式的缩放参数）
    config = vars(args).copy()
    save_config(config, exp_dir, scale_params=data.scale_params)


    print("训练完成!")
    return losshistory, train_state, model, data


if __name__ == "__main__":
    main()