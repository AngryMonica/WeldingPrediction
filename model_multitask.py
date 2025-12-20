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


def save_config(config, exp_dir):
    """保存配置"""
    config_path = os.path.join(exp_dir, "config.json")
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
        self.trunk_train_raw = self.trunk_input[train_idx]
        self.labels_train_raw = {task: labels[train_idx] for task, labels in self.labels_dict.items()}

        self.branch_test_raw = self.branch_input[test_idx]
        self.trunk_test_raw = self.trunk_input[test_idx]
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
        """数据缩放"""
        # Branch输入缩放
        if self.scale_method == 'standard':
            self.branch_scaler = StandardScaler()
            self.branch_train = self.branch_scaler.fit_transform(self.branch_train_raw)
            self.branch_test = self.branch_scaler.transform(self.branch_test_raw)
            self.scale_params['branch'] = {
                'type': 'standard', 'mean': self.branch_scaler.mean_, 'std': self.branch_scaler.scale_
            }
        elif self.scale_method == 'minmax':
            self.branch_scaler = MinMaxScaler()
            self.branch_train = self.branch_scaler.fit_transform(self.branch_train_raw)
            self.branch_test = self.branch_scaler.transform(self.branch_test_raw)
            self.scale_params['branch'] = {
                'type': 'minmax', 'min': self.branch_scaler.data_min_, 'max': self.branch_scaler.data_max_
            }

        # Trunk输入通常不缩放
        # self.trunk_train = self.trunk_train_raw
        # self.trunk_test = self.trunk_test_raw

        # Trunk输入缩放
        if self.scale_method == 'standard':
            self.trunk_scaler = StandardScaler()
            self.trunk_train = self.trunk_scaler.fit_transform(self.trunk_train_raw)
            self.trunk_test = self.trunk_scaler.transform(self.trunk_test_raw)
            self.scale_params['trunk'] = {
                'type': 'standard', 'mean': self.trunk_scaler.mean_, 'std': self.trunk_scaler.scale_
            }
        elif self.scale_method == 'minmax':
            self.trunk_scaler = MinMaxScaler()
            self.trunk_train = self.trunk_scaler.fit_transform(self.trunk_train_raw)
            self.trunk_test = self.trunk_scaler.transform(self.trunk_test_raw)
            self.scale_params['trunk'] = {
                'type': 'minmax', 'min': self.trunk_scaler.data_min_, 'max': self.trunk_scaler.data_max_
            }

        # 标签缩放
        self.labels_train = {}
        self.labels_test = {}

        for task_name in self.task_names:
            y_train_raw = self.labels_train_raw[task_name]
            y_test_raw = self.labels_test_raw[task_name]

            if self.scale_method == 'standard':
                y_scaler = StandardScaler()
                y_train_scaled = y_scaler.fit_transform(y_train_raw.reshape(y_train_raw.shape[0], -1))
                y_test_scaled = y_scaler.transform(y_test_raw.reshape(y_test_raw.shape[0], -1))
                self.scale_params[task_name] = {
                    'type': 'standard', 'mean': y_scaler.mean_, 'std': y_scaler.scale_
                }
            elif self.scale_method == 'minmax':
                y_scaler = MinMaxScaler()
                y_train_scaled = y_scaler.fit_transform(y_train_raw.reshape(y_train_raw.shape[0], -1))
                y_test_scaled = y_scaler.transform(y_test_raw.reshape(y_test_raw.shape[0], -1))
                self.scale_params[task_name] = {
                    'type': 'minmax', 'min': y_scaler.data_min_, 'max': y_scaler.data_max_
                }

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

        # 准备数据
        branch_train, trunk_train, labels_train = self.data.get_train_data(scaled=True)

        branch_tensor = torch.tensor(branch_train, dtype=torch.float32).to(self.device)
        trunk_tensor = torch.tensor(trunk_train, dtype=torch.float32).to(self.device)
        labels_tensor = {
            task: torch.tensor(labels, dtype=torch.float32).to(self.device)
            for task, labels in labels_train.items()
        }
        generator = torch.Generator(device=self.device)  # 确保生成器的设备与模型一致

        dataset = TensorDataset(branch_tensor, trunk_tensor, *labels_tensor.values())
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
            avg_loss = epoch_total_loss / len(dataloader)
            self.losshistory.steps.append(epoch)
            self.losshistory.loss_train.append(avg_loss)
            self.losshistory.loss_test.append(avg_loss)

            # 回调
            for callback in callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    callback.on_epoch_end(epoch, avg_loss, self.model, self)

            if epoch % display_every == 0:
                print(f'Epoch {epoch}: Loss: {avg_loss:.6f}')
                for task, loss in epoch_task_losses.items():
                    print(f'  {task}: {loss / len(dataloader):.6f}')

        return self.losshistory, self.train_state


class MultiTaskTensorBoardCallback:
    def __init__(self, log_dir, log_freq=10):
        self.writer = SummaryWriter(log_dir)
        self.log_freq = log_freq
        self.step = 0

    def on_epoch_end(self, epoch, loss, model, trainer):
        if self.step % self.log_freq == 0:
            self.writer.add_scalar('Loss/total', loss, self.step)
            self.step += 1

    def on_train_end(self):
        self.writer.close()


# ===========================================================
# 主函数
# ===========================================================

def main():
    args = parse_args("D:/Users/MXY/PycharmProjects/data/config.yaml")

    # 创建实验目录
    exp_dir, log_dir, ckpt_dir, output_dir = create_experiment_dir(
        args.base_dir, args.exp_name
    )
    save_config(vars(args), exp_dir)

    # 加载数据
    branch_df, trunk_df, temp_df, stress_df = load_data(
        args.branch_file, args.trunk_file, args.temp_file, args.stress_file
    )

    # 准备多任务数据
    data, pca_obj = prepare_multitask_data(
        branch_df, trunk_df, temp_df, stress_df,  # stress_df设为None
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
    callbacks = [MultiTaskTensorBoardCallback(log_dir, args.log_freq)]

    # 训练模型
    trainer = MultiTaskTrainer(model, data, lr=args.lr)
    losshistory, train_state = trainer.train(
        iterations=args.epochs,
        batch_size=args.batch_size or 32,
        callbacks=callbacks,
        display_every=args.display_every
    )

    print("训练完成!")
    return losshistory, train_state, model, data


if __name__ == "__main__":
    main()