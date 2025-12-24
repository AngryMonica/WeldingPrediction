import os
import json
import deepxde as dde
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
import shutil
import torch.nn as nn
import optuna


def create_optuna_objective(args, log_dir):
    """创建Optuna优化目标函数"""

    def objective(trial):
        #data
        # branch_file=trial.suggest_categorical('branch_file', ["branch_net_K_sampled.csv","herstory_branch_net_K_sampled.csv"])
        branch_file="herstory_branch_net_K_sampled.csv"
        # isPCA=trial.suggest_categorical('isPCA', [True,False])
        isPCA=False
        # pca_dim = trial.suggest_int('pca_dim', 64, 3000, step=32)
        pca_dim=512
        # scaling_method=trial.suggest_categorical('scaling_method',['standard','minmax','none'])

        # model
        p = trial.suggest_int('output_dim', 32, 512, step=32)  # 例如从64到512，步长为32 # 0. 首先建议一个共享的输出维度 p,这个p将是branch net和trunk net最后一层的神经元数量

        # 定义分支网络（Branch Net）的超参数
        branch_num_layers = trial.suggest_int('branch_num_layers', 2, 5)
        branch_layers = []
        for i in range(branch_num_layers - 1):  # 注意：最后一层我们固定为p，所以这里少一层
            # 建议每一层的神经元数量，但最后一层尚未确定
            hidden_units = trial.suggest_int(f'branch_layer_{i}', 32, 512, step=32)
            branch_layers.append(hidden_units)
        # 分支网络的最后一层就是输出层，其大小必须为 p
        branch_layers.append(p)  # 这是关键约束！

        # 定义主干网络（Trunk Net）的超参数
        trunk_num_layers = trial.suggest_int('trunk_num_layers', 2, 5)
        trunk_layers = []
        for i in range(trunk_num_layers - 1):
            hidden_units = trial.suggest_int(f'trunk_layer_{i}', 32, 256, step=32)
            trunk_layers.append(hidden_units)
        # 主干网络的最后一层也必须是 p
        trunk_layers.append(p)  # 这是关键约束！

        # model'
        activation = trial.suggest_categorical('activation',
                                               ['tanh', 'relu',  'sigmoid', 'gelu',  'elu'])
        initializer = trial.suggest_categorical('initializer',
                                                ['Glorot normal', 'Glorot uniform', 'He normal', 'He uniform'])
        # is_output_activation = trial.suggest_categorical('is_output_activation', [True, False])
        is_output_activation=args.is_output_activation
        output_activation = trial.suggest_categorical('output_activation', ['relu', 'sigmoid', 'relu6', 'softplus'])
        isDropout = trial.suggest_categorical('isDropout', [True, False])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3, step=0.1)
        is_bias=trial.suggest_categorical('is_bias', [True,False])

        # training
        epochs=trial.suggest_int('epochs', 10000,50000,step=10000)
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'rmsprop'])
        # optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'rmsprop','L-BFGS','NNCG','L-BFGS-B'])
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True) # 1. 定义超参数搜索空间 [2,3](@ref)
        weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-4, log=True)
        decay_name_list=['exponential', 'cosine', 'step','inverse time']
        decay_name= trial.suggest_categorical('decay_name',decay_name_list)

        if decay_name == "exponential":
            gamma=trial.suggest_float('gamma',0.9,0.999,step=0.09)
            decay=("exponential",gamma)
        elif decay_name == "cosine":
            T_max=trial.suggest_float('T_max',lr * 1e-3,lr * 1e-2,step=lr * 2e-3)
            eta_min=trial.suggest_int('eta_min',epochs/10,epochs/2,step=epochs/10)
            decay=("cosine", T_max, eta_min)
        elif decay_name == "step":
            step_size=trial.suggest_int('step_size',epochs/2,epochs,step=epochs/10)
            gamma=trial.suggest_float('gamma',0.5,0.9,step=0.1)
            decay =("step", step_size, gamma)
        elif decay_name == "inverse time":
            decay_steps=trial.suggest_int('decay_steps',epochs/2,epochs,step=epochs/10)
            decay_rate=trial.suggest_float('decay_rate',0.001,0.1,step=0.01)
            decay=("inverse time", decay_steps, decay_rate)
        else:
            decay=None

        # early_stopping
        isES= trial.suggest_categorical('isES',[True,False])
        min_delta = trial.suggest_float('min_delta', 1e-8, 1e-6, step=1e-7)
        patience = trial.suggest_int('patience', 2000, 10000, step=2000)
        baseline = trial.suggest_float('baseline', 1e-6, 1e-5, step=2e-6)
        start_from_epoch = trial.suggest_int('start_from_epoch', 2000, 6000, step=1000)

        # 加载数据
        branch_df, trunk_df, temp_df, temp_df_with_titles = load_data(
            branch_file,
            args.trunk_file,
            args.temp_file
        )

        # 准备数据
        data, X_branch, scale_params, X_branch_test_scaled, X_trunk_test, y_test = prepare_data(
            branch_df,
            trunk_df,
            temp_df,
            isPCA=isPCA,
            pca_dim=pca_dim,
            test_size=args.test_size,
            random_state=args.random_state,
            scaling_method=args.scaling_method
        )
        print("🎯 使用Optuna进行超参数优化")


        # 2. 创建模型 [5,7](@ref)
        model = create_model(
            data,
            branch_dim=X_branch.shape[1],
            trunk_dim=3,
            branch_layers=branch_layers,
            trunk_layers=trunk_layers,
            activation=activation,
            initializer=initializer,
            is_output_activation=is_output_activation,
            output_activation=output_activation,
            isDropout=isDropout,
            dropout_rate=dropout_rate,
            is_bias=is_bias,
            init_bias=scale_params.get("y_mean", 0.0)
        )

        # 3. 配置优化器
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                model.net.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
            model.compile(optimizer, lr=lr, metrics=args.metrics)
        else:
            model.compile(optimizer_name, lr=lr,decay=decay, metrics=args.metrics)

        # 4. 创建回调 - 为每个trial创建独立的日志目录
        trial_log_dir = os.path.join(log_dir, f"trial_{trial.number}")
        trial_writer = SummaryWriter(log_dir=trial_log_dir)

        early_stopping = dde.callbacks.EarlyStopping(
            monitor=args.monitor,
            min_delta=min_delta,
            patience=patience,
            baseline=baseline,
            start_from_epoch=start_from_epoch
        )

        # 5. 训练模型（减少epochs以加速搜索）
        epochs_per_trial = min(args.epochs, 1000)  # 限制每个trial的训练轮次

        try:
            if isES:
                losshistory, train_state = model.train(
                    iterations=epochs_per_trial,
                    display_every=args.display_every * 10,  # 减少输出频率
                    callbacks=[early_stopping],
                    verbose=0  # 减少输出
                )
            else:
                losshistory, train_state = model.train(
                    iterations=epochs_per_trial,
                    display_every=args.display_every * 10,  # 减少输出频率
                    verbose=0  # 减少输出
                )

            # 6. 返回验证损失作为优化目标 [1,2](@ref)
            best_test_loss = min([sum(loss) for loss in losshistory.loss_test])
            return best_test_loss

        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            return float('inf')  # 返回一个很大的值表示失败

    return objective


def run_optuna_optimization(args, log_dir):
    """运行Optuna超参数优化"""

    # 创建Optuna研究 [2,3](@ref)
    study = optuna.create_study(
        direction='minimize',  # 最小化验证损失
        sampler=optuna.samplers.TPESampler(seed=args.random_state),
        pruner=optuna.pruners.MedianPruner()  # 提前剪枝
    )

    # 创建目标函数
    objective = create_optuna_objective(
        args,log_dir
    )

    # 运行优化 [1](@ref)
    print("开始Optuna超参数优化...")
    study.optimize(
        objective,
        n_trials=200,  # 试验次数
        timeout=36000,  # 最大超时时间（秒）
        show_progress_bar=True
    )

    return study


def analyze_optuna_results(study, output_dir):
    """分析和可视化Optuna结果"""

    print("\n" + "=" * 60)
    print("Optuna超参数优化结果")
    print("=" * 60)

    # 输出最佳超参数 [2](@ref)
    print(f"最佳试验: #{study.best_trial.number}")
    print(f"最佳验证损失: {study.best_value:.6f}")
    print("最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # 保存结果
    results = {
        'best_trial_number': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'trials_dataframe': study.trials_dataframe().to_dict('records')
    }

    results_path = os.path.join(output_dir, "optuna_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4, default=str)  # 添加default=str

    # 可视化结果 [2,5](@ref)
    try:
        import optuna.visualization as vis

        # 优化历史
        fig_history = vis.plot_optimization_history(study)
        fig_history.write_image(os.path.join(output_dir, "optuna_optimization_history.png"))

        # 超参数重要性
        fig_importance = vis.plot_param_importances(study)
        fig_importance.write_image(os.path.join(output_dir, "optuna_param_importance.png"))

        # 平行坐标图
        fig_parallel = vis.plot_parallel_coordinate(study)
        fig_parallel.write_image(os.path.join(output_dir, "optuna_parallel_coordinate.png"))

        print(f"可视化结果已保存到: {output_dir}")

    except Exception as e:
        print(f"可视化生成失败: {e}")

    return study.best_params


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
    """保存配置，包括正确格式化的缩放参数"""
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
if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 创建实验目录
    exp_dir, log_dir, ckpt_dir, output_dir = create_experiment_dir(
        base_dir=args.base_dir,
        exp_name=args.exp_name
    )


    # 准备数据
    # data, X_branch, scale_params, X_branch_test_scaled, X_trunk_test, y_test

    # 运行Optuna优化
    study = run_optuna_optimization(
        args, output_dir
    )

    # 分析和保存结果
    best_params = analyze_optuna_results(study, output_dir)

    # 保存配置
    config = vars(args)
    save_config(config, exp_dir)

    print("\n✅ Optuna优化完成！")
    print("使用最佳超参数重新训练最终模型...")

