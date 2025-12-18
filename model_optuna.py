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

dde.backend.set_default_backend("pytorch")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================
# 实验配置和目录管理
# ===========================================================

def create_experiment_dir(base_dir="runs", exp_name=None):
    """创建实验目录结构

    Args:
        base_dir: 基础目录
        exp_name: 实验名称，如果为None则自动生成

    Returns:
        exp_dir: 实验目录路径
        log_dir: 日志目录路径
        ckpt_dir: 检查点目录路径
        output_dir: 输出目录路径
    """
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")

    # 如果没有提供实验名称，则直接使用时间戳
    # 如果提供了实验名称，则使用名称+时间戳
    if exp_name is None:
        exp_name = timestamp
    else:
        exp_name = f"{timestamp}_{exp_name}"

    # 创建实验目录结构
    exp_dir = os.path.join(base_dir, exp_name)
    log_dir = os.path.join(exp_dir, "logs")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    output_dir = os.path.join(exp_dir, "outputs")

    # 确保目录存在
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"创建实验目录: {exp_dir}")
    return exp_dir, log_dir, ckpt_dir, output_dir


def save_config(config, exp_dir):
    """保存当前实验的配置到 config.json"""
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"配置已保存到: {config_path}")

    # 同时保存一份代码备份
    src_file = os.path.abspath(__file__)
    dst_file = os.path.join(exp_dir, os.path.basename(__file__))
    shutil.copy2(src_file, dst_file)
    print(f"代码已备份到: {dst_file}")


def save_checkpoint(model, optimizer, epoch, ckpt_dir, is_best=False):
    """保存模型 checkpoint

    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮次
        ckpt_dir: 检查点保存目录
        is_best: 是否为最佳模型
    """
    ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, ckpt_path)

    # 如果是最佳模型，则额外保存一份
    if is_best:
        best_path = os.path.join(ckpt_dir, "best_model.pt")
        shutil.copy2(ckpt_path, best_path)


# ===========================================================
# 自定义TensorBoard回调函数
# ===========================================================
class TensorBoardCallback(dde.callbacks.Callback):
    def __init__(self, writer, log_freq=1, ckpt_dir=None, save_freq=1, scale_params=None):
        """
        参数:
            writer: SummaryWriter实例
            log_freq: 记录频率（每多少步记录一次）
            ckpt_dir: 检查点保存目录
            save_freq: 保存检查点频率
            scale_params: 用于反缩放的参数字典

        """
        super().__init__()
        self.writer = writer
        self.log_freq = log_freq
        self.ckpt_dir = ckpt_dir
        self.save_freq = save_freq
        self.scale_params = scale_params  # <<< 新增
        self.step = 0
        self.best_loss = float('inf')

    def on_train_begin(self):
        """训练开始时调用"""
        print("训练开始，TensorBoard记录已启用")

        # 记录模型结构图
        if hasattr(self.model, 'net') and hasattr(self.model.net, 'branch'):
            try:
                dummy_input_branch = torch.zeros(1, self.model.net.branch.linears[0].in_features)
                dummy_input_trunk = torch.zeros(1, self.model.net.trunk.linears[0].in_features)
                self.writer.add_graph(self.model, (dummy_input_branch, dummy_input_trunk))
            except Exception as e:
                print(f"记录模型图结构时出错: {str(e)}")

    def on_epoch_end(self):
        """每个训练步结束时调用"""
        # 每log_freq步记录一次
        if self.step % self.log_freq == 0:
            try:
                loss_train = self.model.losshistory.loss_train[-1]
                self.writer.add_scalar('Loss/Train', loss_train, self.step)

                loss_test = self.model.losshistory.loss_test[-1]
                self.writer.add_scalar('Loss/Test', loss_test, self.step)

                # 记录各项指标
                metrics_names = ["mean l2 relative error", "MAPE", "MAE", "RMSE"]
                for i, name in enumerate(metrics_names):
                    if i < len(self.model.losshistory.metrics_test[-1]):
                        metric_value = self.model.losshistory.metrics_test[-1][i]
                        self.writer.add_scalar(f"Scaled/{name}", metric_value, self.step)

                # === 新增部分: 反缩放后的指标计算 ===
                if self.scale_params is not None:
                    y_true_scaled = self.model.data.test_y
                    y_pred_scaled = self.model.predict(self.model.data.test_x)

                    sp = self.scale_params
                    if sp["type"] == "standard":
                        y_true = y_true_scaled * sp["y_std"] + sp["y_mean"]
                        y_pred = y_pred_scaled * sp["y_std"] + sp["y_mean"]
                    elif sp["type"] == "minmax":
                        y_true = y_true_scaled * (sp["y_max"] - sp["y_min"]) + sp["y_min"]
                        y_pred = y_pred_scaled * (sp["y_max"] - sp["y_min"]) + sp["y_min"]
                    else:
                        y_true, y_pred = y_true_scaled, y_pred_scaled

                    # 计算 RMSE / MAE / R²
                    mse = np.mean((y_true - y_pred) ** 2)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(y_true - y_pred))
                    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
                    ss_res = np.sum((y_true - y_pred) ** 2)
                    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                    r2 = 1 - ss_res / (ss_tot + 1e-8)

                    # 写入 TensorBoard
                    self.writer.add_scalar("Unscaled/MAPE", mape, self.step)
                    self.writer.add_scalar("Unscaled/RMSE", rmse, self.step)
                    self.writer.add_scalar("Unscaled/MAE", mae, self.step)
                    self.writer.add_scalar("Unscaled/R2", r2, self.step)

                # --- 网络权重（branch 和 trunk） ---
                # if hasattr(self.model.net, "branch"):
                #     for i, layer in enumerate(self.model.net.branch.linears):
                #         self.writer.add_histogram(f"Branch/Layer{i}/weights", layer.weight, self.step)
                #         self.writer.add_histogram(f"Branch/Layer{i}/bias", layer.bias, self.step)
                #
                # if hasattr(self.model.net, "trunk"):
                #     for i, layer in enumerate(self.model.net.trunk.linears):
                #         self.writer.add_histogram(f"Trunk/Layer{i}/weights", layer.weight, self.step)
                #         self.writer.add_histogram(f"Trunk/Layer{i}/bias", layer.bias, self.step)
                # 记录学习率
                lr = self.model.opt.param_groups[0]["lr"]
                self.writer.add_scalar("LearningRate", lr, self.step)

                # # 保存检查点
                # if self.ckpt_dir and self.step % self.save_freq == 0:
                #     save_checkpoint(
                #         self.model.net,
                #         self.model.opt,
                #         self.step,
                #         self.ckpt_dir
                #     )

                # 保存最佳模型
                # if loss_test < self.best_loss and self.ckpt_dir:
                #     self.best_loss = loss_test
                #     save_checkpoint(
                #         self.model.net,
                #         self.model.opt,
                #         self.step,
                #         self.ckpt_dir,
                #         is_best=True
                #     )

            except Exception as e:
                print(f"记录TensorBoard数据时出错: {str(e)}")
                # 即使出错也不中断训练
        self.step += 1

    def on_train_end(self):
        """训练结束时调用"""
        print("训练结束，关闭TensorBoard写入器")
        # 保存最终模型
        # if self.ckpt_dir:
        #     save_checkpoint(
        #         self.model.net,
        #         self.model.opt,
        #         self.step,
        #         self.ckpt_dir,
        #         is_best=False
        #     )
        self.writer.close()


# ===========================================================
# 命令行参数解析
# ===========================================================
def load_config(config_file):
    """从YAML配置文件加载参数

    Args:
        config_file: 配置文件路径

    Returns:
        config: 配置字典
    """
    import yaml

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def parse_args():
    """解析命令行参数，优先从配置文件导入参数配置

    Returns:
        args: 解析后的参数
    """
    # 第一步：先解析配置文件路径参数
    config_parser = argparse.ArgumentParser(description="DeepONet模型训练参数", add_help=False)
    config_parser.add_argument("--config", type=str, default="config.yaml", help="YAML配置文件路径")
    config_args, _ = config_parser.parse_known_args()

    # 初始化默认参数
    default_args = {}

    # 如果配置文件存在，从配置文件加载参数
    if config_args.config and os.path.exists(config_args.config):
        config = load_config(config_args.config)
        print("~~~~~~~~~~~~~~~~~~~~从配置文件加载了参数~~~~~~~~~~~~~~~~~~~~~~~~")

        # 从配置文件提取参数
        if 'experiment' in config:
            exp_config = config['experiment']
            default_args['base_dir'] = exp_config.get('base_dir', 'runs')
            default_args['exp_name'] = exp_config.get('exp_name', None)

        if 'data' in config:
            data_config = config['data']
            default_args['branch_file'] = data_config.get('branch_file', 'branch_net_K_sampled.csv')
            default_args['trunk_file'] = data_config.get('trunk_file', 'trunk_net_K_sampled.csv')
            default_args['temp_file'] = data_config.get('temp_file', 'merged_all_time_points_K_sampled.csv')
            default_args['isPCA'] = data_config.get('isPCA', False)
            default_args['pca_dim'] = data_config.get('pca_dim', 512)
            default_args['test_size'] = data_config.get('test_size', 0.2)
            default_args['random_state'] = data_config.get('random_state', 42)
            default_args['scaling_method'] = data_config.get('scaling_method', "standard")

        if 'model' in config:
            model_config = config['model']
            default_args['branch_layers'] = model_config.get('branch_layers', [256, 256, 128])
            default_args['trunk_layers'] = model_config.get('trunk_layers', [64, 64, 128])
            default_args['activation'] = model_config.get('activation', 'tanh')
            default_args['initializer'] = model_config.get('initializer', 'Glorot normal')
            default_args['is_output_activation'] = model_config.get('is_output_activation', False)
            default_args['output_activation'] = model_config.get('output_activation', 'relu')
            default_args['isDropout'] = model_config.get('isDropout', False)
            default_args['dropout_rate'] = model_config.get('dropout_rate', 0.1)
            default_args['is_bias'] = model_config.get('is_bias', True)


        if 'training' in config:
            training_config = config['training']
            default_args['optimizer'] = training_config.get('optimizer', 'adamw')
            default_args['lr'] = training_config.get('lr', 0.0001)
            default_args['weight_decay'] = training_config.get('weight_decay', 1e-3)
            default_args['decay'] = training_config.get('decay', None)
            default_args['epochs'] = training_config.get('epochs', 10000)
            default_args['batch_size'] = training_config.get('batch_size', None)
            default_args['display_every'] = training_config.get('display_every', 1)

        if 'callbacks' in config:
            callbacks_config = config['callbacks']
            default_args['log_freq'] = callbacks_config.get('log_freq', 1)
            default_args['save_freq'] = callbacks_config.get('save_freq', 1000)
            default_args['metrics'] = callbacks_config.get('metrics', ["mean l2 relative error", "MAPE", "MAE", "RMSE"])

        if 'early_stopping' in config:
            early_stopping_config = config['early_stopping']
            default_args['isES'] = early_stopping_config.get('isES', False)
            default_args['monitor'] = early_stopping_config.get('monitor', 'loss_test')
            default_args['min_delta'] = early_stopping_config.get('min_delta', 1e-5)
            default_args['patience'] = early_stopping_config.get('patience', 3000)
            default_args['baseline'] = early_stopping_config.get('baseline', 1e-5)
            default_args['start_from_epoch'] = early_stopping_config.get('start_from_epoch', 1000)
    else:
        print(f"警告: 配置文件 {config_args.config} 不存在，将使用默认参数")
        # 设置默认值
        default_args = {
            'base_dir': 'runs',
            'exp_name': None,
            'branch_file': 'branch_net_K_sampled.csv',
            'trunk_file': 'trunk_net_K_sampled.csv',
            'temp_file': 'merged_all_time_points_K_sampled.csv',
            'isPCA': False,
            'pca_dim': 512,
            'test_size': 0.2,
            'random_state': 42,
            'scaling_method': "standard",
            'branch_layers': [256, 256, 128],
            'trunk_layers': [64, 64, 128],
            'activation': 'tanh',
            'initializer': 'Glorot normal',
            'is_output_activation': False,
            'output_activation': 'relu',
            'isDropout': False,
            'dropout_rate': 0.1,
            'is_bias': True,
            'optimizer': 'adamw',
            'lr': 0.0001,
            'weight_decay': 1e-3,
            'decay': None,
            'epochs': 10000,
            'batch_size': None,
            'display_every': 1,
            'log_freq': 1,
            'save_freq': 1000,
            'metrics': ["mean l2 relative error", "MAPE", "MAE", "RMSE"],
            'isES': False,
            'monitor': 'loss_test',
            'min_delta': 1e-5,
            'patience': 3000,
            'baseline': 1e-5,
            'start_from_epoch': 1000
        }

    # 第二步：使用配置文件中的参数作为默认值，创建主参数解析器
    parser = argparse.ArgumentParser(description="DeepONet模型训练参数")

    # 配置文件
    parser.add_argument("--config", type=str, default=config_args.config, help="YAML配置文件路径")

    # 实验配置
    parser.add_argument("--base_dir", type=str, default=default_args['base_dir'], help="实验结果保存的基础目录")
    parser.add_argument("--exp_name", type=str, default=default_args['exp_name'], help="实验名称，默认使用时间戳")

    # 数据配置
    parser.add_argument("--branch_file", type=str, default=default_args['branch_file'], help="分支网络输入文件")
    parser.add_argument("--trunk_file", type=str, default=default_args['trunk_file'], help="主干网络输入文件")
    parser.add_argument("--temp_file", type=str, default=default_args['temp_file'], help="温度标签文件")
    parser.add_argument("--isPCA", type=bool, default=default_args['isPCA'], help="是否使用PCA降维")
    parser.add_argument("--pca_dim", type=int, default=default_args['pca_dim'], help="PCA降维维度")
    parser.add_argument("--test_size", type=float, default=default_args['test_size'], help="测试集比例")
    parser.add_argument("--random_state", type=int, default=default_args['random_state'], help="随机种子")
    parser.add_argument("--scaling_method", type=str, default=default_args['scaling_method'], help="数据预处理方法")

    # 模型配置
    parser.add_argument("--branch_layers", type=int, nargs="+", default=default_args['branch_layers'],
                        help="分支网络隐藏层结构")
    parser.add_argument("--trunk_layers", type=int, nargs="+", default=default_args['trunk_layers'],
                        help="主干网络隐藏层结构")
    parser.add_argument("--activation", type=str, default=default_args['activation'], help="激活函数")
    parser.add_argument("--initializer", type=str, default=default_args['initializer'], help="初始化方法")

    parser.add_argument("--is_output_activation", type=str, default=default_args['is_output_activation'],
                        help="是否设置输出层激活函数")
    parser.add_argument("--output_activation", type=str, default=default_args['output_activation'],
                        help="输出层激活函数")
    parser.add_argument("--isDropout", type=bool, default=default_args['isDropout'], help="是否dropout")
    parser.add_argument("--dropout_rate", type=float, default=default_args['dropout_rate'], help="dropout率")
    parser.add_argument("--is_bias", type=bool, default=default_args['is_bias'], help="输出层是否设置偏置")

    # 训练配置
    parser.add_argument("--optimizer", type=str, default=default_args['optimizer'], help="优化器类型")
    parser.add_argument("--lr", type=float, default=default_args['lr'], help="学习率")
    parser.add_argument("--weight_decay", type=float, default=default_args['weight_decay'], help="权重衰减")
    parser.add_argument("--decay", type=str, nargs="+", default=default_args['decay'],
                        help="学习率衰减策略，例如: exponential 0.995 或 cosine 10000 1e-7")
    parser.add_argument("--epochs", type=int, default=default_args['epochs'], help="训练轮次")
    parser.add_argument("--batch_size", type=int, default=default_args['batch_size'], help="批次大小")
    parser.add_argument("--display_every", type=int, default=default_args['display_every'], help="显示频率")

    # 回调配置
    parser.add_argument("--log_freq", type=int, default=default_args['log_freq'], help="日志记录频率")
    parser.add_argument("--save_freq", type=int, default=default_args['save_freq'], help="模型保存频率")
    parser.add_argument("--metrics", type=str, nargs="+", default=default_args['metrics'], help="评估指标")

    # 早停配置
    parser.add_argument("--isES", type=str, default=default_args['isES'], help="是否设置早停")
    parser.add_argument("--monitor", type=str, default=default_args['monitor'], help="早停监控指标")
    parser.add_argument("--min_delta", type=float, default=default_args['min_delta'], help="最小改善幅度")
    parser.add_argument("--patience", type=int, default=default_args['patience'], help="早停耐心值")
    parser.add_argument("--baseline", type=float, default=default_args['baseline'], help="早停基准值")
    parser.add_argument("--start_from_epoch", type=int, default=default_args['start_from_epoch'], help="早停开始轮次")

    args = parser.parse_args()

    # 处理学习率衰减参数
    if args.decay is not None:
        if isinstance(args.decay, list) and len(args.decay) > 0:
            if args.decay[0] == "exponential":
                args.decay = ("exponential", float(args.decay[1]))
            elif args.decay[0] == "cosine":
                args.decay = ("cosine", int(args.decay[1]), float(args.decay[2]))
            elif args.decay[0] == "step":
                args.decay = ("step", int(args.decay[1]), float(args.decay[2]))
            elif args.decay[0] == "inverse time":
                args.decay = ("step", int(args.decay[1]), float(args.decay[2]))
            # elif args.decay[0] == "lambda": # ("lambda", lambda_fn: Callable[[step], float])
            #     args.decay = ("lambda", args.decay[1])

    print("*" * 100)
    print("最终参数配置:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("*" * 100)

    return args


def load_data(branch_file, trunk_file, temp_file):
    """加载数据文件

    Args:
        branch_file: 分支网络输入文件路径
        trunk_file: 主干网络输入文件路径
        temp_file: 温度标签文件路径

    Returns:
        branch_df: 分支网络输入数据
        trunk_df: 主干网络输入数据
        temp_df: 温度数据
        temp_df_with_titles: 带标题的温度数据
    """
    # trunk_net.csv: 每行一个 node_id (index)，列为 x,y,z
    trunk_df = pd.read_csv(trunk_file, index_col=0)
    print(f"[INFO] trunk_net 维度: {trunk_df.shape} (nodes × xyz)")

    # merged_all_time_points.csv: 每个样本真实温度 (test, step, increment, node_1...node_N)
    temp_df = pd.read_csv(temp_file, index_col=0)
    temp_df_with_titles = temp_df.copy()  # 保留完整列名的副本
    temp_df.drop(columns=['test', 'step', 'increment'], inplace=True)
    print(f"[INFO] 温度数据 {temp_file} 维度: {temp_df.shape}")

    # branch_net.csv: 每行一个 sample，每列一个 node (列名为 node_id)
    branch_df = pd.read_csv(branch_file, index_col=0)
    branch_df = branch_df.iloc[temp_df.index]
    print(f"[INFO] branch_net 维度: {branch_df.shape} (samples × nodes)")

    return branch_df, trunk_df, temp_df, temp_df_with_titles


def prepare_data(branch_df, trunk_df, temp_df, isPCA=False, pca_dim=512, test_size=0.2, random_state=42,
                 scaling_method="standard"):
    """
    准备 DeepONet 输入输出数据

    Args:
        branch_df: 分支网络输入数据
        trunk_df: 主干网络输入数据
        temp_df: 温度数据
        isPCA: 是否使用 PCA 降维
        pca_dim: PCA 降维维度
        test_size: 测试集比例
        random_state: 随机种子
        scaling_method: 数据缩放方式 ("standard" | "minmax" | "none")

    Returns:
        data: DeepONet 数据结构
        X_branch: 原始分支输入
        scale_params: 包含用于逆变换的参数字典
        X_branch_test_scaled: 处理后的测试集分支输入
        X_trunk_test: 测试集主干输入
        y_test_scaled: 处理后的测试集输出
    """
    # Branch 输入：热流密度
    X_branch = branch_df.to_numpy(dtype=np.float32)
    if isPCA:
        pca = PCA(n_components=pca_dim)
        X_branch = pca.fit_transform(X_branch)
        print(f"[INFO] PCA 后维度: {X_branch.shape}")

    # Trunk 输入：节点坐标
    X_trunk = trunk_df.to_numpy(dtype=np.float32)

    # 输出 y：温度
    y = temp_df.to_numpy(dtype=np.float32)

    # 划分训练集与测试集
    X_branch_train, X_branch_test, y_train, y_test = train_test_split(
        X_branch, y, test_size=test_size, random_state=random_state
    )
    X_trunk_train = X_trunk
    X_trunk_test = X_trunk

    print(f"[INFO] 训练样本数: {X_branch_train.shape[0]}, 测试样本数: {X_branch_test.shape[0]}")
    print(f"[INFO] 节点数: {X_trunk.shape[0]}")

    # 根据 scaling_method 选择不同的数据预处理方式
    if scaling_method == "standard":
        # === 标准化 ===
        branch_mean = X_branch_train.mean(axis=0, keepdims=True)
        branch_std = X_branch_train.std(axis=0, keepdims=True) + 1e-8
        X_branch_train_scaled = (X_branch_train - branch_mean) / branch_std
        X_branch_test_scaled = (X_branch_test - branch_mean) / branch_std

        # y_mean = np.mean(y_train)
        # y_std = np.std(y_train) + 1e-8
        # y_train_scaled = (y_train - y_mean) / y_std
        # y_test_scaled = (y_test - y_mean) / y_std

        y_mean = np.mean(y_train, axis=0, keepdims=True)
        y_std = np.std(y_train, axis=0, keepdims=True) + 1e-8
        y_train_scaled = (y_train - y_mean) / y_std
        y_test_scaled = (y_test - y_mean) / y_std

        scale_params = {
            "type": "standard",
            "branch_mean": branch_mean,
            "branch_std": branch_std,
            "y_mean": y_mean,
            "y_std": y_std,
        }

    elif scaling_method == "minmax":
        # === 归一化 ===
        branch_min = X_branch_train.min(axis=0, keepdims=True)
        branch_max = X_branch_train.max(axis=0, keepdims=True)
        X_branch_train_scaled = (X_branch_train - branch_min) / (branch_max - branch_min + 1e-8)
        X_branch_test_scaled = (X_branch_test - branch_min) / (branch_max - branch_min + 1e-8)

        y_min = np.min(y_train)
        y_max = np.max(y_train)
        y_train_scaled = (y_train - y_min) / (y_max - y_min + 1e-8)
        y_test_scaled = (y_test - y_min) / (y_max - y_min + 1e-8)

        scale_params = {
            "type": "minmax",
            "branch_min": branch_min,
            "branch_max": branch_max,
            "y_min": y_min,
            "y_max": y_max,
        }

    else:
        # === 不做任何处理 ===
        X_branch_train_scaled = X_branch_train
        X_branch_test_scaled = X_branch_test
        y_train_scaled = y_train
        y_test_scaled = y_test

        scale_params = {"type": "none"}

    # 构造 DeepONet 数据结构
    data = dde.data.TripleCartesianProd(
        X_train=(X_branch_train_scaled, X_trunk_train),
        y_train=y_train_scaled,
        X_test=(X_branch_test_scaled, X_trunk_test),
        y_test=y_test_scaled,
    )

    return data, X_branch, scale_params, X_branch_test_scaled, X_trunk_test, y_test


class CustomDeepONet(dde.nn.DeepONetCartesianProd):
    def __init__(
            self,
            layer_sizes_branch,
            layer_sizes_trunk,
            activation,
            kernel_initializer,
            is_output_activation=False,
            output_activation="relu",  # 新增参数：输出层激活函数
            num_outputs=1,
            multi_output_strategy=None,
            isDropout=False,
            dropout_rate=0.1,
            is_bias=True,
            init_bias=None
    ):
        # 调用父类初始化
        super().__init__(
            layer_sizes_branch,
            layer_sizes_trunk,
            activation,
            kernel_initializer,
            num_outputs,
            multi_output_strategy
        )

        # 保存输出激活函数
        self.output_activation_name = output_activation
        self.output_activation = self._get_output_activation(output_activation)

        # 为branch和trunk网络添加dropout层
        self.isDropout = isDropout
        self.is_output_activation = is_output_activation
        self.branch_dropout = nn.Dropout(dropout_rate)
        self.trunk_dropout = nn.Dropout(dropout_rate)

        self.is_bias = is_bias
        # 可学习偏置，维度为输出维度（通常等于节点数的一部分）；如果num_outputs==1也可以用标量
        if is_bias and init_bias is None:
            init_bias = 0.0
            # 使用参数张量，允许反向传播学习
        self.output_bias = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32))

    def _get_output_activation(self, activation_name):
        """获取输出激活函数"""
        activations = {
            "relu": nn.ReLU(),
            "softplus": nn.Softplus(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "linear": lambda x: x  # 恒等映射，即无激活
        }
        return activations.get(activation_name.lower(), nn.ReLU())

    def merge_branch_trunk(self, x_func, x_loc, index):
        """重写合并方法，在输出前应用激活函数"""
        # 调用父类的计算
        y = super().merge_branch_trunk(x_func, x_loc, index)
        if self.is_output_activation:
            # 应用输出激活函数
            y = self.output_activation(y)
        if self.is_bias:
            y=y + self.output_bias
        return y

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]

        # Trunk net input transform
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)

        if self.isDropout:
            # 应用dropout
            x_func = self.branch_dropout(x_func)
            x_loc = self.trunk_dropout(x_loc)

        x = self.multi_output_strategy.call(x_func, x_loc)

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


# ===========================================================
# 模型定义模块
# ===========================================================
def create_model(data, branch_dim, trunk_dim=3, branch_layers=[256, 256, 128],
                 trunk_layers=[64, 64, 128], activation="tanh", initializer="Glorot normal", is_output_activation=False,
                 output_activation="softplus",isDropout=False, dropout_rate=0.1,is_bias=True,init_bias=None):
    """创建DeepONet模型

    Args:
        data: DeepONet数据结构
        branch_dim: 分支网络输入维度
        trunk_dim: 主干网络输入维度
        branch_layers: 分支网络隐藏层结构
        trunk_layers: 主干网络隐藏层结构
        activation: 激活函数
        initializer: 初始化方法

    Returns:
        model: DeepONet模型
    """
    # 构建完整的网络结构
    branch_net = [branch_dim] + branch_layers
    trunk_net = [trunk_dim] + trunk_layers

    # 创建DeepONet网络
    # net = dde.nn.DeepONetCartesianProd(
    #     branch_net,  # branch net
    #     trunk_net,  # trunk net
    #     activation,
    #     initializer,
    # )

    net = CustomDeepONet(
        branch_net,
        trunk_net,
        activation,
        initializer,
        is_output_activation=is_output_activation,
        output_activation=output_activation,  # 推荐使用softplus，比ReLU更平滑
        isDropout=isDropout,
        dropout_rate=dropout_rate,
        is_bias=is_bias,
        init_bias=init_bias
    )
    # 创建模型
    model = dde.Model(data, net)

    return model


# ===========================================================
# 结果可视化和评估模块
# ===========================================================
def plot_loss_history(loss_history, model, fname=None):
    """绘制损失曲线

    Args:
        loss_history: 损失历史
        model: 模型
        fname: 保存文件名
    """
    # 处理不同长度的数组
    loss_train = np.array([np.sum(loss) for loss in loss_history.loss_train])
    loss_test = np.array([np.sum(loss) for loss in loss_history.loss_test])

    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history.steps, loss_train, label="Train loss")
    plt.semilogy(loss_history.steps, loss_test, label="Test loss")
    # 使用实际的metrics名称而不是函数引用
    # metrics_names = ["mean l2 relative error", "MAPE", "MAE", "RMSE"]
    # for i in range(len(loss_history.metrics_test[0])):
    #     plt.semilogy(
    #         loss_history.steps,
    #         np.array(loss_history.metrics_test)[:, i],
    #         label=metrics_names[i],
    #     )
    plt.xlabel("# Steps")
    plt.ylabel("Loss/Metrics")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)

    if isinstance(fname, str):
        plt.savefig(fname, dpi=300)
        print(f"损失曲线已保存到: {fname}")


def evaluate_model(y_true, y_pred, output_dir):
    """评估模型性能

    Args:
        y_true: 真实值
        y_pred: 预测值
        output_dir: 输出目录
    """
    # 计算各种评估指标
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    # 打印评估结果
    print("\n" + "=" * 50)
    print("模型评估结果:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²: {r2:.6f}")
    print("=" * 50)

    # 保存评估结果
    metrics = {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape),
        "R2": float(r2)
    }

    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"评估指标已保存到: {metrics_path}")


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

    # 创建SummaryWriter实例
    writer = SummaryWriter(log_dir=log_dir)

    # 保存配置
    config = vars(args)
    save_config(config, exp_dir)

    # 准备数据
    # data, X_branch, scale_params, X_branch_test_scaled, X_trunk_test, y_test

    # 运行Optuna优化
    study = run_optuna_optimization(
        args, output_dir
    )

    # 分析和保存结果
    best_params = analyze_optuna_results(study, output_dir)

    print("\n✅ Optuna优化完成！")
    print("使用最佳超参数重新训练最终模型...")

