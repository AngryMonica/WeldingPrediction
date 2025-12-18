import socket
import json
import torch
import numpy as np
import os
import json
import torch
import numpy as np
import pandas as pd
import deepxde as dde
import time
import threading
import queue

from D_model_train import CustomDeepONet
from pathlib import Path

data_dir = Path(r'D:\Users\MXY\PycharmProjects\data')
############################################
# 1. 加载 PyTorch 模型
############################################
# 注意：必须是 torch.save(model) 保存的方式
model_path = data_dir/"final_model.pt"
config_path= data_dir/"config.json"
trunk_net_path= data_dir/"t1s1/sampled_trunk_net.csv"
nodes_path=data_dir/ "t1s1/sampled_nodes.csv"



model = torch.load(model_path, map_location="cpu",weights_only=False)
############################################
# 2. 读取节点坐标（nodes.txt）
############################################
def load_nodes():
    nodes = []
    with open(nodes_path, "r") as f:
        for line in f:
            items = line.strip().split(",")
            x = float(items[0])
            y = float(items[1])
            z = float(items[2])
            nodes.append([x, y, z])
    return np.array(nodes, dtype=np.float32)

nodes = load_nodes()
n_nodes = nodes.shape[0]

############################################
# 3. 双椭球热流密度计算（可替换为你的公式）
############################################
def double_ellipsoid_heat_source(X, Y, Z, params):
    """
    输入 X,Y,Z 为 numpy 数组（shape: n_nodes），params 是字典（数值类型）。
    返回 q（numpy 数组，shape: n_nodes）。
    """
    I = params[0]; U = params[1]; v = params[2]
    xc = params[3]; yc = params[4]; zc = params[5]
    a1 = params[6]; a2 = params[7]; b = params[8]; c = params[9]
    eta = params[10]; f1 = params[11]; f2 = params[12]

    # 基本参数检查
    if v == 0 or any(np.isnan([a1, a2, b, c, I, U, eta])):
        # 返回全 NaN 并打印警告
        print(f"[警告] 参数异常: v={v}, a1={a1}, a2={a2}, b={b}, c={c}, I={I}, U={U}, eta={eta}")
        return np.full_like(X, np.nan, dtype=float)

    Q = eta * I * U / v

    # 坐标变换（向量化）
    Xc = X - xc
    Yc = Y - yc
    Zc = Z - zc

    # 前/后半椭球系数（向量化）
    coef_f = 6.0 * np.sqrt(3.0) * f1 * Q / (a1 * b * c * np.pi * np.sqrt(np.pi))
    coef_r = 6.0 * np.sqrt(3.0) * f2 * Q / (a2 * b * c * np.pi * np.sqrt(np.pi))

    # 指数项
    # 为避免浮点下溢或上溢，直接计算（numpy 会处理极小值为 0）
    q_f = coef_f * np.exp(-3.0 * Xc**2 / (a1**2)) * np.exp(-3.0 * Yc**2 / (b**2)) * np.exp(-3.0 * Zc**2 / (c**2))
    q_r = coef_r * np.exp(-3.0 * Xc**2 / (a2**2)) * np.exp(-3.0 * Yc**2 / (b**2)) * np.exp(-3.0 * Zc**2 / (c**2))

    q = np.where(Xc <= 0.0, q_f, q_r)
    return q


############################################
# 4. 根据 Unity 参数生成每个节点的热流密度输入
############################################
q_values_list=[]
def compute_heat_flux_realtime(nodes, params):
    # X = nodes["x"].to_numpy()
    # Y = nodes["y"].to_numpy()
    # Z = nodes["z"].to_numpy()
    # for i in range(n_nodes):
    X, Y, Z = nodes.T
    q_values = double_ellipsoid_heat_source(X, Y, Z, params)

    if len(q_values_list)>=10:
        q_values_list.pop(0)
    q_values_list.append(q_values)
    sum_q_values =np.sum(q_values_list,axis=0)

    return sum_q_values.astype(np.float32)


############################################
# 5. 输入模型推理
############################################
def run_inference(branch_input, config_path="config.json", model_path="final_model.pt"):
    """
    执行模型推理 (修复了读取 config.json 中字符串格式数组报错的问题)
    """

    # ---------------------------------------------------------
    # 1. 读取训练配置信息
    # ---------------------------------------------------------
    with open(config_path, "r") as f:
        cfg = json.load(f)

    branch_layers = cfg["branch_layers"]
    trunk_layers = cfg["trunk_layers"]
    activation = cfg["activation"]
    initializer = cfg["initializer"]
    is_output_activation = cfg["is_output_activation"]
    output_activation = cfg["output_activation"]
    is_bias = cfg["is_bias"]
    scaling_method = cfg["scaling_method"]

    # --- [新增] 辅助函数：解析可能是字符串格式的数组 ---
    def parse_config_value(val):
        # 如果是字符串 (例如 "[-0.02  5.37]")，则去除括号并按空格分割解析
        if isinstance(val, str):
            val = val.replace('[', '').replace(']', '').replace('\n', ' ')
            return np.fromstring(val, sep=' ', dtype=np.float32)
        # 如果已经是列表，直接转换
        return np.array(val, dtype=np.float32)

    # ---------------------------------------------------------
    # 2. 读取 trunk 坐标
    # ---------------------------------------------------------
    trunk_df = pd.read_csv(trunk_net_path, index_col=0)
    X_trunk = trunk_df.to_numpy(dtype=np.float32)   # shape = [num_nodes, 3]

    # ---------------------------------------------------------
    # 3. 构建 DeepONet 网络结构
    # ---------------------------------------------------------
    branch_dim = len(branch_input)
    trunk_dim = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = CustomDeepONet(
        [branch_dim] + branch_layers,
        [trunk_dim] + trunk_layers,
        activation,
        initializer,
        is_output_activation=is_output_activation,
        output_activation=output_activation,
        isDropout=False,
        dropout_rate=0,
        is_bias=is_bias,
    )

    # ---------------------------------------------------------
    # 4. 加载训练好的模型权重
    # ---------------------------------------------------------
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    # ---------------------------------------------------------
    # 5. 数据预处理 (Branch 和 Trunk 同时处理)
    # ---------------------------------------------------------
    branch_input = np.array(branch_input, dtype=np.float32)

    if scaling_method == "standard":
        # --- Branch 标准化 ---
        branch_mean = parse_config_value(cfg["branch_mean"])
        branch_std = parse_config_value(cfg["branch_std"])
        branch_scaled = (branch_input - branch_mean) / branch_std

        # --- Trunk (坐标) 标准化 ---
        trunk_mean = parse_config_value(cfg["trunk_mean"])
        trunk_std = parse_config_value(cfg["trunk_std"])
        X_trunk = (X_trunk - trunk_mean) / trunk_std

    elif scaling_method == "minmax":
        # --- Branch 归一化 ---
        branch_min = parse_config_value(cfg["branch_min"])
        branch_max = parse_config_value(cfg["branch_max"])
        branch_scaled = (branch_input - branch_min) / (branch_max - branch_min)

        # --- Trunk (坐标) 归一化 ---
        trunk_min = parse_config_value(cfg["trunk_min"])
        trunk_max = parse_config_value(cfg["trunk_max"])
        X_trunk = (X_trunk - trunk_min) / (trunk_max - trunk_min)

    else:
        # 无缩放
        branch_scaled = branch_input
        # X_trunk 保持原样

    # 调整 Branch 形状
    branch_scaled = branch_scaled.reshape(1, -1)   # shape = [1, branch_dim]

    # 转为 Tensor
    branch_tensor = torch.from_numpy(branch_scaled).float().to(device)
    trunk_tensor = torch.from_numpy(X_trunk).float().to(device)

    # ---------------------------------------------------------
    # 6. 推理
    # ---------------------------------------------------------
    t0 = time.perf_counter()

    with torch.no_grad():
        # DeepONetCartesianProd 输入为 tuple (branch, trunk)
        y_scaled = net((branch_tensor, trunk_tensor)).cpu().numpy()

    t1 = time.perf_counter()
    print(f"Inference time: {(t1 - t0) * 1000:.3f} ms")

    # ---------------------------------------------------------
    # 7. 反归一化输出 (温度)
    # ---------------------------------------------------------
    if scaling_method == "standard":
        y_mean = float(cfg["y_mean"])
        y_std = float(cfg["y_std"])
        y_pred = y_scaled * y_std + y_mean

    elif scaling_method == "minmax":
        y_min = float(cfg["y_min"])
        y_max = float(cfg["y_max"])
        y_pred = y_scaled * (y_max - y_min) + y_min
    else:
        y_pred = y_scaled

    return y_pred  # shape = [num_nodes, 1]

############################################
# 6. 接收 Unity 信息并处理
############################################
# def handle_message(msg, conn):
#     if not msg.startswith("fea"):
#         print(r"returen: msg.startswith(fea) ",msg.startswith("fea"))
#         return
#
#     # 去掉 fea,
#     msg = msg.replace("fea,", "")
#     items = msg.split(",")
#     params = [float(x) for x in items]
#     heat_density = compute_heat_flux_realtime(nodes,params)
#     pred = run_inference(heat_density,config_path=config_path,model_path=model_path)
#     print(pred)
#     send_str = json.dumps(pred[0].tolist()) + "\n"
#     conn.send(send_str.encode("utf-8"))
task_queue = queue.Queue()
def handle_message(msg, conn):
    if not msg.startswith("fea"):
        return

    msg = msg.replace("fea,", "")
    items = msg.split(",")
    params = [float(x) for x in items]

    task_queue.put((params, conn))

def inference_worker():
    while True:
        params, conn = task_queue.get()

        heat_density = compute_heat_flux_realtime(nodes, params)
        pred = run_inference(heat_density, config_path=config_path, model_path=model_path)

        send_str = json.dumps(pred[0].tolist()) + "\n"
        print(pred)
        conn.send(send_str.encode("utf-8"))

        task_queue.task_done()


############################################
# 7. 启动 socket 服务器
############################################
def run_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 8000))
    server.listen(1)

    print("Python Server 启动，等待 Unity 连接...")

    conn, addr = server.accept()
    print("已连接:", addr)

    # 设置读超时（例如 5 秒）
    # 超时后不会断开，而是继续循环等待
    conn.settimeout(50)
    buffer = ""
    while True:
        data = conn.recv(4096)
        # 如果 recv() 超时，不会返回 data，而是进入 except
        # 因此这里代表真正的断开
        if data == b"":
            print("Unity 断开连接")
            break

        msg = data.decode("utf-8")
        print("收到:", msg)
        buffer += msg
        while "#END#" in buffer:
            m, buffer = buffer.split("#END#", 1)
            m = m.strip()
            if m:
                handle_message(m, conn)

        # handle_message(msg, conn)

        # messages=msg.split("\n")
        # for msg in messages:
        #     if msg.strip() == "":
        #         continue
        #     handle_message(msg, conn)


        # try:
        #     data = conn.recv(4096)
        #     # 如果 recv() 超时，不会返回 data，而是进入 except
        #     # 因此这里代表真正的断开
        #     if data == b"":
        #         print("Unity 断开连接")
        #         break
        #
        #     msg = data.decode("utf-8")
        #     print("收到:", msg)
        #
        #     handle_message(msg, conn)
        #
        # except socket.timeout:
        #     # 超时代表 Unity 暂时没有发送数据 → 正常现象
        #     # 继续等待即可
        #     continue
        #
        # except Exception as e:
        #     print("错误:", e)
        #     continue

    conn.close()
    server.close()


if __name__ == "__main__":
    threading.Thread(target=inference_worker, daemon=True).start()
    run_server()
