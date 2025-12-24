# python
import socket
import json
import torch
import numpy as np
import pandas as pd
import time
import threading
import queue
import queue
from pathlib import Path

# 修改导入：使用多任务模型
from D3_model_train_unity import MultiTaskDeepONet

# data_dir = Path(r'D:\Users\MXY\PycharmProjects\data\t1s1')
BASE_DIR = Path(__file__).parent
data_dir = BASE_DIR

model_path = data_dir / "final_model.pt"
config_path = data_dir / "config.json"
trunk_net_path = data_dir / "sampled_trunk_net.csv"
nodes_path = data_dir / "sampled_nodes.csv"


def load_nodes():
    nodes = []
    with open(nodes_path, "r") as f:
        for line in f:
            items = line.strip().split(",")
            x, y, z = float(items[0]), float(items[1]), float(items[2])
            nodes.append([x, y, z])
    return np.array(nodes, dtype=np.float32)


nodes = load_nodes()
n_nodes = nodes.shape[0]

q_values_list = []


def double_ellipsoid_heat_source(X, Y, Z, params):

    I, U, v = params[0], params[1], params[2]
    xc, yc, zc = params[3], params[4], params[5]
    a1, a2, b, c = params[6], params[7], params[8], params[9]
    eta, f1, f2 = params[10], params[11], params[12]

    if v == 0 or any(np.isnan([a1, a2, b, c, I, U, eta])):
        return np.full_like(X, np.nan, dtype=float)

    Q = eta * I * U / v
    Xc, Yc, Zc = X - xc, Y - yc, Z - zc

    coef_f = 6.0 * np.sqrt(3.0) * f1 * Q / (a1 * b * c * np.pi * np.sqrt(np.pi))
    coef_r = 6.0 * np.sqrt(3.0) * f2 * Q / (a2 * b * c * np.pi * np.sqrt(np.pi))

    q_f = coef_f * np.exp(-3.0 * Xc ** 2 / a1 ** 2) * np.exp(-3.0 * Yc ** 2 / b ** 2) * np.exp(-3.0 * Zc ** 2 / c ** 2)
    q_r = coef_r * np.exp(-3.0 * Xc ** 2 / a2 ** 2) * np.exp(-3.0 * Yc ** 2 / b ** 2) * np.exp(-3.0 * Zc ** 2 / c ** 2)

    return np.where(Xc <= 0.0, q_f, q_r)


def compute_heat_flux_realtime(nodes, params):
    X, Y, Z = nodes.T

    if params[0]=="reset":
        isReset=True
        params.pop(0)
    else:
        isReset=False
    params=[float(p) for p in params]
    if isReset:
        q_values_list.clear()
    q_values = double_ellipsoid_heat_source(X, Y, Z, params)

    if len(q_values_list) >= 10:
        q_values_list.pop(0)
    q_values_list.append(q_values)
    sum_q_values = np.sum(q_values_list, axis=0)

    return sum_q_values.astype(np.float32)


def init_model(config_path, model_path, trunk_net_path):
    global MODEL, TRUNK_TENSOR, CFG

    # 1. 读配置
    with open(config_path, "r") as f:
        CFG = json.load(f)

    cfg=CFG
    scaling_method=cfg["scaling_method"]
    scale_params=cfg.get("scale_params", {})

    # 2. 读 trunk 坐标（只一次）
    trunk_df = pd.read_csv(trunk_net_path, index_col=0)
    X_trunk = trunk_df.to_numpy(dtype=np.float32)

    if scaling_method == "standard":
        trunk_mean = np.array(scale_params["trunk_mean"], dtype=np.float32)
        trunk_std = np.array(scale_params["trunk_std"], dtype=np.float32)
        trunk_std = np.where(trunk_std == 0, 1.0, trunk_std)
        trunk_scaled = (X_trunk - trunk_mean) / trunk_std
    elif scaling_method == "minmax":
        trunk_min = np.array(scale_params["trunk_min"], dtype=np.float32)
        trunk_max = np.array(scale_params["trunk_max"], dtype=np.float32)
        trunk_range = np.where(trunk_max - trunk_min == 0, 1.0, trunk_max - trunk_min)
        trunk_scaled = (X_trunk - trunk_min) / trunk_range
    else:
        trunk_scaled = X_trunk

    trunk_dim = X_trunk.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 3. 构建模型（branch_dim 先占位）
    MODEL = {
        "net": None,
        "device": device
    }

    # 4. 构建 trunk tensor（常驻）
    TRUNK_TENSOR = torch.from_numpy(trunk_scaled).float().to(device)

    # 5. 加载权重（注意：模型结构稍后补）
    MODEL["state_dict"] = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    print("Model initialized")

def run_inference(branch_input):
    global MODEL, TRUNK_TENSOR, CFG

    cfg = CFG
    scaling_method = cfg["scaling_method"]
    scale_params = cfg.get("scale_params", {})

    branch_input = np.array(branch_input, dtype=np.float32)

    # 1. scaling
    if scaling_method == "standard":
        branch_mean = np.array(scale_params["branch_mean"], dtype=np.float32)
        branch_std = np.array(scale_params["branch_std"], dtype=np.float32)
        branch_std = np.where(branch_std == 0, 1.0, branch_std)
        branch_scaled = (branch_input - branch_mean) / branch_std
    elif scaling_method == "minmax":
        branch_min = np.array(scale_params["branch_min"], dtype=np.float32)
        branch_max = np.array(scale_params["branch_max"], dtype=np.float32)
        branch_range = np.where(branch_max - branch_min == 0, 1.0, branch_max - branch_min)
        branch_scaled = (branch_input - branch_min) / branch_range
    else:
        branch_scaled = branch_input

    branch_scaled = branch_scaled.reshape(1, -1)

    device = MODEL["device"]

    # 2. 第一次推理时才真正构建网络
    if MODEL["net"] is None:
        branch_dim = branch_scaled.shape[1]
        trunk_dim = TRUNK_TENSOR.shape[1]

        net = MultiTaskDeepONet(
            layer_sizes_branch=[branch_dim] + cfg["branch_layers"],
            layer_sizes_trunk=[trunk_dim] + cfg["trunk_layers"],
            activation=cfg["activation"],
            kernel_initializer=cfg["initializer"],
            tasks_config=cfg["tasks_config"],
            is_output_activation=cfg["is_output_activation"],
            output_activation=cfg["output_activation"],
            multi_output_strategy=cfg.get("multi_output_strategy", "split_branch"),
            isDropout=False,
            dropout_rate=0,
            is_bias=cfg["is_bias"],
        )

        net.load_state_dict(MODEL["state_dict"])
        net.to(device)
        net.eval()

        MODEL["net"] = net
        print("Network constructed")

    branch_tensor = torch.from_numpy(branch_scaled).float().to(device)

    # 3. 推理
    with torch.no_grad():
        outputs = MODEL["net"]([branch_tensor, TRUNK_TENSOR])

    # 4. 反归一化
    results = {}
    for task_name, y_scaled_tensor in outputs.items():
        y_scaled = y_scaled_tensor.cpu().numpy()

        if scaling_method == "standard":
            y_mean = np.array(scale_params[f"{task_name}_mean"], dtype=np.float32)
            y_std = np.array(scale_params[f"{task_name}_std"], dtype=np.float32)
            y_pred = y_scaled * y_std + y_mean
        elif scaling_method == "minmax":
            y_min = np.array(scale_params[f"{task_name}_min"], dtype=np.float32)
            y_max = np.array(scale_params[f"{task_name}_max"], dtype=np.float32)
            y_pred = y_scaled * (y_max - y_min) + y_min
        else:
            y_pred = y_scaled

        results[task_name] = y_pred.flatten().tolist()
    return results



task_queue = queue.Queue()


def handle_message(msg, conn):
    # 移除 #END# 后缀
    msg = msg.replace("#END#", "").strip()

    if msg == "ping":
        conn.send(b"pong\n")
        return

    if not msg.startswith("fea"):
        return

    msg = msg.replace("fea,", "")
    params = msg.split(",")
    task_queue.put((params, conn))


def inference_worker():
    while True:
        params, conn = task_queue.get()
        heat_density = compute_heat_flux_realtime(nodes, params)
        print('heat_density',heat_density[:5])  # 打印前5个值以供调试
        results = run_inference(heat_density)
        print(results['temperature'][:5])  # 打印前5个值以供调试
        # 发送包含多任务结果的 JSON
        send_str = json.dumps(results) + "\n"
        conn.send(send_str.encode("utf-8"))
        task_queue.task_done()



def run_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 8000))
    server.listen(1)

    print("Python Server 启动，等待 Unity 连接...")
    conn, addr = server.accept()
    print("已连接:", addr)
    conn.settimeout(50)

    buffer = ""
    while True:
        data = conn.recv(4096).decode("utf-8")
        print("收到:", data)
        if not data:
            break
        buffer += data
        while "#END#" in buffer:
            line, buffer = buffer.split("#END#", 1)
            if line.strip():
                handle_message(line.strip(), conn)

        # try:
        #     data = conn.recv(4096).decode("utf-8")
        #     print("收到:", data)
        #     if not data:
        #         break
        #     buffer += data
        #     while "\n" in buffer:
        #         line, buffer = buffer.split("\n", 1)
        #         if line.strip():
        #             handle_message(line.strip(), conn)
        # except socket.timeout:
        #     continue
        # except Exception as e:
        #     print(f"接收错误: {e}")
        #     break

    conn.close()
    server.close()


if __name__ == "__main__":
    init_model(
        config_path=config_path,
        model_path=model_path,
        trunk_net_path=trunk_net_path
    )

    threading.Thread(target=inference_worker, daemon=True).start()
    run_server()

