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
from D1_model_train_multi import MultiTaskDeepONet

data_dir = Path(r'D:\Users\MXY\PycharmProjects\data')

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
    q_values = double_ellipsoid_heat_source(X, Y, Z, params)

    if len(q_values_list) >= 10:
        q_values_list.pop(0)
    q_values_list.append(q_values)
    sum_q_values = np.sum(q_values_list, axis=0)

    return sum_q_values.astype(np.float32)


def run_inference(branch_input, config_path="config.json", model_path="final_model.pt"):
    """
    执行多任务模型推理，返回温度场和应力场预测结果
    """
    # 1. 读取配置
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
    tasks_config = cfg["tasks_config"]
    scale_params = cfg.get("scale_params", {})

    # 2. 读取 trunk 坐标
    trunk_df = pd.read_csv(trunk_net_path, index_col=0)
    X_trunk = trunk_df.to_numpy(dtype=np.float32)

    # 3. 构建多任务模型
    branch_dim = len(branch_input)
    trunk_dim = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = MultiTaskDeepONet(
        layer_sizes_branch=[branch_dim] + branch_layers,
        layer_sizes_trunk=[trunk_dim] + trunk_layers,
        activation=activation,
        kernel_initializer=initializer,
        tasks_config=tasks_config,
        is_output_activation=is_output_activation,
        output_activation=output_activation,
        multi_output_strategy=cfg.get("multi_output_strategy", "split_branch"),
        isDropout=False,
        dropout_rate=0,
        is_bias=is_bias,
    )

    # 4. 加载权重
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    # 5. 数据预处理
    branch_input = np.array(branch_input, dtype=np.float32)

    if scaling_method == "standard":
        branch_mean = np.array(scale_params["branch"]["mean"], dtype=np.float32)
        branch_std = np.array(scale_params["branch"]["std"], dtype=np.float32)
        branch_scaled = (branch_input - branch_mean) / branch_std

        trunk_mean = np.array(scale_params["trunk"]["mean"], dtype=np.float32)
        trunk_std = np.array(scale_params["trunk"]["std"], dtype=np.float32)
        X_trunk = (X_trunk - trunk_mean) / trunk_std
    elif scaling_method == "minmax":
        branch_min = np.array(scale_params["branch"]["min"], dtype=np.float32)
        branch_max = np.array(scale_params["branch"]["max"], dtype=np.float32)
        branch_scaled = (branch_input - branch_min) / (branch_max - branch_min)

        trunk_min = np.array(scale_params["trunk"]["min"], dtype=np.float32)
        trunk_max = np.array(scale_params["trunk"]["max"], dtype=np.float32)
        X_trunk = (X_trunk - trunk_min) / (trunk_max - trunk_min)
    else:
        branch_scaled = branch_input


    branch_scaled = branch_scaled.reshape(1, -1)
    branch_tensor = torch.from_numpy(branch_scaled).float().to(device)
    trunk_tensor = torch.from_numpy(X_trunk).float().to(device)
    # 6. 推理
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = net([branch_tensor, trunk_tensor])  # 返回 dict: {task_name: tensor}
    t1 = time.perf_counter()
    print(f"Inference time: {(t1 - t0) * 1000:.3f} ms")

    # 7. 反归一化各任务输出
    results = {}
    for task_name, y_scaled_tensor in outputs.items():
        y_scaled = y_scaled_tensor.cpu().numpy()

        if scaling_method == "standard":
            y_mean = np.array(scale_params[task_name]["mean"], dtype=np.float32)
            y_std = np.array(scale_params[task_name]["std"], dtype=np.float32)
            y_pred = y_scaled * y_std + y_mean
        elif scaling_method == "minmax":
            y_min = np.array(scale_params[task_name]["min"], dtype=np.float32)
            y_max = np.array(scale_params[task_name]["max"], dtype=np.float32)
            y_pred = y_scaled * (y_max - y_min) + y_min
        else:
            y_pred = y_scaled

        results[task_name] = y_pred.flatten().tolist()

    print(f"预测结果: temperature 范围 [{min(results['temperature']):.2f}, {max(results['temperature']):.2f}]")
    if 'stress' in results:
        print(f"          stress 范围 [{min(results['stress']):.2f}, {max(results['stress']):.2f}]")

    return results  # {"temperature": [...], "stress": [...]}


task_queue = queue.Queue()


def handle_message(msg, conn):
    # 移除 #END# 后缀
    msg = msg.replace("#END#", "").strip()

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
        results = run_inference(heat_density, config_path=config_path, model_path=model_path)
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
    threading.Thread(target=inference_worker, daemon=True).start()
    run_server()
