import socket
import json
import numpy as np
import pandas as pd
import threading
import queue
import time
from pathlib import Path

# 如果不需要跑模型，可以注释掉 torch 和 deepxde 相关引用，防止报错
# import torch
# import deepxde as dde
# from D_model_train import CustomDeepONet

data_dir = Path(r'D:\Users\MXY\PycharmProjects\data\t1s1')

############################################
# 1. 配置路径
############################################
nodes_path = data_dir / "sampled_nodes.csv"
# [修改] 真实温度数据路径
real_data_path = data_dir / "sampled_temperature.csv"


############################################
# 2. 读取节点坐标 (保留用于验证或后续使用)
############################################
def load_nodes():
    nodes = []
    if not nodes_path.exists():
        print(f"[警告] 找不到节点文件: {nodes_path}")
        return np.array([])

    with open(nodes_path, "r") as f:
        for line in f:
            items = line.strip().split(",")
            # 简单的容错处理
            if len(items) >= 3:
                x = float(items[0])
                y = float(items[1])
                z = float(items[2])
                nodes.append([x, y, z])
    return np.array(nodes, dtype=np.float32)


nodes = load_nodes()
n_nodes = nodes.shape[0]
print(f"节点数量: {n_nodes}")

############################################
# 3. [新增] 加载真实温度数据
############################################
real_temp_data = None
current_frame_index = 0


def load_real_data():
    global real_temp_data
    if not real_data_path.exists():
        print(f"[错误] 找不到温度数据文件: {real_data_path}")
        return

    print(f"正在加载真实温度数据: {real_data_path} ...")
    df = pd.read_csv(real_data_path)

    # 定义需要移除的元数据列
    meta_cols = ["test", "step", "increment", "step_time"]

    # 移除元数据列，只保留温度数据
    # errors='ignore' 防止某些列不存在时报错
    df_temps = df.drop(columns=meta_cols, errors='ignore')

    # 转换为 numpy 数组 (Rows, Nodes)
    real_temp_data = df_temps.values.astype(np.float32)

    print(f"真实数据加载完成。")
    print(f"  - 总帧数 (时间步): {real_temp_data.shape[0]}")
    print(f"  - 每帧节点数: {real_temp_data.shape[1]}")

    if real_temp_data.shape[1] != n_nodes:
        print(f"[警告] 数据列数 ({real_temp_data.shape[1]}) 与 nodes.csv 节点数 ({n_nodes}) 不一致！")


# 初始化加载数据
load_real_data()

############################################
# 4. 任务队列与处理逻辑
############################################
task_queue = queue.Queue()


def handle_message(msg, conn):
    # 即使是播放真实数据，Unity 依然会发送 "fea,..." 格式的消息来请求数据
    if not msg.startswith("fea"):
        return

    # 这里我们不再需要解析参数来计算热流，但保留解析逻辑以备不时之需
    msg = msg.replace("fea,", "")
    items = msg.split(",")
    try:
        params = [float(x) for x in items]
        # 将请求放入队列
        task_queue.put((params, conn))
    except ValueError:
        print("参数解析错误")


def inference_worker():
    global current_frame_index

    while True:
        # 等待 Unity 的请求
        params, conn = task_queue.get()

        if real_temp_data is None:
            print("[错误] 数据未加载，无法发送。")
            task_queue.task_done()
            continue

        # -------------------------------------------------
        # [修改] 核心逻辑：获取下一帧真实数据
        # -------------------------------------------------

        # 检查索引是否越界，如果越界则循环回 0 (循环播放)
        if current_frame_index >= len(real_temp_data):
            print(">>> 数据播放完毕，重新开始...")
            current_frame_index = 0

        # 获取当前帧数据 (1D array: [node_1, node_2, ...])
        current_temps = real_temp_data[current_frame_index]

        # 打印进度
        print(f"发送第 {current_frame_index + 1}/{len(real_temp_data)} 帧数据")

        # 索引递增
        current_frame_index += 1

        # -------------------------------------------------
        # 发送数据
        # -------------------------------------------------
        try:
            # current_temps 已经是 numpy array，tolist() 后即为浮点数列表
            # 对应原代码中的 pred[0].tolist()
            send_str = json.dumps(current_temps.tolist()) + "\n"
            conn.send(send_str.encode("utf-8"))
        except Exception as e:
            print(f"发送数据异常: {e}")

        task_queue.task_done()


############################################
# 5. 启动 socket 服务器
############################################
def run_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 允许端口复用，防止重启脚本时报端口被占用
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server.bind(("127.0.0.1", 8080))
        server.listen(1)
        print("Python Server (Real Data Mode) 启动，等待 Unity 连接...")

        conn, addr = server.accept()
        print("已连接:", addr)
        conn.settimeout(50)  # 设置超时

        buffer = ""
        while True:
            try:
                data = conn.recv(4096)
                if data == b"":
                    print("Unity 断开连接")
                    break

                msg = data.decode("utf-8")
                # print("收到:", msg) # 调试时可打开

                # 处理粘包问题
                buffer += msg
                while "#END#" in buffer:
                    m, buffer = buffer.split("#END#", 1)
                    m = m.strip()
                    if m:
                        handle_message(m, conn)

            except socket.timeout:
                continue
            except ConnectionResetError:
                print("连接被重置")
                break
            except Exception as e:
                print(f"通信错误: {e}")
                break

        conn.close()

    except Exception as e:
        print(f"服务器启动失败: {e}")
    finally:
        server.close()


if __name__ == "__main__":
    # 启动处理线程
    threading.Thread(target=inference_worker, daemon=True).start()
    # 启动服务器
    run_server()
