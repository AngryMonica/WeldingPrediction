import pandas as pd
import numpy as np
import sys
from pathlib import Path

data_dir = Path(r'D:\Users\MXY\PycharmProjects\data')
node_file = data_dir/"sampled_nodes.csv"
weld_file = data_dir/"welding_data.csv"
out_branch=data_dir/"sampled_herstory_branch_net.csv"
out_trunk=data_dir/"sampled_trunk_net.csv"
# =============================
# 1. 读取并清洗节点坐标
# =============================
def load_nodes(node_file):
    """
    读取 ID-coord.csv：假定真正的数据从第三行开始（前两行是元信息）。
    返回：按 node_id 排序、index 为 int 的 DataFrame，列为 ['x','y','z']。
    """
    # 先按原样读（全部当字符串），跳过前两行
    df = pd.read_csv(node_file,index_col=None, header=None, dtype=str, engine="python")
    df.columns = ["x", "y", "z"]

    return df[["x", "y", "z"]]

# =============================
# 2. 向量化双椭球热源函数
# =============================
def double_ellipsoid_heat_source(X, Y, Z, params):
    """
    输入 X,Y,Z 为 numpy 数组（shape: n_nodes），params 是字典（数值类型）。
    返回 q（numpy 数组，shape: n_nodes）。
    """
    a1 = params["a1"]; a2 = params["a2"]; b = params["b"]; c = params["c"]
    eta = params["eta"]; f1 = params["f1"]; f2 = params["f2"]
    I = params["current"]; U = params["voltage"]; v = params["speed"]
    xc = params["center_x"]; yc = params["center_y"]; zc = params["center_z"]

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

    q = np.where(Zc <= 0.0, q_f, q_r)

    # for qq in q:
    #     if qq>=1e+30:
    #         print(q)
    #         print(params)
    idx=0
    for q_value in q:
        idx+=1
        if q_value >= 1e+15:
            print(a1[idx])
            print(q_value)
    # 输出统计信息，便于调试
    # nan_count = np.isnan(q).sum()
    # zero_count = np.isclose(q, 0.0).sum()
    # print(f"  -> q len={len(q)}, NaN={nan_count}, approx zeros={zero_count}, min={np.nanmin(q):.3e}, max={np.nanmax(q):.3e}")
    return q

# =============================
# 3. 主计算函数（带检查）
# =============================
def compute_heat_flux(node_file, weld_file, out_branch, out_trunk):
    # 读取并清洗节点
    nodes = load_nodes(node_file)
    nodes.to_csv(out_trunk)
    print(f"trunk_net 已保存到: {out_trunk}\n")

    # 节点数组
    X = nodes["x"].to_numpy().astype(float)
    Y = nodes["y"].to_numpy().astype(float)
    Z = nodes["z"].to_numpy().astype(float)
    n_nodes = len(nodes)

    # 读取焊接数据
    weld_data = pd.read_csv(weld_file)
    print(f"读取 welding_data: 共 {len(weld_data)} 条记录\n")

    results = []
    sample_names = []
    sum_results=[]
    zhizhen=0
    # increment0_counts=0
    for idx, row in weld_data.iterrows():

        print(f"计算 step idx={idx}  (step={row.get('step','NA')}, increment={row.get('increment','NA')}) ...")
        # 安全地把每个参数转换为 float（出错则成 NaN）
        try:
            params = {
                "increment":int(row["increment"]),
                "current": float(row["current"]),
                "voltage": float(row["voltage"]),
                "speed": float(row["speed"]),
                "center_x": float(row["center_x"]),
                "center_y": float(row["center_y"]),
                "center_z": float(row["center_z"]),
                "a1": float(row["a1"]),
                "a2": float(row["a2"]),
                "b": float(row["b"]),
                "c": float(row["c"]),
                "eta": float(row["eta"]),
                "f1": float(row["f1"]),
                "f2": float(row["f2"]),
            }
        except Exception as e:
            print(f"[错误] 参数转换失败: {e}. 该步将被记录为 NaN。")
            results.append(np.full((n_nodes,), np.nan))
            sample_names.append(f"row_{idx}")
            continue
        # if params["increment"]==0:
        #     increment0_counts+=1
        #     continue
        # 计算（向量化）
        q_values = double_ellipsoid_heat_source(X, Y, Z, params)
        sum_q_values = q_values.copy()
        if params["increment"]==1:
            zhizhen=idx
        elif params["increment"]<=10:
            for i in range(zhizhen,idx):
                sum_q_values+=results[i]
        else:
            for i in range(idx-10, idx):
                sum_q_values += results[i]
        # # 检查长度一致性
        # if q_values.shape[0] != n_nodes:
        #     print(f"[错误] 计算结果长度与节点数不一致: {q_values.shape[0]} vs {n_nodes}")
        #     # 兜底：用 NaN 填充或截断
        #     if q_values.shape[0] < n_nodes:
        #         q_values = np.concatenate([q_values, np.full((n_nodes - q_values.shape[0],), np.nan)])
        #     else:
        #         q_values = q_values[:n_nodes]

        # 输出统计信息，便于调试
        nan_count = np.isnan(q_values).sum()
        zero_count = np.isclose(q_values, 0.0).sum()
        print(f"  -> q len={len(q_values)}, NaN={nan_count}, approx zeros={zero_count}, min={np.nanmin(q_values):.3e}, max={np.nanmax(q_values):.3e}")
        #
        nan_count = np.isnan(sum_q_values).sum()
        zero_count = np.isclose(sum_q_values, 0.0).sum()
        print(f"  -> q_sum len={len(sum_q_values)}, NaN={nan_count}, approx zeros={zero_count}, min={np.nanmin(sum_q_values):.3e}, max={np.nanmax(sum_q_values):.3e}")

        results.append(q_values)
        sum_results.append(sum_q_values)
        sample_names.append(f"sample_{idx}")

    # 将结果拼成 DataFrame，并**显式**使用 nodes.index 作为列名，保证顺序与 trunk_net 一致
    branch_df = pd.DataFrame(sum_results, columns=nodes.index)
    branch_df.index = sample_names
    branch_df.index.name = "sample_id"

    # 运行检查：列数是否与节点数一致
    print("\n=== 输出检查 ===")
    print(f"期望节点数: {n_nodes}")
    print(f"branch_df 列数: {len(branch_df.columns)}")
    print(f"branch_df 前 10 列名: {list(branch_df.columns[:10])}")
    print(f"是否包含 node_id=1: { (1 in branch_df.columns) }")
    # 查找缺失的 node_id（若你期望 node_id 连续，可做此检查）
    expected_ids = set(nodes.index)
    actual_ids = set(branch_df.columns)
    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids
    print(f"missing ids count: {len(missing)}, extra ids count: {len(extra)}")
    if len(missing) > 0:
        print("示例缺失 id（前10）:", list(sorted(missing))[:10])
    print("=================\n")

    # 最后保存
    branch_df.to_csv(out_branch)
    print(f"branch_net 已保存到: {out_branch}")

if __name__ == "__main__":
    compute_heat_flux(node_file, weld_file,out_branch, out_trunk)
