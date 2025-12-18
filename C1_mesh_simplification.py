import pandas as pd
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from pathlib import Path

data_dir = Path(r'D:\Users\MXY\PycharmProjects\data\t1s1')
# ============================================================
# 输入路径
# ============================================================
nodes_path = data_dir/"data/nodes.csv"
triangles_path =data_dir/ "data/triangles.csv"
temperature_path =data_dir/ "data/t1s1/temperature_t1s1.csv"
stress_path = data_dir/"data/t1s1/stress_t1s1_aligned.csv"
# ============================================================
# 输出路径
# ============================================================
sampled_nodes_path = data_dir/"data/t1s1/sampled_nodes.csv"
sampled_triangles_path =data_dir/ "data/t1s1/sampled_triangles.csv"
sampled_temperature_path =data_dir/ "data/t1s1/sampled_temperature.csv"
sampled_stress_path = data_dir/"data/t1s1/sampled_stress.csv"


# ============================================================
# 参数：简化后的目标三角面数 (根据需要调整)
# ============================================================
target_triangles = 16384  # 例如简化到 2 万个三角面

# ============================================================
# 1. 读取原始节点与三角面
# ============================================================
print("读取 nodes.csv 和 triangles.csv ...")

nodes_df = pd.read_csv(nodes_path, header=None, sep=r"[,\s]", engine="python")

nodes = nodes_df.values  # (N,3)

tri_raw = pd.read_csv(triangles_path, header=None, sep=r"[,\s]", engine="python").values
if len(tri_raw) % 3 != 0:
    raise ValueError("triangles.csv 行数必须为 3 的倍数")

triangles = tri_raw.reshape(-1, 3).astype(int)

print("节点数量:", len(nodes))
print("三角面数量:", len(triangles))

# ============================================================
# 2. 建 Open3D mesh
# ============================================================
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(nodes)
mesh.triangles = o3d.utility.Vector3iVector(triangles)
mesh.compute_vertex_normals()

mesh.remove_duplicated_vertices()
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_triangles()
mesh.remove_non_manifold_edges()


# ============================================================
# 3. 网格简化 (QEM)
# ============================================================
print("执行 QEM 网格简化中...")
mesh_simplified = mesh.simplify_quadric_decimation(target_triangles)
mesh_simplified.remove_duplicated_vertices()
mesh_simplified.remove_degenerate_triangles()

new_nodes = np.asarray(mesh_simplified.vertices)
new_tris = np.asarray(mesh_simplified.triangles)

print("简化后节点数:", len(new_nodes))
print("简化后三角面数:", len(new_tris))

# ============================================================
# 4. 将简化 mesh 写入 CSV
# ============================================================
pd.DataFrame(new_nodes).to_csv(sampled_nodes_path, header=False, index=False)
np.savetxt(sampled_triangles_path, new_tris.reshape(-1, 1), fmt="%d")

print("简化 mesh 输出完成。")

# ============================================================
# 5. 为后续插值建立 KDTree（用旧 mesh 的节点）
# ============================================================
tree = cKDTree(nodes)

# ============================================================
# 6. 读取 temperature.csv 并进行插值
# ============================================================
print("读取 temperature.csv，执行温度插值...")

temp_df = pd.read_csv(temperature_path)
meta_cols = ["test", "step", "increment","step_time"]

# 原温度列：node_0, node_1, ...
temp_node_cols = [c for c in temp_df.columns if c.startswith("N")]

# 原温度矩阵 (T, N_old)
temp_matrix = temp_df[temp_node_cols].values

# 新节点数量
N_new = len(new_nodes)
T = len(temp_df)

new_temp_matrix = np.zeros((T, N_new), dtype=float)



# 对每个新节点，找最近 3 个旧节点进行 barycentric-like 插值
print("构建三邻点插值...")
dists, idxs = tree.query(new_nodes, k=3)  # (N_new, 3)

# 防止除零
weights = 1 / (dists + 1e-12)
weights = weights / weights.sum(axis=1, keepdims=True)

# 插值
for i in range(3):
    new_temp_matrix += temp_matrix[:, idxs[:, i]] * weights[:, i].reshape(1, -1)

# 写出 sampled_temperature.csv
out_df = pd.DataFrame()
out_df[meta_cols] = temp_df[meta_cols]
# for i in range(N_new):
#     out_df[f"node_{i}"] = new_temp_matrix[:, i]
# 改写法
node_cols = [pd.Series(new_temp_matrix[:, i], name=f"node_{i}")
             for i in range(N_new)]
out_df = pd.concat([out_df] + node_cols, axis=1)

out_df.to_csv(sampled_temperature_path, index=False)
print("温度插值完成。")

# ============================================================
# 7. 读取 stress.csv 并进行插值
# ============================================================
print("读取 stress.csv，执行应力插值...")

stress_df = pd.read_csv(stress_path)
stress_node_cols = [c for c in stress_df.columns if c.startswith("N")]

# 原应力矩阵 (T, N_old)
stress_matrix = stress_df[stress_node_cols].values

# 新应力矩阵 (T, N_new)
new_stress_matrix = np.zeros((T, N_new), dtype=float)

# 插值
for i in range(3):
    new_stress_matrix += stress_matrix[:, idxs[:, i]] * weights[:, i].reshape(1, -1)

# 写出 sampled_stress.csv
out_stress_df = pd.DataFrame()
out_stress_df[meta_cols] = stress_df[meta_cols]
stress_node_cols = [pd.Series(new_stress_matrix[:, i], name=f"N{i}") for i in range(N_new)]
out_stress_df = pd.concat([out_stress_df] + stress_node_cols, axis=1)

out_stress_df.to_csv(sampled_stress_path, index=False)
print("应力插值完成。")