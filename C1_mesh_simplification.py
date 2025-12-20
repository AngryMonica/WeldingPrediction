import pandas as pd
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from pathlib import Path

data_dir = Path(r'D:\Users\MXY\PycharmProjects\data')
# ============================================================
# 输入路径
# ============================================================
nodes_path = data_dir/"nodes.csv"
triangles_path = data_dir/ "triangles.csv"
temperature_path = data_dir/"all_temperature_combined.csv"
stress_path = data_dir/"all_stress_combined.csv"

# ============================================================
# 输出路径
# ============================================================
sampled_nodes_path = data_dir/"sampled_nodes.csv"
sampled_triangles_path = data_dir/ "sampled_triangles.csv"
sampled_temperature_path = data_dir/ "sampled_temperature.csv"
sampled_stress_path = data_dir/"sampled_stress.csv"

# ============================================================
# [新增] 参数：区域定义 (请根据实际模型坐标修改)
# ============================================================
# 1. 焊缝区 (Weld Zone) - 最核心区域
WELD_X_MIN, WELD_X_MAX = -5.7, 5.7
WELD_Y_MIN, WELD_Y_MAX = 0, 10
WELD_Z_MIN, WELD_Z_MAX = 0.0, 100.0

# 2. 热影响区 (HAZ) - 包裹焊缝的过渡区域
HAZ_X_MIN, HAZ_X_MAX = -14.7, 14.7
HAZ_Y_MIN, HAZ_Y_MAX = 0, 10
HAZ_Z_MIN, HAZ_Z_MAX = 0.0, 100.0

# ============================================================
# [修改] 参数：三层简化策略 (保留比例)
# ============================================================
ratio_weld = 0.4  # 焊缝区：保留 95% (几乎不简化)
ratio_haz  = 0.5 # 热影响区：保留 40% (中等简化)
ratio_base = 0.8  # 母材区：保留 5% (大幅简化)

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

print(f"原始节点数量: {len(nodes)}")
print(f"原始三角面数量: {len(triangles)}")

# ============================================================
# 2. 建 Open3D mesh 并预处理
# ============================================================
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(nodes)
mesh.triangles = o3d.utility.Vector3iVector(triangles)
mesh.compute_vertex_normals()

# 基础清理
mesh.remove_duplicated_vertices()
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_triangles()
mesh.remove_non_manifold_edges()

# ============================================================
# 3. 三层分级网格简化 (修复版：基于面片分割 + 边界锁定)
# ============================================================
print("执行三层分级网格简化 (Weld -> HAZ -> Base)...")

vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)

# --- 3.1 计算三角面中心 (避免跨区域面片丢失) ---
# 形状: (N_triangles, 3, 3) -> mean -> (N_triangles, 3)
tri_centers = vertices[triangles].mean(axis=1)

# --- 3.2 定义区域掩码 (基于面片中心) ---
# 焊缝区掩码
mask_weld = (
    (tri_centers[:, 0] >= WELD_X_MIN) & (tri_centers[:, 0] <= WELD_X_MAX) &
    (tri_centers[:, 1] >= WELD_Y_MIN) & (tri_centers[:, 1] <= WELD_Y_MAX) &
    (tri_centers[:, 2] >= WELD_Z_MIN) & (tri_centers[:, 2] <= WELD_Z_MAX)
)

# 热影响区掩码
mask_haz_raw = (
    (tri_centers[:, 0] >= HAZ_X_MIN) & (tri_centers[:, 0] <= HAZ_X_MAX) &
    (tri_centers[:, 1] >= HAZ_Y_MIN) & (tri_centers[:, 1] <= HAZ_Y_MAX) &
    (tri_centers[:, 2] >= HAZ_Z_MIN) & (tri_centers[:, 2] <= HAZ_Z_MAX)
)
mask_haz = mask_haz_raw & (~mask_weld)

# 母材区掩码
mask_base = (~mask_weld) & (~mask_haz)

# --- 3.3 获取索引并分离网格 ---
idx_weld = np.where(mask_weld)[0]
idx_haz  = np.where(mask_haz)[0]
idx_base = np.where(mask_base)[0]

print(f"三角面分布 -> 焊缝区: {len(idx_weld)}, 热影响区: {len(idx_haz)}, 母材区: {len(idx_base)}")

def create_submesh(original_mesh, tri_indices):
    # 创建一个新的 mesh
    sub_mesh = o3d.geometry.TriangleMesh()
    # 复制所有顶点 (稍后清理)
    sub_mesh.vertices = original_mesh.vertices
    # 只选取属于该区域的三角面
    sub_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(original_mesh.triangles)[tri_indices])
    # 清理未引用的顶点 (这一步至关重要，去除不属于该区域的顶点)
    sub_mesh.remove_unreferenced_vertices()
    sub_mesh.compute_vertex_normals()
    return sub_mesh

mesh_weld = create_submesh(mesh, idx_weld)
mesh_haz  = create_submesh(mesh, idx_haz)
mesh_base = create_submesh(mesh, idx_base)

# --- 3.4 分别简化 (增加 boundary_weight 防止边界裂缝) ---
def simplify_submesh(sub_mesh, ratio, name):
    if len(sub_mesh.triangles) == 0:
        return sub_mesh
    target_n = int(len(sub_mesh.triangles) * ratio)
    print(f"  [{name}] 简化: {len(sub_mesh.triangles)} -> {target_n} (Ratio: {ratio})")
    if target_n > 0:
        # [关键修改] boundary_weight=100.0
        # 强行锁定边界顶点，确保不同区域拼接时边界能对齐
        return sub_mesh.simplify_quadric_decimation(target_n, boundary_weight=100.0)
    return sub_mesh

mesh_weld_simple = simplify_submesh(mesh_weld, ratio_weld, "Weld")
mesh_haz_simple  = simplify_submesh(mesh_haz,  ratio_haz,  "HAZ ")
mesh_base_simple = simplify_submesh(mesh_base, ratio_base, "Base")

# --- 3.5 合并与缝合 ---
print("合并网格并缝合边界...")
mesh_simplified = mesh_weld_simple + mesh_haz_simple + mesh_base_simple

# 移除重复顶点以缝合交界处
# 由于使用了 boundary_weight，边界顶点位置几乎未变，可以被成功合并
mesh_simplified.remove_duplicated_vertices()
mesh_simplified.remove_degenerate_triangles()
mesh_simplified.compute_vertex_normals()

new_nodes = np.asarray(mesh_simplified.vertices)
new_tris = np.asarray(mesh_simplified.triangles)

print("------------------------------------------------")
print(f"最终简化后节点数: {len(new_nodes)}")
print(f"最终简化后三角面数: {len(new_tris)}")
print("------------------------------------------------")
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
temp_node_cols = [c for c in temp_df.columns if c.startswith("N")]
temp_matrix = temp_df[temp_node_cols].values

N_new = len(new_nodes)
T = len(temp_df)
new_temp_matrix = np.zeros((T, N_new), dtype=float)

print("构建三邻点插值...")
dists, idxs = tree.query(new_nodes, k=3)

weights = 1 / (dists + 1e-12)
weights = weights / weights.sum(axis=1, keepdims=True)

for i in range(3):
    new_temp_matrix += temp_matrix[:, idxs[:, i]] * weights[:, i].reshape(1, -1)

out_df = temp_df[meta_cols].copy()
# 优化 DataFrame 构建速度
new_temp_df = pd.DataFrame(new_temp_matrix, columns=[f"node_{i}" for i in range(N_new)])
out_df = pd.concat([out_df, new_temp_df], axis=1)

out_df.to_csv(sampled_temperature_path, index=False)
print("温度插值完成。")

# ============================================================
# 7. 读取 stress.csv 并进行插值
# ============================================================
print("读取 stress.csv，执行应力插值...")

stress_df = pd.read_csv(stress_path)
stress_node_cols = [c for c in stress_df.columns if c.startswith("N")]
stress_matrix = stress_df[stress_node_cols].values

new_stress_matrix = np.zeros((T, N_new), dtype=float)

for i in range(3):
    new_stress_matrix += stress_matrix[:, idxs[:, i]] * weights[:, i].reshape(1, -1)

out_stress_df = stress_df[meta_cols].copy()
new_stress_df = pd.DataFrame(new_stress_matrix, columns=[f"N{i}" for i in range(N_new)])
out_stress_df = pd.concat([out_stress_df, new_stress_df], axis=1)

out_stress_df.to_csv(sampled_stress_path, index=False)
print("应力插值完成。")
