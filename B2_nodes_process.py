import csv
from pathlib import Path

data_dir = Path(r'D:\Users\MXY\PycharmProjects\data\t1s1')
# -------------------------------
# 读取 nodes.txt
# -------------------------------
node_coords = {}
with open(data_dir/ "data/nodes.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        nid, x, y, z = line.split(",")
        node_coords[int(nid)] = (float(x), float(y), float(z))

# ---------------------------------------------------
# 读取 elements.txt：每 8 行构成一个 C3D8 单元
# ---------------------------------------------------
elements = []
with open(data_dir/"data/elements.txt", "r") as f:
    current = []
    last_eid = None
    for line in f:
        line = line.strip()
        if not line:
            continue
        eid, nid = line.split(",")
        eid = int(eid)
        nid = int(nid)

        if last_eid is None:
            last_eid = eid

        if eid != last_eid:
            if len(current) != 8:
                raise ValueError("一个单元没有读到 8 个节点，请检查 elements.txt 格式")
            elements.append(current)
            current = []
            last_eid = eid

        current.append(nid)

    if len(current) == 8:
        elements.append(current)

# ---------------------------------------------------
# 六面体 6 个面四个节点的顺序
# ---------------------------------------------------
faces = [
    (0, 1, 2, 3),  # front
    (4, 5, 6, 7),  # back
    (0, 4, 5, 1),  # left
    (1, 5, 6, 2),  # bottom
    (2, 6, 7, 3),  # right
    (3, 7, 4, 0),  # top
]

# ---------------------------------------------------
# 生成三角形并去重
# ---------------------------------------------------
triangles = []
tri_set = set()

for elem in elements:
    for a, b, c, d in faces:
        nA, nB, nC, nD = elem[a], elem[b], elem[c], elem[d]

        tri1 = (nA, nB, nC)
        tri2 = (nA, nC, nD)

        key1 = tuple(sorted(tri1))
        key2 = tuple(sorted(tri2))

        if key1 not in tri_set:
            tri_set.add(key1)
            triangles.append(tri1)

        if key2 not in tri_set:
            tri_set.add(key2)
            triangles.append(tri2)

# ---------------------------------------------------
# 输出 nodes.csv（不改变顺序，只写坐标）
# ---------------------------------------------------
with open(data_dir/"data/nodes.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for nid in sorted(node_coords.keys()):
        x, y, z = node_coords[nid]
        writer.writerow([x, y, z])

# ---------------------------------------------------
# 输出 node_index.csv（改为 0-based，每个索引独立一行）
# ---------------------------------------------------
with open(data_dir/"data/triangles.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for tri in triangles:
        writer.writerow([tri[0] - 1])
        writer.writerow([tri[1] - 1])
        writer.writerow([tri[2] - 1])

print("已生成 nodes.csv 与 triangles.csv（索引为 0-based，每个节点独立一行）")
