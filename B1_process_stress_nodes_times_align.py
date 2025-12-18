import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_element_node_mapping(elements_file):
    node_to_elems = defaultdict(list)
    with open(elements_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                elem_label = int(parts[0])
                node_label = int(parts[1])
                node_to_elems[node_label].append(elem_label)
    print(f"映射了 {len(node_to_elems)} 个节点")
    return node_to_elems

def compute_node_stress_from_elements(stress_df, node_to_elems, target_nodes, elem_labels):
    meta_cols = ['test', 'step', 'increment', 'step_time']
    meta_df = stress_df[meta_cols].copy()

    node_data = {}
    for node in target_nodes:
        elems = node_to_elems.get(node, [])
        valid_cols = [elem_labels[e] for e in elems if e in elem_labels]
        if valid_cols:
            node_data[f'N{node}'] = stress_df[valid_cols].mean(axis=1).values
        else:
            node_data[f'N{node}'] = np.full(len(stress_df), np.nan)

    nodes_df = pd.DataFrame(node_data, index=stress_df.index)
    return pd.concat([meta_df, nodes_df], axis=1)

def align_stress_by_time_vectorized(temp_df, stress_node_df):
    node_cols = [c for c in stress_node_df.columns if c.startswith('N')]

    results = []
    grouped_temp = temp_df.groupby(['test', 'step'])

    for (test_val, step_val), temp_group in grouped_temp:
        target_test_name = f"{test_val}_stress"
        target_step_name = f"Step-{step_val.split('-')[-1].lower()}" if '-' in str(step_val) else f"Step-{str(step_val).lower()}"

        mask = (stress_node_df['test'] == target_test_name) & (stress_node_df['step'] == target_step_name)
        stress_group = stress_node_df.loc[mask].copy()

        if len(stress_group) == 0:
            print(f"警告: 找不到 test={test_val}, step={step_val} 的应力数据", file=sys.stderr)
            result = temp_group[['test', 'step', 'increment', 'step_time']].copy()
            for col in node_cols:
                result[col] = np.nan
            results.append(result)
            continue

        temp_sorted = temp_group.sort_values('step_time').copy()
        stress_sorted = stress_group.sort_values('step_time').copy()

        merged = pd.merge_asof(
            temp_sorted[['test', 'step', 'increment', 'step_time']],
            stress_sorted[['step_time'] + node_cols],
            on='step_time',
            direction='nearest'
        )
        results.append(merged)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

def main():
    data_dir = Path(r'D:\Users\MXY\PycharmProjects\data\t1s1')
    temp_file = data_dir / 'temperature_t1s1.csv'
    stress_file = data_dir / 'stress_t1s1.csv'
    elements_file = data_dir / 'elements.txt'
    output_file = data_dir / 'stress_t1s1_aligned_quick.csv'

    print("读取单元-节点映射...")
    node_to_elems = load_element_node_mapping(elements_file)

    print("读取温度数据...")
    temp_df = pd.read_csv(temp_file)
    print(f"温度数据: {temp_df.shape[0]} 行, {temp_df.shape[1]} 列")

    target_nodes = [int(c[1:]) for c in temp_df.columns if c.startswith('N')]
    print(f"目标节点数: {len(target_nodes)}")

    print("读取应力数据...")
    stress_df = pd.read_csv(stress_file, low_memory=False)
    print(f"应力数据: {stress_df.shape[0]} 行, {stress_df.shape[1]} 列")

    elem_cols = [c for c in stress_df.columns if c.startswith('E')]
    elem_labels = {int(c[1:]): c for c in elem_cols}
    print(f"单元数: {len(elem_labels)}")

    print("转换应力数据（单元->节点）...")
    stress_node_df = compute_node_stress_from_elements(stress_df, node_to_elems, target_nodes, elem_labels)
    print(f"转换完成: {stress_node_df.shape}")

    del stress_df

    print("对齐应力数据到温度数据...")
    aligned_df = align_stress_by_time_vectorized(temp_df, stress_node_df)
    print(f"对齐完成: {aligned_df.shape}")

    print(f"保存到 {output_file}...")
    aligned_df.to_csv(output_file, index=False)

    print("完成!")
    print(aligned_df.head())

if __name__ == '__main__':
    main()
