import pandas as pd
import numpy as np
import pyvista as pv
import pandas as pd
import numpy as np
import pyvista as pv

def plot_heat_flux_volume(trunk_file, temp_file, timestep=0, cmap="turbo", low_value_color="lightgray"):
    print("📘 正在加载节点坐标与热流密度数据...")
    nodes = pd.read_csv(trunk_file, index_col=0)
    if 'pred' in temp_file:
        temp = pd.read_csv(temp_file)
    else:
        temp = pd.read_csv(temp_file, index_col=0)

    temp.drop(columns=['test','step', 'increment'], inplace=True)


    print(f"📈 当前绘制时间步: {timestep}")
    q_values = temp.iloc[timestep].values
    x, y, z = nodes["x"].values, nodes["y"].values, nodes["z"].values

    print(f"✅ 读取完成，共 {len(nodes)} 个节点")

    # 清理异常值
    q_values = np.nan_to_num(q_values, nan=0.0, posinf=0.0, neginf=0.0)
    q_values[np.abs(q_values) < 1e-6] = 0
    zero_ratio = np.sum(q_values == 0) / len(q_values)
    print(f"🧮 零值比例: {zero_ratio:.2%}")

    # 归一化
    q_min, q_max = np.min(q_values), np.max(q_values)
    if q_max > q_min:
        q_norm = (q_values - q_min) / (q_max - q_min)
    else:
        q_norm = q_values

    points = np.column_stack((x, y, z))
    point_cloud = pv.PolyData(points)
    point_cloud["temp"] = q_norm

    plotter = pv.Plotter()

    # ✅ 如果大部分为零 → 灰色显示
    if zero_ratio > 0.9 or np.allclose(q_values, 0):
        print("⚪ 大部分区域热流密度为0，显示为灰色点云。")
        plotter.add_points(point_cloud, color=low_value_color, point_size=4.0, render_points_as_spheres=True)

    else:
        nonzero_mask = q_values > 0
        nonzero_points = points[nonzero_mask]
        nonzero_q = q_norm[nonzero_mask]

        if len(nonzero_points) < 10:
            print("⚠️ 非零点太少，使用散点渲染。")
            plotter.add_points(pv.PolyData(nonzero_points), scalars=nonzero_q, cmap=cmap,
                               point_size=6.0, render_points_as_spheres=True)
        else:
            print(f"🧩 正在生成体网格...（非零节点 {len(nonzero_points)}）")
            grid = pv.PolyData(nonzero_points).delaunay_3d(alpha=5.0)

            # ✅ 手动附加标量（防止 KeyError）
            grid.point_data["temp"] = nonzero_q

            # 检查类型
            print(f"📦 网格类型: {type(grid)}")

            if isinstance(grid, pv.PolyData):
                print("⚠️ delaunay_3d 退化为表面网格，使用 add_mesh 绘制。")
                plotter.add_mesh(grid, scalars="temp", cmap=cmap)
            else:
                print("🎨 绘制体渲染中...")
                plotter.add_volume(grid, scalars="temp", cmap=cmap, opacity="sigmoid_5", shade=True)

        # 绘制零值区域为灰色点
        zero_points = points[q_values == 0]
        if len(zero_points) > 0:
            zero_cloud = pv.PolyData(zero_points)
            plotter.add_points(zero_cloud, color=low_value_color, point_size=3.0, render_points_as_spheres=True)

    plotter.add_scalar_bar(title="Normalized Temp", n_labels=5)
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()
    plotter.save_graphic('graphic')


    print("✅ 绘制完成。")

def plot_temp_fields(trunk_file, pred_file, real_file, timestep=0, cmap="turbo", low_value_color="lightgray"):
    """
    绘制预测温度场、实际温度场及残差场
    :param trunk_file: 节点坐标文件
    :param pred_file: 预测温度文件
    :param real_file: 实际温度文件
    :param timestep: 时间步
    :param cmap: 颜色映射
    :param low_value_color: 零值或低值颜色
    """
    print("📘 正在加载节点坐标与温度数据...")
    nodes = pd.read_csv(trunk_file, index_col=0)
    pred = pd.read_csv(pred_file)
    # if pred.columns[0] != 'index':
    #     pred.insert(0, 'index', pred.index)
    # pred.set_index('index', inplace=True)
    real = pd.read_csv(real_file, index_col=0)

    # 清理无关列
    for df in [pred, real]:
        df.drop(columns=[col for col in ['test','step','increment'] if col in df.columns], inplace=True)

    print(f"📈 当前绘制时间步: {timestep}")
    pred_vals = pred.iloc[timestep].values
    real_vals = real.iloc[timestep].values
    residual = pred_vals - real_vals

    x, y, z = nodes["x"].values, nodes["y"].values, nodes["z"].values
    points = np.column_stack((x, y, z))

    # 清理异常值
    for arr in [pred_vals, real_vals, residual]:
        np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    # 统一颜色范围（以实际场为基准）
    vmin, vmax = np.nanmin(real_vals), np.nanmax(real_vals)
    # real_min,real_vmax=np.nanmin(real_vals), np.nanmax(real_vals)
    # pred_min,pred_vmax=np.nanmin(pred_vals), np.nanmax(pred_vals)
    # vmin=min(real_min,pred_min)
    # vmax=max(real_vmax,pred_vmax)
    # 单窗口 1×3 子图布局
    plotter = pv.Plotter(shape=(1, 3))

    def render_field_subplot(plotter, col, points, values, title, clim, scalar_name="temp", cmap_local=cmap):
        plotter.subplot(0, col)
        plotter.add_text(title, font_size=10)

        # 零值比例与掩码
        zero_ratio = float(np.sum(values == 0)) / len(values) if len(values) > 0 else 1.0
        nonzero_mask = values != 0
        nonzero_points = points[nonzero_mask]
        nonzero_vals = values[nonzero_mask]

        # 大部分为零 → 灰点显示
        if zero_ratio > 0.9 or np.allclose(values, 0):
            cloud = pv.PolyData(points)
            plotter.add_points(cloud, color=low_value_color, point_size=4.0, render_points_as_spheres=True)
        else:
            if len(nonzero_points) < 10:
                # 非零点太少，用散点
                plotter.add_points(pv.PolyData(nonzero_points), scalars=nonzero_vals, cmap=cmap_local,
                                   point_size=6.0, render_points_as_spheres=True)
            else:
                # 生成体或表面网格（与 plot_heat_flux_volume 保持一致风格）
                try:
                    grid = pv.PolyData(nonzero_points).delaunay_3d(alpha=5.0)
                    grid.point_data[scalar_name] = nonzero_vals
                    if isinstance(grid, pv.PolyData):
                        plotter.add_mesh(grid, scalars=scalar_name, cmap=cmap_local, clim=clim)
                    else:
                        plotter.add_volume(grid, scalars=scalar_name, cmap=cmap_local, clim=clim,
                                           opacity="sigmoid_5", shade=True)
                except Exception:
                    # 三角剖分异常时回退到散点
                    plotter.add_points(pv.PolyData(nonzero_points), scalars=nonzero_vals, cmap=cmap_local,
                                       point_size=6.0, render_points_as_spheres=True)

            # 叠加零值点为灰色
            zero_points = points[~nonzero_mask]
            if len(zero_points) > 0:
                zero_cloud = pv.PolyData(zero_points)
                plotter.add_points(zero_cloud, color=low_value_color, point_size=3.0, render_points_as_spheres=True)

        plotter.add_scalar_bar(title=scalar_name.capitalize())
        plotter.add_axes()
        plotter.show_grid()

    # 预测温度场（颜色范围与实际一致）
    render_field_subplot(plotter, 0, points, pred_vals, "预测温度场", clim=[vmin, vmax], scalar_name="temp", cmap_local=cmap)

    # 实际温度场
    render_field_subplot(plotter, 1, points, real_vals, "实际温度场", clim=[vmin, vmax], scalar_name="temp", cmap_local=cmap)

    # 残差场（颜色范围对称）
    max_abs = float(max(np.nanmax(np.abs(residual)), 1e-8))
    render_field_subplot(plotter, 2, points, residual, "残差场（预测-实际）", clim=[-max_abs, max_abs], scalar_name="residual", cmap_local="coolwarm")

    # 统一视角与交互
    plotter.link_views()
    plotter.view_isometric()
    plotter.show()

    print("✅ 三场绘制完成。")

# 示例调用
if __name__ == "__main__":
    # plot_heat_flux_volume("trunk_net_K_sampled.csv", "predicted_temperature.csv", timestep=1,
    #                       cmap="turbo", low_value_color="lightgray")

    plot_temp_fields("trunk_net_K_sampled.csv",
                     "predicted_temperature.csv",
                     "merged_all_time_points_K_sampled.csv",
                     timestep=1000,
                     cmap="turbo",
                     low_value_color="lightgray")
