import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import scripts.resource_processor as rp 

# ==============================================================================
# 1. 定义各支路的车流量函数模型
#    Define Traffic Flow Models for Each Branch
# ==============================================================================

def F1(t, p):
    """
    支路1: 稳定车流
    Branch 1: Stable flow
    p: [c1] - 稳定流量 (stable flow value)
    """
    c1 = p[0]
    # 修正: 确保函数返回与输入t相同维度的数组
    # FIX: Ensure the function returns an array with the same shape as input t
    t = np.asarray(t)
    flow = np.full_like(t, c1, dtype=float)
    # 确保流量非负
    return np.maximum(0, flow)

def F2(t, p):
    """
    支路2: 分段函数 (线性增长 -> 稳定 -> 线性增长)
    Branch 2: Piecewise function (linear growth -> stable -> linear growth)
    p: [a2, b2, a3]
        a2: 第一段增长率 (growth rate for the first part)
        b2: 第一段初始流量 (initial flow for the first part)
        a3: 第二段增长率 (growth rate for the second part)
    """
    a2, b2, a3 = p
    t = np.asarray(t)
    flow = np.zeros_like(t, dtype=float)

    # 时间段定义 (Time intervals based on problem description)
    # 6:58 -> t=-1, 7:48 -> t=24, 8:14 -> t=37, 8:58 -> t=59
    
    # 第一段: [6:58, 7:48] -> t in [-1, 24]
    mask1 = (t >= -1) & (t <= 24)
    flow[mask1] = a2 * t[mask1] + b2

    # 连续性约束: 计算并设置稳定期流量
    c2 = a2 * 24 + b2
    mask2 = (t > 24) & (t < 37)
    flow[mask2] = c2

    # 连续性约束: 计算第三段的截距
    b3 = c2 - a3 * 37
    mask3 = (t >= 37) & (t <= 59)
    flow[mask3] = a3 * t[mask3] + b3

    # 确保所有流量非负
    return np.maximum(0, flow)

def F3(t, p, t3_changepoint):
    """
    支路3: 分段函数 (线性增长 -> 稳定)
    Branch 3: Piecewise function (linear growth -> stable)
    p: [a4, b4]
    """
    a4, b4 = p
    t = np.asarray(t)
    flow = np.zeros_like(t, dtype=float)

    mask1 = t <= t3_changepoint
    flow[mask1] = a4 * t[mask1] + b4

    # 连续性约束
    c3 = a4 * t3_changepoint + b4
    mask2 = t > t3_changepoint
    flow[mask2] = c3
    
    return np.maximum(0, flow)

def F4(t, p):
    """
    支路4: 周期性规律 (正弦函数)
    Branch 4: Periodic flow (sinusoidal function)
    p: [A, omega, phi, B]
    """
    A, omega, phi, B = p
    flow = A * np.sin(omega * t + phi) + B
    return np.maximum(0, flow)

def total_flow_model(t, params, t3_changepoint):
    """总车流量模型"""
    t = np.asarray(t)
    p_f1, p_f2, p_f3, p_f4 = [params[0]], params[1:4], params[4:6], params[6:10]
    flow1 = F1(t - 1, p_f1)
    flow2 = F2(t - 1, p_f2)
    flow3 = F3(t, p_f3, t3_changepoint)
    flow4 = F4(t, p_f4)
    return flow1 + flow2 + flow3 + flow4

def residuals(params, t_data, M_data, t3_changepoint):
    """残差函数，用于优化"""
    model_predictions = total_flow_model(t_data[1:], params, t3_changepoint)
    return model_predictions - M_data[1:]

# ==============================================================================
# 2. 可视化函数
#    Visualization Function
# ==============================================================================

def visualize_results(t_full, M_data, M_fitted, f1_flow, f2_flow, f3_flow, f4_flow):
    """绘制所有结果图"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(3, 1, figsize=(14, 18), sharex=True)

    # 图1: 主路流量拟合情况
    axs[0].plot(t_full, M_data, 'o', label='Original Main Road Data', markersize=4)
    axs[0].plot(t_full[1:], M_fitted, '-', label='Fitted Total Flow Model', linewidth=2)
    axs[0].set_ylabel('Traffic Flow')
    axs[0].set_title('Main Road (Road 5) Traffic Flow Analysis')
    axs[0].legend()
    axs[0].grid(True)

    # 图2: 趋势部分 (支路1, 2, 3)
    axs[1].plot(t_full, f1_flow, label='Branch 1 Flow (F1 - Stable)', linewidth=2)
    axs[1].plot(t_full, f2_flow, label='Branch 2 Flow (F2 - Piecewise)', linewidth=2)
    axs[1].plot(t_full, f3_flow, label='Branch 3 Flow (F3 - Piecewise)', linewidth=2)
    axs[1].set_ylabel('Traffic Flow')
    axs[1].set_title('Deduced Trend-Based Branch Flows')
    axs[1].legend()
    axs[1].grid(True)

    # 图3: 周期部分 (支路4)
    axs[2].plot(t_full, f4_flow, label='Branch 4 Flow (F4 - Periodic)', color='purple', linewidth=2)
    axs[2].set_xlabel('Time t (0 = 7:00, 1 unit = 2 mins)')
    axs[2].set_ylabel('Traffic Flow')
    axs[2].set_title('Deduced Periodic Branch Flow')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

# ==============================================================================
# 3. 主求解器函数
#    Main Solver Function
# ==============================================================================

def solve_and_visualize_problem2():
    """
    解决问题二的核心函数：读取数据、进行优化、打印结果并可视化
    Core function to solve Problem 2: reads data, optimizes, prints results, and visualizes.
    """
    #--- 1. 读取文件 ---
    # 文件路径
    excel_file_path = 'resource/B题-支路车流量推测问题 附件(Attachment).xlsx'
    # 表名
    table_name = "表2 (Table 2)"

    print(f"--- 问题二求解启动 ---")
    print(f"正在从 '{excel_file_path}' 读取工作表 '{table_name}'...")
    df2 = rp.read_excel(table_name, excel_file_path)

    if df2 is None:
        print("错误: 数据读取失败，程序终止。")
        return

    #--- 2. 处理数据 ---
    try:
        # 将观测到的主路流量数据存为一个Numpy数组
        M_data = df2.iloc[:, 2].to_numpy()
        # 创建时间数组 t, 从 0 到 59
        t_data = df2.iloc[:, 1].to_numpy()
        print("数据加载成功！")
    except Exception as e:
        print(f"错误: 数据处理失败: {e}")
        # 如果数据加载失败，退出程序
        return

    # ------------------ 3. 优化过程 ------------------
    best_result = None
    min_error = np.inf
    
    print("开始优化过程，将遍历 t3 的可能值以寻找最优解...")
    for t3_candidate in range(10, 50):
        # 定义参数的初始猜测值和边界
        initial_params = [5, 0.8, 10, 0.1, 1.5, 5, 10, 2*np.pi/8, 0, 15]
        bounds = ([0, 0, 0, 0, 0, 0, 0, 2*np.pi/12, -np.pi, 0],
                  [50, 5, 50, 5, 5, 50, 50, 2*np.pi/4, np.pi, 50])

        result = least_squares(residuals, initial_params, bounds=bounds,
                               args=(t_data, M_data, t3_candidate), method='trf')
        
        current_error = np.sum(result.fun**2)
        if current_error < min_error:
            min_error = current_error
            # 修正: 使用 result.x 获取优化后的参数
            # FIX: Use result.x to get the optimized parameters
            best_result = {'params': result.x, 't3': t3_candidate, 'error': min_error}
            print(f"  找到新的更优解: t3 = {t3_candidate}, 均方根误差(RMSE) = {np.sqrt(min_error / len(M_data)) :.4f}")

    print("优化完成！")
    
    # ------------------ 4. 结果处理与输出 ------------------
    final_params = best_result['params']
    final_t3 = best_result['t3']
    
    p_f1, p_f2, p_f3, p_f4 = [final_params[0]], final_params[1:4], final_params[4:6], final_params[6:10]
    
    print("\n" + "="*50)
    print("         问题二：各支路车流量函数推测结果")
    print("="*50)
    print(f"支路1 (F1): 稳定车流")
    print(f"  F1(t) = {p_f1[0]:.4f}")
    
    c2 = p_f2[0] * 24 + p_f2[1]
    b3 = c2 - p_f2[2] * 37
    print(f"\n支路2 (F2): 分段函数")
    print(f"  F2(t) = {p_f2[0]:.4f}*t + {p_f2[1]:.4f},  for t in [-1, 24]")
    print(f"  F2(t) = {c2:.4f},  for t in (24, 37)")
    print(f"  F2(t) = {p_f2[2]:.4f}*t + {b3:.4f},  for t in [37, 59]")
    
    c3 = p_f3[0] * final_t3 + p_f3[1]
    print(f"\n支路3 (F3): 分段函数 (最优转折点 t3 = {final_t3})")
    print(f"  F3(t) = {p_f3[0]:.4f}*t + {p_f3[1]:.4f},  for t in [0, {final_t3}]")
    print(f"  F3(t) = {c3:.4f},  for t > {final_t3}")
    
    print(f"\n支路4 (F4): 周期函数")
    print(f"  F4(t) = {p_f4[0]:.4f}*sin({p_f4[1]:.4f}*t + {p_f4[2]:.4f}) + {p_f4[3]:.4f}")
    
    t_full_range = np.arange(-1, 60)
    f1_res, f2_res, f3_res, f4_res = (F1(t_full_range, p_f1), F2(t_full_range, p_f2),
                                      F3(t_full_range, p_f3, final_t3), F4(t_full_range, p_f4))
    
    t_730_idx, t_830_idx = 15 + 1, 45 + 1
    
    print("\n" + "="*50)
    print("         特定时刻各支路车流量数值")
    print("="*50)
    print(f"时刻 7:30 (t=15):")
    print(f"  支路1: {f1_res[t_730_idx]:.4f}, 支路2: {f2_res[t_730_idx]:.4f}, 支路3: {f3_res[t_730_idx]:.4f}, 支路4: {f4_res[t_730_idx]:.4f}")
    
    print(f"\n时刻 8:30 (t=45):")
    print(f"  支路1: {f1_res[t_830_idx]:.4f}, 支路2: {f2_res[t_830_idx]:.4f}, 支路3: {f3_res[t_830_idx]:.4f}, 支路4: {f4_res[t_830_idx]:.4f}")
    print("="*50)

    # ------------------ 5. 可视化 ------------------
    M_fitted = total_flow_model(t_data[1:], final_params, final_t3)
    visualize_results(t_data, M_data, M_fitted, f1_res[1:], f2_res[1:], f3_res[1:], f4_res[1:])