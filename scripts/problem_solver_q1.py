import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import scripts.resource_processor as rp

# ==============================================================================
# 1. 定义各支路的车流量函数模型
#    Define Traffic Flow Models for Each Branch
# ==============================================================================

def F1_q1(t, p):
    """
    支路1: 线性增长
    Branch 1: Linear growth
    p: [a, b]
    """
    a, b = p
    t = np.asarray(t)
    flow = a * t + b
    # 确保流量非负
    return np.maximum(0, flow)

def F2_q1(t, p, t_turn):
    """
    支路2: 分段线性 (先增后减)
    Branch 2: Piecewise linear (growth then decrease)
    p: [c, d, m]
    """
    c, d, m = p
    t = np.asarray(t)
    # 使用np.where实现分段
    # 连续性约束在表达式中直接体现
    flow = np.where(t <= t_turn, c * t + d, m * (t - t_turn) + (c * t_turn + d))
    # 确保流量非负
    return np.maximum(0, flow)

def total_flow_model_q1(t, params, t_turn):
    """
    总车流量模型
    Total flow model combining all branches
    params: [a, b, c, d, m]
    """
    t = np.asarray(t)
    p_f1 = params[0:2]
    p_f2 = params[2:5]
    
    flow1 = F1_q1(t, p_f1)
    flow2 = F2_q1(t, p_f2, t_turn)
    
    return flow1 + flow2

def residuals_q1(params, t, F_data, t_turn):
    """
    残差函数，用于 least_squares 优化
    Residual function for least_squares optimization
    """
    model_predictions = total_flow_model_q1(t, params, t_turn)
    return model_predictions - F_data

# ==============================================================================
# 2. 可视化函数
#    Visualization Function
# ==============================================================================

def visualize_results_q1(t, F_data, F_fitted, f1_flow, f2_flow, t_turn):
    """绘制问题一的结果图"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 8))
    
    plt.plot(t, F_data, 'o', label='Observed Data (Main Road 3)', markersize=5)
    plt.plot(t, F_fitted, color='red', linewidth=2.5, label='Fitted Total Flow (Model)')
    plt.plot(t, f1_flow, '--', color='green', linewidth=2, label='Inferred Flow (Branch 1)')
    plt.plot(t, f2_flow, '--', color='orange', linewidth=2, label='Inferred Flow (Branch 2)')
    
    # 标记转折点
    plt.axvline(x=t_turn, color='gray', linestyle=':', label=f'Turning Point t={t_turn}')
    
    plt.title('Problem 1: Traffic Flow Fitting and Inference', fontsize=16)
    plt.xlabel('Time t (0 = 7:00, 1 unit = 2 mins)')
    plt.ylabel('Traffic Flow')
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig('figs/problem_solver_q1.png')
    plt.show()
    

# ==============================================================================
# 3. 主求解器函数
#    Main Solver Function
# ==============================================================================

def solve_and_visualize_problem1():
    """
    解决问题一的核心函数：读取数据、进行优化、打印结果并可视化
    Core function to solve Problem 1: reads data, optimizes, prints results, and visualizes.
    """
    #--- 1. 读取文件 ---
    excel_file_path = 'resource/B题-支路车流量推测问题 附件(Attachment).xlsx'
    table_name = "表1 (Table 1)"

    print(f"--- 问题一求解启动 ---")
    print(f"正在从 '{excel_file_path}' 读取工作表 '{table_name}'...")
    df1 = rp.read_excel(table_name, excel_file_path)

    if df1 is None:
        print("错误: 数据读取失败，程序终止。")
        return

    #--- 2. 处理数据 ---
    try:
        F_data = df1.iloc[:, 2].to_numpy()
        t_data = df1.iloc[:, 1].to_numpy()
        print("数据加载成功！")
    except Exception as e:
        print(f"错误: 数据处理失败: {e}")
        return

    # ------------------ 3. 优化过程 ------------------
    best_result = None
    min_error = np.inf
    
    print("开始优化过程，将遍历 t_turn 的可能值以寻找最优解...")
    # 遍历所有可能的转折点
    for t_turn_candidate in range(1, len(t_data) - 1):
        # 定义参数的初始猜测值和边界
        # params: [a, b, c, d, m]
        initial_params = [1, 1, 1, 1, -1]
        bounds = ([0, 0, 0, 0, -np.inf], # Lower bounds (a,b,c,d >= 0)
                  [np.inf, np.inf, np.inf, np.inf, 0]) # Upper bounds (m <= 0)

        result = least_squares(
            residuals_q1,
            initial_params,
            bounds=bounds,
            args=(t_data, F_data, t_turn_candidate),
            method='trf'
        )
        
        current_error = np.sum(result.fun**2)
        if result.success and current_error < min_error:
            min_error = current_error
            best_result = {'params': result.x, 't_turn': t_turn_candidate, 'error': min_error}
            print(f"  找到新的更优解: t_turn = {t_turn_candidate}, 均方根误差(RMSE) = {np.sqrt(min_error / len(F_data)) :.4f}")

    print("优化完成！")
    
    # ------------------ 4. 结果处理与输出 ------------------
    if best_result is None:
        print("优化失败，未能找到有效解。")
        return
        
    final_params = best_result['params']
    final_t_turn = best_result['t_turn']
    
    a, b, c, d, m = final_params
    
    print("\n" + "="*50)
    print("         问题一：各支路车流量函数推测结果")
    print("="*50)
    print(f"最优转折点 t_turn: {final_t_turn} (对应时刻: {7 + (final_t_turn * 2) // 60:02d}:{(final_t_turn * 2) % 60:02d})")
    
    print(f"\n支路1 (F1): 线性增长")
    print(f"  F1(t) = {a:.4f}*t + {b:.4f}")
    
    # 为了显示表达式，计算出分段点的值
    turn_point_value = c * final_t_turn + d
    
    print(f"\n支路2 (F2): 分段线性")
    print(f"  F2(t) = {c:.4f}*t + {d:.4f},  for t <= {final_t_turn}")
    print(f"  F2(t) = {m:.4f}*(t - {final_t_turn}) + {turn_point_value:.4f},  for t > {final_t_turn}")
    print("="*50)
    
    # ------------------ 5. 可视化 ------------------
    F_fitted = total_flow_model_q1(t_data, final_params, final_t_turn)
    f1_flow = F1_q1(t_data, final_params[0:2])
    f2_flow = F2_q1(t_data, final_params[2:5], final_t_turn)
    
    visualize_results_q1(t_data, F_data, F_fitted, f1_flow, f2_flow, final_t_turn)