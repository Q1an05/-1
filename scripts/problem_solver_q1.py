import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scripts.resource_processor as rp

#文件路径
excel_file_path = 'resource/B题-支路车流量推测问题 附件(Attachment).xlsx'
#表名
table_name = "表1 (Table 1)"



#——————第一题——————


#--- 1. 读取文件 ---

df1 = rp.read_excel(table_name, excel_file_path)

#处理数据
try:
    # 将观测到的主路流量数据存为一个Numpy数组，方便后续计算
    F_data = df1['主路3的车流量 (Traffic flow on the Main road 3)'].to_numpy()
    # 创建时间数组 t, 从 0 到 59
    t = np.arange(len(F_data))
    print("数据加载成功！")
except Exception as e:
    print(f"数据加载失败: {e}")
    # 如果数据加载失败，退出程序
    exit()

# --- 2. 模型定义 --- 

# 支路1的模型: 线性增长 f1(t) = a*t + b
def model_branch1(params, t):
    a, b = params
    return a * t + b

# 支路2的模型: 先增后减的分段线性函数
def model_branch2(params, t, t_turn):
    c, d, m = params
    # 使用np.where实现分段，效率更高
    # 当 t <= t_turn 时，流量为 c*t + d
    # 当 t > t_turn 时，流量为 m*(t-t_turn) + (c*t_turn+d)
    return np.where(t <= t_turn, c * t + d, m * (t - t_turn) + (c * t_turn + d))

# 完整的总流量模型 F_model(t) = f1(t) + f2(t)
# 我们将所有参数合并到一个列表中，方便优化器使用
# params_all = [a, b, c, d, m]
def F_model(params_all, t, t_turn):
    # 从总参数列表中解包出两路的参数
    params1 = params_all[0:2] # a, b
    params2 = params_all[2:5] # c, d, m
    
    # 计算两条支路在所有时间点t的流量
    flow1 = model_branch1(params1, t)
    flow2 = model_branch2(params2, t, t_turn)
    
    # 返回总流量
    return flow1 + flow2

print("模型函数定义成功！")

#--- 3. 优化目标定义 ---

# 定义误差函数
def objective_function(params_all, t, t_turn, F_data):
    # 计算模型在当前参数下的预测值
    F_predicted = F_model(params_all, t, t_turn)
    
    # 计算并返回总平方误差
    return np.sum((F_predicted - F_data)**2)

print("误差函数定义成功！")

# --- 4. 遍历搜索与优化 ---

#这是第一题的总函数，用于解决第一题并可视化结果
def solve_problem_and_visualize_q1():

    # 初始化用于存储最佳结果的变量
    lowest_error = float('inf')
    best_t_turn = -1
    best_params_all = None
    optimization_successful = False

    # 定义参数的边界条件
    # [a, b, c, d, m]
    bounds = [
        (0, None),     # a >= 0 (增长)
        (0, None),     # b >= 0 (初始流量非负)
        (0, None),     # c >= 0 (增长)
        (0, None),     # d >= 0 (初始流量非负)
        (None, 0),     # m <= 0 (减少)
    ]

    print("开始执行遍历搜索... 这可能需要几十秒钟，请稍候。")

    # 遍历所有可能的 t_turn (1到58)
    for t_turn_candidate in range(1, 59):
        # 为参数提供一个初始猜测值
        initial_guess = [1, 1, 1, 1, -1]
        
        # 调用优化器
        result = minimize(
            objective_function,       # 要最小化的函数
            initial_guess,            # 初始猜测
            args=(t, t_turn_candidate, F_data), # 需要传递给函数的额外固定参数
            method='L-BFGS-B',        # 一种支持边界约束的优化算法
            bounds=bounds             # 参数的边界
        )
        
        # 检查优化是否成功，并且当前误差是否更小
        if result.success and result.fun < lowest_error:
            optimization_successful = True
            lowest_error = result.fun
            best_t_turn = t_turn_candidate
            best_params_all = result.x

    print("遍历搜索完成！")

    # --- 5. 结果展示与可视化 ---

    if optimization_successful:
        print("\n--- 最终优化结果 ---")
        print(f"最佳分段点 (t_turn): {best_t_turn}")
        print(f"对应时刻: {7 + (best_t_turn * 2) // 60:02d}:{(best_t_turn * 2) % 60:02d}")
        print(f"最小平方误差: {lowest_error:.4f}")
        
        a, b, c, d, m = best_params_all
        print("\n最优参数值:")
        print(f"  a = {a:.4f}, b = {b:.4f}")
        print(f"  c = {c:.4f}, d = {d:.4f}, m = {m:.4f}")
        
        print("\n--- 支路车流量函数表达式 ---")
        print(f"支路1: f1(t) = {a:.4f}*t + {b:.4f}")
        print(f"支路2: f2(t) = (当t<={best_t_turn}) {c:.4f}*t + {d:.4f}; (当t>{best_t_turn}) {m:.4f}*(t-{best_t_turn}) + {c*best_t_turn+d:.4f}")

        # --- 可视化 ---
        
        # 计算最终模型的预测流量
        F_predicted_final = F_model(best_params_all, t, best_t_turn)
        flow1_final = model_branch1(best_params_all[0:2], t)
        flow2_final = model_branch2(best_params_all[2:5], t, best_t_turn)
        
        plt.figure(figsize=(14, 8))
        plt.plot(t, F_data, 'o', label='Observed Data (Main Road 3)')
        plt.plot(t, F_predicted_final, color='red', linewidth=2, label='Fitted Total Flow (Model)')
        plt.plot(t, flow1_final, '--', color='green', label='Inferred Flow (Branch 1)')
        plt.plot(t, flow2_final, '--', color='orange', label='Inferred Flow (Branch 2)')
        
        # 在图上标记出分段点
        plt.axvline(x=best_t_turn, color='gray', linestyle=':', label=f'Split Point t={best_t_turn}')
        
        plt.title('Problem 1: Traffic Flow Fitting and Inference', fontsize=16)
        plt.xlabel('Time (t, where 7:00 is t=0)')
        plt.ylabel('Traffic Flow')
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.show()
        #保存图片
        try:
            plt.savefig('figs/problem_solver_q1.png')
        except Exception as e:
            print(f"保存图片失败: {e}")

    else:
        print("优化失败，未能找到有效解。")