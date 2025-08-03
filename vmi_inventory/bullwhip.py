import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# =====================
# 解决中文显示问题
# =====================
# 设置中文字体（根据系统安装的字体选择）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 微软雅黑/黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# =====================
# 供应链牛鞭效应模拟器
# =====================

# 参数设置
np.random.seed(42)
num_periods = 50               # 模拟周期数
base_demand = 12               # 基础需求（对应图片中的12 units）
demand_variation = 0.2         # 需求波动率（20%）
supply_chain_levels = 4        # 供应链层级：顾客 -> 零售商 -> 分销商 -> 制造商

# 初始化需求矩阵（行：时间周期，列：供应链层级）
demands = np.zeros((num_periods, supply_chain_levels))
demands[:, 0] = base_demand * (1 + demand_variation * np.random.randn(num_periods))  # 顾客需求

# 定义各层级订单放大规则
def amplify_order(prev_demand, level):
    """模拟层级订单放大逻辑"""
    # 安全库存系数随层级递增（牛鞭效应核心机制）
    safety_factor = 1 + 0.3 * level
    # 预测误差放大（移动平均法）
    forecast_window = max(3 - level, 1)  # 上游使用更长预测窗口
    ma = np.convolve(prev_demand, np.ones(forecast_window)/forecast_window, mode='valid')
    return safety_factor * ma[-1] if len(ma) > 0 else prev_demand[-1]

# 模拟供应链需求传递
for t in range(1, num_periods):
    for level in range(1, supply_chain_levels):
        # 每个层级的订单基于前一层级历史需求计算
        demands[t, level] = amplify_order(demands[:t, level-1], level)

# ===============
# 可视化结果
# ===============
plt.figure(figsize=(12, 6))
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
labels = ['顾客需求', '零售商订单', '分销商订单', '制造商订单']

for level in range(supply_chain_levels):
    plt.plot(demands[:, level],
             color=colors[level],
             linewidth=2 if level==0 else 1.5,
             linestyle='--' if level>0 else '-',
             label=labels[level])

plt.title("供应链牛鞭效应模拟（层级需求放大）", fontsize=14, pad=20)
plt.xlabel("时间周期", fontsize=12)
plt.ylabel("订单量", fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

# 添加注释说明关键机制
plt.annotate('需求波动逐级放大',
             xy=(15, demands[15, 3]),
             xytext=(20, demands[15, 3]*1.5),
             arrowprops=dict(arrowstyle="->", color='grey'),
             fontsize=10)

plt.tight_layout()
plt.show()