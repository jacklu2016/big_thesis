import numpy as np
import scipy.stats as stats
from scipy.integrate import dblquad
import matplotlib.pyplot as plt

import config
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 参数设置
p_i = config.p_i  # 零售价格
w = config.w  # 批发价格
h = config.h_r_h_v  # 库存持有成本
s = config.s  # 库存共享成本
delta = config.delta  # 竞争强度
mu = config.mu  # 正态分布均值
sigma = config.sigma  # 正态分布标准差
q_i = config.q_i  # 订购量
q_neg_i = config.q_minus_i - 10  # 竞争对手的订购量
t = 15
delta_range = np.linspace(0.2, 0.8, 10)  # 设定竞争强度
mean_d = 220

# 正态分布概率密度函数
def f_D(x, y):
    return stats.norm.pdf(x, mu, sigma) * stats.norm.pdf(y, mu, sigma)


# 销售利润部分 E{min[D, q]}
def sales_profit(q_i):
    # 假设D服从正态分布，计算 E{min[D, q_i]}

    return p_i * min(mean_d, q_i)


# 调拨部分积分：部分1，涉及两个积分项
def integral_part_1(delta, q_i, q_neg_i):
    # 计算第一个积分项
    def integrand1(x, y):
        return delta * (x - q_i) * f_D(x, y)

    integral1, _ = dblquad(integrand1, q_i, np.inf, lambda x: 0, lambda x: max(0, q_neg_i - delta * (x - q_i)))  # 限制 y 上限始终 >= 0
    #print('integral1 = ', integral1)
    return (p_i - t) * integral1


# 调拨部分积分：部分2，涉及第二个积分项
def integral_part_2(delta, q_i, q_neg_i):
    # 计算第二个积分项
    def integrand2(x, y):
        return (q_neg_i - y) * f_D(x, y)


    integral2, _ = dblquad(integrand2, q_i, np.inf, lambda x: q_neg_i - delta * (x - q_i), np.inf)
    #print('integral2 = ', integral2)
    return (p_i - t) * integral2


# 库存共享成本部分：部分1
def inventory_sharing_part_1(delta, q_i, q_neg_i):
    def integrand1(x, y):
        return (q_i - x) * f_D(x, y)

    integral1, _ = dblquad(integrand1, 0, q_i, lambda x: (1 / delta) * (q_i - x) + q_neg_i, np.inf)
    return (t - w - s) * integral1


# 库存共享成本部分：部分2
def inventory_sharing_part_2(delta, q_i, q_neg_i):
    def integrand2(x, y):
        return delta * (q_neg_i - y) * f_D(x, y)

    integral2, _ = dblquad(integrand2, 0, q_i, lambda x: q_neg_i, lambda x: (1 / delta) * (q_i - x) + q_neg_i)
    return (t - w - s) * integral2


# 库存持有成本部分
def holding_cost(q_i):
    return -w * q_i


# 总利润函数
def profit(delta, q_i, q_neg_i):
    part_1 = sales_profit(q_i)
    #D > q
    part_2 = abs(integral_part_1(delta, q_i, q_neg_i))
    part_3 = abs(integral_part_2(delta, q_i, q_neg_i))
    print(f'part_2 = {part_2}, part_3 = {part_3}')
    #D < q
    part_4 = abs(inventory_sharing_part_1(delta, q_i, q_neg_i))
    part_5 = abs(inventory_sharing_part_2(delta, q_i, q_neg_i))
    print(f'part_4 = {part_4}, part_5 = {part_5}')
    part_6 = holding_cost(q_i)

    if mean_d > q_i:
        total_profit = part_1 + part_2 + part_3 + part_6
    else:
        total_profit = part_1 + part_4 + part_5 + part_6
    return total_profit


# 计算不同调拨价格下的利润
profits_D_greater_than_q = [profit(delta, q_i, q_neg_i) for delta in delta_range]

t = 17
profits_D_greater_than_q1 = [profit(delta, q_i, q_neg_i) for delta in delta_range]
t = 19
profits_D_greater_than_q2 = [profit(delta, q_i, q_neg_i) for delta in delta_range]

#mean_d = 180
#profits_D_less_than_q = [profit(delta, q_i, q_neg_i) for delta in delta_range]

# 绘制图形
plt.figure(figsize=(8, 6))

# D > q的利润曲线
plt.subplot(1, 1, 1)
#plt.plot(delta_range, profits_D_greater_than_q, label="药店利润",marker='D', markersize=8,)
plt.plot(delta_range, profits_D_greater_than_q, label='调拨价格=15',color='blue',marker='v', markersize=8,)
plt.plot(delta_range, profits_D_greater_than_q1, label='调拨价格=17',color='yellow',marker='D', markersize=8,)
plt.plot(delta_range, profits_D_greater_than_q2, label='调拨价格=19',color='green',marker='>', markersize=8,)

plt.xlabel("竞争强度δ",fontsize=14)
plt.ylabel("药店利润",fontsize=14)
#plt.title("当需求量大于订购量时的药店利润")
plt.legend(fontsize=14)
plt.grid(True)

# D < q的利润曲线
#plt.subplot(1, 1, 1)
# plt.plot(delta_range, profits_D_less_than_q, label="D=180,q=200(需求量小于订购量)", color="orange",marker='^', markersize=8,)
# plt.xlabel("竞争强度δ",fontsize=14)
# plt.ylabel("药店利润",fontsize=14)
# #plt.title("当需求量小于订购量时的药店利润")
# plt.legend(fontsize=14)
# plt.grid(True)
#
plt.tight_layout()
plt.show()
