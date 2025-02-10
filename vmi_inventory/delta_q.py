import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.integrate as integrate
import p_to_order_quantity as pq
import config
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 调拨价格 t 的范围
t = 15
delta_range = np.linspace(0.2, 0.8, 10)  # 设定竞争强度

# 参数设置
w = config.w# 批发价格
p = config.p_i  # 零售价格
h_r_h_v = config.h_r_h_v  # 库存持有成本
s = config.s  # 库存共享成本
delta = config.delta  # 竞争强度
mu = config.mu  # 正态分布均值
sigma = config.sigma  # 正态分布标准差


# 正态分布概率密度函数 f(x,y)
def f_X(x):
    return norm.pdf(x, loc=mu, scale=sigma)


def f_Y(y):
    return norm.pdf(y, loc=mu, scale=sigma)


# 联合概率密度函数 f(x,y)
def f_D(x, y):
    return f_X(x) * f_Y(y)


# 计算 F(q) 中的积分部分
def calc_integral_part(q, t, w, delta, s):
    # 第一部分的积分
    def integrand1(y, x):
        return f_D(x, y)

    integral1, _ = integrate.dblquad(
        integrand1, 0, q, lambda x: (q - x) / delta + q, lambda x: np.inf
    )

    # 第二部分的积分
    def integrand2(y, x):
        return f_D(x, y)

    integral2, _ = integrate.dblquad(
        integrand2, 0, q, lambda x: q, lambda x: (q - x) / delta + q
    )

    # 返回最终的结果
    return (p - w - (p - t) * delta * integral2) + (t - w - s) *  integral1


# 计算订单量 q
def calc_order_quantity(delta_arr):
    q_values = []
    for delta_val in delta_arr:
        perception = calc_integral_part(200, t, w, delta_val, s) / p

        q = pq.p_to_order_quantity_norm(perception, mu, sigma )

        q_values.append(q)
    return q_values

# 计算对应的订单量 q
order_quantities = calc_order_quantity(delta_range)

t = 17
order_quantities1 = calc_order_quantity(delta_range)

t = 19
order_quantities2 = calc_order_quantity(delta_range)

print(order_quantities)
# 绘制图表
plt.figure(figsize=(8, 6))
plt.plot(delta_range, order_quantities, label='调拨价格=15',color='blue',marker='v', markersize=8,)
plt.plot(delta_range, order_quantities1, label='调拨价格=17',color='yellow',marker='D', markersize=8,)
plt.plot(delta_range, order_quantities2, label='调拨价格=19',color='green',marker='>', markersize=8,)
#plt.title("Relationship between Allocation Price (t) and Order Quantity (q)")
plt.xlabel("竞争强度δ",fontsize=14)
plt.ylabel("最优订购量q",fontsize=14)
# 设置图例及其位置
plt.legend(loc='upper right', fontsize=14)
plt.grid(True)

plt.show()
