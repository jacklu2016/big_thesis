import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 常量定义
p_i = 30  # 零售价格
w_init = 12  # 初始批发价格
c = 8  # 采购成本
h_r_h_v = 1  # 库存持有成本
s = 0.1  # 库存共享成本
#delta = 0.3  # 竞争强度
D = 0.1  # 需求量
q_i = 200  # 订购量
q_neg_i = 200  # 竞争者订购量
t = 15  # 调拨价格的变化范围
delta_range = np.linspace(0.2, 0.8, 10)
# 定义正态分布的概率密度函数和累积分布函数
mu, sigma = 200, 20
f_D = norm.pdf  # 概率密度函数
F_D = norm.cdf  # 累积分布函数


# 计算积分项，返回积分结果
def integral_term_1(q_i, q_neg_i):
    # 需要计算的积分项 f_D(x, (q_i - x)/δ + q_neg_i)
    def integrand(x):
        return f_D(x, mu, sigma)   # 计算 (q_i - x)/δ + q_neg_i

    integral, _ = quad(integrand, 0, q_i)  # 积分范围为 0 到 q_i
    return integral


# 批发价格计算公式
def calculate_wholesale_price(t, q_i, p_i, c, h_r_h_v, delta, s, D):
    # 计算需要的积分项
    integral_value = integral_term_1(q_i, q_neg_i)
    print(integral_value)

    numerator = (q_i * p_i + ((p_i - t) / delta)  * integral_value +
                 (c + h_r_h_v) * (1 + (1 / delta) * integral_value) - F_D(q_i, mu, sigma) +
                 (t - s) * D)

    denominator = 1 + q_i + delta

    w_star = numerator / denominator / 3 + 1  # 计算批发价格 w^*
    return w_star


# 分析调拨价格 t 对批发价格的影响
def analyze_wholesale_price_effect():
    wholesale_prices = []

    for delta in delta_range:
        w_star = calculate_wholesale_price(t, q_i, p_i, c, h_r_h_v, delta, s, D)
        wholesale_prices.append(w_star)

    return wholesale_prices


# 进行分析
wholesale_prices = analyze_wholesale_price_effect()
t = 17
wholesale_prices1 = analyze_wholesale_price_effect()
t = 19
wholesale_prices2 = analyze_wholesale_price_effect()

# 绘制调拨价格 t 对批发价格的影响图
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(delta_range, wholesale_prices, label='调拨价格=15' ,color='blue', marker='v', markersize=8)
plt.plot(delta_range, wholesale_prices1, label='调拨价格=17' ,color='yellow', marker='D', markersize=8)
plt.plot(delta_range, wholesale_prices2, label='调拨价格=19' ,color='green', marker='>', markersize=8)
plt.xlabel('竞争强度δ',fontsize=14)
plt.ylabel('批发价格w',fontsize=14)
#plt.title('调拨价格对批发价格的影响')
plt.legend(fontsize=14)
plt.grid(True)
plt.show()
