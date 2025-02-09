
import numpy as np
import scipy.stats as stats
from scipy.integrate import dblquad

# 参数
mu, sigma = 200, 20  # 正态分布 N(200, 20) 参数
delta = 0.2  # δ 值
q_i = 200  # 设定 q_i
q_minus_i = 220  # 设定 q_{-i}

# 定义独立正态分布概率密度函数
f_X = lambda x: stats.norm.pdf(x, mu, sigma)
f_Y = lambda y: stats.norm.pdf(y, mu, sigma)

# 第一项积分 I1
def integrand_1(y, x, q_i, q_minus_i, delta):
    return delta * (x - q_i) * f_X(x) * f_Y(y)

I1, _ = dblquad(
    integrand_1,
    q_i, np.inf,
    lambda x: 0, lambda x: q_minus_i - delta * (x - q_i),
    args=(q_i, q_minus_i, delta)
)

# 第二项积分 I2
def integrand_2(y, x, q_i, q_minus_i, delta):
    return (q_minus_i - y) * f_X(x) * f_Y(y)

I2, _ = dblquad(
    integrand_2,
    q_i, np.inf,
    lambda x: q_minus_i - delta * (x - q_i), lambda x: np.inf,
    args=(q_i, q_minus_i, delta)
)

# 计算最终积分
total_integral = I1 + I2
print("最终积分结果:", total_integral)
