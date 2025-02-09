import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm

# 参数设置
q_i = 200
q_neg_i = 200
delta = 0.2

# 正态分布参数
mu = 200
variance = 20
sigma = np.sqrt(variance)  # 标准差

# 定义 x 和 y 的边缘密度函数
def f_X(x):
    return norm.pdf(x, loc=mu, scale=sigma)

def f_Y(y):
    return norm.pdf(y, loc=mu, scale=sigma)

# 联合概率密度函数：x 与 y 独立
def f_D(x, y):
    return f_X(x) * f_Y(y)

# 定义 y 的积分下限: (q_i - x)/δ + q_{-i}
def lower_bound_y(x, q_i, delta, q_neg_i):
    return (q_i - x) / delta + q_neg_i

# y 的积分上限为正无穷
def upper_bound_y(x, q_i, delta, q_neg_i):
    return np.inf

# 被积函数，注意 dblquad 要求内层变量为第一个参数，外层变量为第二个参数
def integrand(y, x, q_i, delta, q_neg_i):
    return f_D(x, y)

# 使用 dblquad 进行双重积分，x 从 0 到 q_i，y 从 lower_bound_y(x) 到 +∞
result, error = integrate.dblquad(
    integrand,         # 被积函数：函数参数顺序为 (y, x)
    0, q_i,           # 外层积分变量 x 的积分范围 [0, q_i]
    lambda x: lower_bound_y(x, q_i, delta, q_neg_i),  # 内层积分变量 y 的下限
    lambda x: upper_bound_y(x, q_i, delta, q_neg_i),    # 内层积分变量 y 的上限
    args=(q_i, delta, q_neg_i)  # 额外参数传递给 integrand
)

print("积分结果: {:.6e}, 误差估计: {:.6e}".format(result, error))
