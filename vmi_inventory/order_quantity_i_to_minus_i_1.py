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
sigma = np.sqrt(variance)

# 定义 x 和 y 的边缘密度函数，均服从 N(200,20)
def f_X(x):
    return norm.pdf(x, loc=mu, scale=sigma)

def f_Y(y):
    return norm.pdf(y, loc=mu, scale=sigma)

# 联合概率密度函数
def f_D(x, y):
    return f_X(x) * f_Y(y)

# 对于每个 x，y 的积分区间为：
# 下限：q_{-i} = 200
# 上限： (q_i - x)/delta + q_{-i} = (200 - x)/0.2 + 200
def y_lower(x):
    return q_neg_i

def y_upper(x):
    return (q_i - x) / delta + q_neg_i

# 被积函数，注意 dblquad 要求内层变量为第一个参数，外层变量为第二个参数
def integrand(y, x):
    return f_D(x, y)

# 使用 dblquad 进行双重积分
result, error = integrate.dblquad(
    integrand,     # 被积函数
    0, q_i,       # 外层积分变量 x 的积分区间 [0, q_i]
    y_lower,      # 内层积分变量 y 的下限（函数，参数为 x）
    y_upper       # 内层积分变量 y 的上限（函数，参数为 x）
)

print("积分结果: {:.6e}".format(result))
print("误差估计: {:.6e}".format(error))
