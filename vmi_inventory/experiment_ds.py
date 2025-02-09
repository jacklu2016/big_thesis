import numpy as np
from scipy.integrate import dblquad
from scipy.stats import norm

# 定义参数
mu = 200
sigma = 20
qi = 200
delta = 0.5

# 定义联合概率密度函数 f_D(x, y)
def f_D(x, y):
    return norm.pdf(x, mu, sigma) * norm.pdf(y, mu, sigma)

# 定义第一个积分的被积函数
def integrand1(y, x):
    return 0.5 * (x - qi) * f_D(x, y)

# 定义第二个积分的被积函数
def integrand2(y, x):
    return (mu - y) * f_D(x, y)

# 计算第一个积分
def lower_limit1(x):
    return 0

def upper_limit1(x):
    return mu - delta * (x - qi)

result1, _ = dblquad(integrand1, qi, np.inf, lower_limit1, upper_limit1)

# 计算第二个积分
def lower_limit2(x):
    return mu - delta * (x - qi)

def upper_limit2(x):
    return np.inf

result2, _ = dblquad(integrand2, qi, np.inf, lower_limit2, upper_limit2)

# 总结果
total_result = result1 + result2
print("积分结果:", total_result)