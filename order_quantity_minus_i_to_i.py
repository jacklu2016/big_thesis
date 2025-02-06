import numpy as np
import scipy.stats as stats
from scipy.integrate import dblquad

# 参数设置
mu, sigma = 200, 20         # 正态分布 N(200,20) 的参数
delta = 0.5                 # δ 值
q_i = 250                   # 设定 q_i（可调整）
q_minus_i = 220             # q_{-i} 的值

# 定义独立正态分布的概率密度函数
f_X = lambda x: stats.norm.pdf(x, mu, sigma)
f_Y = lambda y: stats.norm.pdf(y, mu, sigma)

# 联合概率密度函数：由于 x,y 独立，有 f_D(x,y)=f_X(x)*f_Y(y)
f_D = lambda x, y: f_X(x) * f_Y(y)

# 定义积分被积函数
def integrand(y, x, q_i, q_minus_i, delta):
    return f_D(x, y)

# 外层积分变量 x 的区间为 [0, q_i]
# 对于每个 x, 内层积分变量 y 的区间为 [ (q_i - x)/δ + q_minus_i, +∞ )
I, err = dblquad(
    integrand,
    0, q_i,                              # x 积分区间
    lambda x: (q_i - x)/delta + q_minus_i, # y 下限
    lambda x: np.inf,                    # y 上限为无穷大
    args=(q_i, q_minus_i, delta)
)

print("积分结果 I =", I)
print("积分估计误差 =", err)
