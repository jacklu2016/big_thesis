import numpy as np
import scipy.stats as stats
from scipy.integrate import dblquad

# 参数设定
mu, sigma = 200, 20         # 正态分布参数 N(200,20)
delta = 0.5                 # δ 值
q_i = 250                   # 假定 qᵢ = 250（你可根据需要调整）
q_minus_i = 220             # q₋ᵢ = 220

# 定义独立正态分布的概率密度函数
f_X = lambda x: stats.norm.pdf(x, mu, sigma)
f_Y = lambda y: stats.norm.pdf(y, mu, sigma)
# 联合密度函数
f_D = lambda x, y: f_X(x)*f_Y(y)

# 第一部分积分 I1:
# 对于每个 x ∈ [0, qᵢ]，y 从下限 L1(x) = (qᵢ - x)/δ + q₋ᵢ 到 +∞，
# 积分的被积函数为 (qᵢ - x)*f_D(x,y)
def integrand1(y, x, q_i, q_minus_i, delta):
    return (q_i - x) * f_D(x, y)

# 第二部分积分 I2:
# 对于每个 x ∈ [0, qᵢ]，y 从 q₋ᵢ 到上限 L1(x) = (qᵢ - x)/δ + q₋ᵢ，
# 被积函数为 δ*(y - q₋ᵢ)*f_D(x,y)
def integrand2(y, x, q_i, q_minus_i, delta):
    return delta * (y - q_minus_i) * f_D(x, y)

# 对第一部分积分：x 从 0 到 q_i，y 从 L1(x) 到 ∞
I1, err1 = dblquad(
    integrand1,
    0, q_i,                                   # x 积分区间
    lambda x: (q_i - x)/delta + q_minus_i,      # y 下限
    lambda x: np.inf,                          # y 上限
    args=(q_i, q_minus_i, delta)
)

# 对第二部分积分：x 从 0 到 q_i，y 从 q_minus_i 到 L1(x)
I2, err2 = dblquad(
    integrand2,
    0, q_i,
    lambda x: q_minus_i,
    lambda x: (q_i - x)/delta + q_minus_i,
    args=(q_i, q_minus_i, delta)
)

total_integral = I1 + I2

print("第一部分积分 I1 =", I1)
print("第二部分积分 I2 =", I2)
print("总积分 I =", total_integral)
