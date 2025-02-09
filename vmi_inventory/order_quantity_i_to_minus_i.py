import scipy.stats as stats
from scipy.integrate import dblquad
import config

# 参数设定
mu, sigma = config.mu, config.sigma         # 正态分布 N(200,20) 的参数
delta = config.delta                 # δ 值
q_i = config.q_i                   # 设定 q_i 的值（可根据需要修改）
q_minus_i = config.q_minus_i             # q_{-i} 的值

# 定义独立正态分布的概率密度函数
f_X = lambda x: stats.norm.pdf(x, mu, sigma)
f_Y = lambda y: stats.norm.pdf(y, mu, sigma)
# 联合概率密度函数
f_D = lambda x, y: f_X(x) * f_Y(y)

# 定义积分区域：
# 对于 x，从 0 到 q_i；
# 对于每个 x，y 从下限 q_{-i} 到上限: (q_i - x)/delta + q_{-i}
def integrand(y, x, q_i, q_minus_i, delta):
    return f_D(x, y)

# 使用 dblquad 计算双重积分
# I, err = dblquad(
#     integrand,
#     0, q_i,                              # x 积分区间：[0, q_i]
#     lambda x: q_minus_i,                 # y 下限为 q_{-i}
#     lambda x: (q_i - x)/delta + q_minus_i,  # y 上限为 (q_i - x)/δ + q_{-i}
#     args=(q_i, q_minus_i, delta)
# )

def i_to_minus_i_dblquad():
    return dblquad(
    integrand,
    0, q_i,                              # x 积分区间：[0, q_i]
    lambda x: q_minus_i,                 # y 下限为 q_{-i}
    lambda x: (q_i - x)/delta + q_minus_i,  # y 上限为 (q_i - x)/δ + q_{-i}
    args=(q_i, q_minus_i, delta)
)

I,err = i_to_minus_i_dblquad()
print("积分结果 I =", I)
print("积分估计误差 =", err)
