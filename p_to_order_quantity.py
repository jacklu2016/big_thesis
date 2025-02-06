from scipy.stats import norm

# 给定概率 p
p = 0.95

# 均值 mu 和标准差 sigma
mu = 200
sigma = 20

# 计算分位数 q
q = norm.ppf(p, loc=mu, scale=sigma)
print(f"对于概率 {p}，对应的分位数 q 为 {q}")
