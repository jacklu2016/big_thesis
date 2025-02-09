from scipy.stats import norm

# 给定概率 p
p1 = 0.9

# 均值 mu 和标准差 sigma
mu1 = 200
sigma1 = 20

# 计算分位数 q
q = norm.ppf(p1, loc=mu1, scale=sigma1)
print(f"对于概率 {p1}，对应的分位数 q 为 {q}")

def p_to_order_quantity_norm(p,mu,sigma):
    return norm.ppf(p, loc=mu, scale=sigma)

