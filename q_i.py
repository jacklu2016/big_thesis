#∫_0^(q_i)▒∫_((q_i-x)/δ+q_(-i))^(+∞)▒〖f_D (x,y)〗 □(24&dy)□(24&dx)

import order_quantity_minus_i_to_i
import order_quantity_i_to_minus_i
import config
import p_to_order_quantity
#调拨价格
t_arr = [12,13,14,15,16,17]

def t_to_q_i(t):
    #(p_i-t)后面的积分
    q_i_t_calculus,err = order_quantity_minus_i_to_i.minus_i_to_i_dblquad()
    t_w_s_calculus,err1 = order_quantity_i_to_minus_i.i_to_minus_i_dblquad()
    F_q_i = ((config.p_i - config.w)+(config.p_i-t)*q_i_t_calculus - (t - config.w - config.s)*t_w_s_calculus)/config.p_i
    print(f"F_q_i:{F_q_i}")
    q_i = p_to_order_quantity.p_to_order_quantity(F_q_i,config.mu,config.sigma)
    print(f"q_i:{q_i}")
    return q_i

q_i_arr = [t_to_q_i(x) for x in t_arr]
print(q_i_arr)

