# -*-coding:utf-8-*-
import numpy as np

# 对应状态集合Q
states = ('Healthy', 'Fever')
# 对应观测集合V
observations = ('normal', 'cold', 'dizzy')
# 初始状态概率向量π
start_probability = {'Healthy': 0.6, 'Fever': 0.4}
# 状态转移矩阵A
transition_probability = {
    'Healthy': {'Healthy': 0.7, 'Fever': 0.3},
    'Fever': {'Healthy': 0.4, 'Fever': 0.6},
}
# 观测概率矩阵B
emission_probability = {
    'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
    'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
}


# 随机生成观测序列和状态序列
def simulate(T):
    def draw_from(probs):
        return np.where(np.random.multinomial(1, probs) == 1)[0][0]

    observations = np.zeros(T, dtype=int)
    states = np.zeros(T, dtype=int)
    states[0] = draw_from(pi)
    observations[0] = draw_from(B[states[0], :])
    for t in range(1, T):
        states[t] = draw_from(A[states[t - 1], :])
        observations[t] = draw_from(B[states[t], :])
    return observations, states
