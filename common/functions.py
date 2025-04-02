import numpy as np


# 0일때 0.5 값을 리턴하는 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 자기 자신을 그대로 리턴하는 항등함수
def identity_function(x):
    return x

def softmax(a):
    c = np.max(a) # 오버플로 방지
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t)**2)
