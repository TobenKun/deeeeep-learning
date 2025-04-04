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

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x): # 신호가 순방향으로 전달되는 순전파
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

# a = np.array([0.3, 2.9, 4.0])
# y = softmax(a)
# print(y)
# print(np.sum(y))
