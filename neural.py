import numpy as np


# 0일때 0.5 값을 리턴하는 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 자기 자신을 그대로 리턴하는 항등함수
def identity_function(x):
    return x

X = np.array([1.0, 0.5]) # x1, x2 에 들어오는 입력값
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # x1 에서 a1, a2, a3 | x2 에서 a1, a2, a3
B1 = np.array([0.1, 0.2, 0.3]) # a1, a2, a3 에 더해질 편향값

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)

print(A1)
print(Z1)
print("-------------------")

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]) # 1층에서 2층으로 가는 가중치
B2 = np.array([0.1, 0.2]) # 2층 뉴런에 더해질 편향값

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

W3 = np.array([[0.1, 0.3], [0.2, 0.4]]) # 2층에서 출력층으로 가는 가중치
B3 = np.array([0.1, 0.2]) # 3층 뉴런에 더해질 편향값

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)

print(Y)

# 함수로 정리하면 아래가 됨

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y


network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
