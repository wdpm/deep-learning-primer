import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# layer 1
# 1 x 2
X = np.array([1.0, 0.5])
# 2 x 3
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print("X.shape: ", X.shape)
print("W1.shape: ", W1.shape)
print("B1.shape: ", B1.shape)

# 1 x 3
A1 = np.dot(X, W1) + B1
print("A1: ", A1)

# 1 x 3
Z1 = sigmoid(A1)
print("Z1: ", Z1)

# layer 2
# 3 x 2
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print("Z1.shape: ", Z1.shape)
print("W2.shape: ", W2.shape)
print("B2.shape: ", B2.shape)

# 1 x 2
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print("A2.shape: ", A2.shape)
print("Z2.shape: ", Z2.shape)


# layer 3
def identity_function(x):
    return x


# 2 x 2
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

# 1 x 2
A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)

print("W3.shape: ", W3.shape)
print("B3.shape: ", B3.shape)
print("A3.shape: ", A3.shape)
print("Y.shape: ", Y.shape)
print(Y)
