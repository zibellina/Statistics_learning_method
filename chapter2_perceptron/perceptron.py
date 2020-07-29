import numpy as np


# 采用原始形式更新
def perceptron_Primal(X, Y, eta):
    """
    采用原始形式更新
    w <- w + eta * y * x
    b <- b + eta * y

    Args:
        X: 样本
        Y: 标签
        eta: 学习率

    return:
        W: 权重(权值向量)
        b: 偏置
    """
    n = len(X)
    p = len(X[0])
    W = np.zeros((p))
    b = 0

    while True:
        Flag = True
        for ind, item in enumerate(X):
            if Y[ind] * (np.dot(W, item) + b) > 0:
                continue
            Flag = False
            while Y[ind] * (np.dot(W, item) + b) <= 0:
                # print(ind, W, b)
                W += eta * Y[ind] * item
                b += eta * Y[ind]
        if Flag:
            break

    return W, b


def perceptron_dual(X, Y, eta):
    """
    采用对偶形式更新
    alpha = n*eta
    w = sum(alpha*y*x)
    b = sum(alpha*y)

    alpha <- alpha+eta
    b <- b + eta * y

    Args:
        X: 样本
        Y: 标签
        eta: 学习率

    return:
        W: 权重(权值向量)
        b: 偏置
    """
    n = len(X)
    p = len(X[0])
    alpha = np.zeros((n))
    b = 0
    gram = np.dot(X, X.T)

    while True:
        Flag = True
        for ind, item in enumerate(X):
            if Y[ind] * (np.dot(Y * gram[:, ind], alpha) + b) > 0:
                continue
            Flag = False
            while Y[ind] * (np.dot(Y * gram[:, ind], alpha) + b) <= 0:
                # print(ind, alpha, b)
                alpha[ind] += eta
                b += eta * Y[ind]
        if Flag:
            break

    # print(alpha, b)

    W = np.dot(alpha * Y, X)
    return W, b


if __name__ == "__main__":
    X = [[3, 3], [4, 3], [1, 1]]
    Y = [1, 1, -1]

    X = np.array(X)
    Y = np.array(Y)
    eta = 1

    w, b = perceptron_Primal(X, Y, eta)
    print(w, b)

    w, b = perceptron_dual(X, Y, eta)
    print(w, b)
