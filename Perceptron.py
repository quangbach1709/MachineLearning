import numpy as np
def predict(w, X):
    ''' dự đoán nhãn cho mỗi hàng của X, dựa trên w
    X: một mảng numpy 2 chiều có kích thước (N, d), mỗi hàng là một điểm dữ liệu
    w: một mảng numpy 1 chiều có kích thước (d)
    '''
    return np.sign(X.dot(w))
def perceptron(X, y, w_init):
    ''' thực hiện thuật toán học perceptron
    X: một mảng numpy 2 chiều có kích thước (N, d), mỗi hàng là một điểm dữ liệu
    y: một mảng numpy 1 chiều có kích thước (N), nhãn của mỗi hàng của X. y[i] = 1/-1
    w_init: một mảng numpy 1 chiều có kích thước (d)
    '''
    w = w_init
    while True:
        pred = predict(w, X)
        # tìm các chỉ số của các điểm được phân loại sai
        mis_idxs = np.where(np.equal(pred, y) == False)[0]
        num_mis = mis_idxs.shape[0]
        if num_mis == 0:  # không còn điểm dữ liệu nào được phân loại sai
            return w
        # chọn ngẫu nhiên một điểm dữ liệu được phân loại sai
        random_id = np.random.choice(mis_idxs, 1)[0]
        # cập nhật trọng số w
        w = w + y[random_id] * X[random_id]

means = [[-1, 0], [1, 0]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X = np.concatenate((X0, X1), axis=0)
y = np.concatenate((np.ones(N), -1 * np.ones(N)))

Xbar = np.concatenate((np.ones((2 * N, 1)), X), axis=1)
w_init = np.random.randn(Xbar.shape[1])

w = perceptron(Xbar, y, w_init)

# dự đoán nhãn cho dữ liệu huấn luyện
y_pred = predict(w, Xbar)

# in ra dự đoán
print("Predictions:")
for i, pred in enumerate(y_pred):
    print(f"Data point {i + 1}: {'Positive' if pred > 0 else 'Negative'}")

# tính và in ra độ chính xác
accuracy = np.mean(y_pred == y)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

import matplotlib.pyplot as plt

def visualize_perceptron(X, y, w, iterations):
    plt.figure(figsize=(10, 8))
    
    # vẽ dữ liệu
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='b', label='Positive')
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='r', label='Negative')
    
    # vẽ đường phân chia
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = predict(w, np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='g', levels=[0], alpha=0.5, linestyles=['--'])
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(f'Perceptron Decision Boundary (Iterations: {iterations})')
    plt.legend()
    plt.show()

# sửa đổi hàm perceptron để trả về cả w và số lần lặp
def perceptron(X, y, w_init):
    w = w_init
    iterations = 0
    while True:
        iterations += 1
        pred = predict(w, X)
        mis_idxs = np.where(np.equal(pred, y) == False)[0]
        num_mis = mis_idxs.shape[0]
        if num_mis == 0:
            return w, iterations
        random_id = np.random.choice(mis_idxs, 1)[0]
        w = w + y[random_id] * X[random_id]

# chạy thuật toán perceptron đã sửa đổi
w, iterations = perceptron(Xbar, y, w_init)

# vẽ kết quả
visualize_perceptron(X, y, w, iterations)

print(f"\nNumber of iterations: {iterations}")



