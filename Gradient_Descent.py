# Để hỗ trợ cả Python 2 và 3
from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt

# Khởi tạo dữ liệu ngẫu nhiên
np.random.seed(2)
X = np.random.rand(1000, 1)
y = 4 + 3 * X + 0.2 * np.random.randn(1000, 1)  # Thêm nhiễu vào dữ liệu

# Hàm tính gradient của hàm mất mát
def grad(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

# Hàm tính giá trị của hàm mất mát (cost function)
def cost(w):
    N = Xbar.shape[0]
    return 0.5/N * np.linalg.norm(y - Xbar.dot(w), 2)**2

# Xây dựng ma trận Xbar từ X (thêm cột 1 vào để tính hệ số w_0)
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

# Giải bài toán bằng công thức giải tích
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w_lr = np.dot(np.linalg.pinv(A), b)
print('Solution found by formula: w = ', w_lr.T)

# Hiển thị kết quả
w = w_lr
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(0, 1, 2, endpoint=True)
y0 = w_0 + w_1 * x0

# Vẽ dữ liệu và đường hồi quy
plt.plot(X, y, 'b.')  # dữ liệu
plt.plot(x0, y0, 'y', linewidth=2)  # đường hồi quy
plt.axis([0, 1, 0, 10])
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Analytical Solution')
plt.show()

# Hàm tính gradient theo phương pháp số
def numerical_grad(w, cost):
    eps = 1e-4
    g = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy()
        w_n = w.copy()
        w_p[i] += eps
        w_n[i] -= eps
        g[i] = (cost(w_p) - cost(w_n)) / (2 * eps)
    return g

# Hàm kiểm tra tính đúng của gradient
def check_grad(w, cost, grad):
    w = np.random.rand(w.shape[0], w.shape[1])
    grad1 = grad(w)
    grad2 = numerical_grad(w, cost)
    return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False

# Kiểm tra gradient
print('Checking gradient...', check_grad(np.random.rand(2, 1), cost, grad))

# Hàm Gradient Descent
def myGD(w_init, grad, eta):
    w = [w_init]
    for it in range(100):
        w_new = w[-1] - eta * grad(w[-1])
        if np.linalg.norm(grad(w_new)) / len(w_new) < 1e-3:
            break
        w.append(w_new)
    return (w, it)

# Khởi tạo w và chạy thuật toán Gradient Descent
w_init = np.array([[2], [1]])
(w1, it1) = myGD(w_init, grad, eta=0.1)
print('Solution found by GD: w = ', w1[-1].T, ',\nafter %d iterations.' % (it1 + 1))

# Hiển thị kết quả từ Gradient Descent
w_0_gd = w1[-1][0][0]
w_1_gd = w1[-1][1][0]
y0_gd = w_0_gd + w_1_gd * x0

# Vẽ dữ liệu và đường hồi quy tìm được bằng Gradient Descent
plt.plot(X, y, 'b.')  # dữ liệu
plt.plot(x0, y0_gd, 'r', linewidth=2, label='GD solution')  # đường hồi quy từ GD
plt.plot(x0, y0, 'y', linewidth=2, label='Analytical solution')  # đường hồi quy từ giải tích
plt.axis([0, 1, 0, 10])
plt.xlabel('X')
plt.ylabel('y')
plt.legend(loc='best')
plt.title('Linear Regression with Gradient Descent')
plt.show()
