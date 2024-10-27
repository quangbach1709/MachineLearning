from __future__ import print_function
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from cvxopt import matrix, solvers

# Khởi tạo dữ liệu
np.random.seed(22)
means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)  # class 1
X1 = np.random.multivariate_normal(means[1], cov, N)  # class -1 
X = np.concatenate((X0.T, X1.T), axis=1)  # Tất cả dữ liệu
y = np.concatenate((np.ones((1, N)), -1 * np.ones((1, N))), axis=1)  # Nhãn

# Xây dựng ma trận V từ dữ liệu X0 và X1
V = np.concatenate((X0.T, -X1.T), axis=1)
K = matrix(V.T.dot(V))  # Ma trận K từ công thức tối ưu hóa

# Xây dựng các thành phần của bài toán tối ưu
p = matrix(-np.ones((2 * N, 1)))  # Vector toàn 1
G = matrix(-np.eye(2 * N))  # Ràng buộc lambda >= 0
h = matrix(np.zeros((2 * N, 1)))  # Vector toàn 0
A = matrix(y)  # Ràng buộc y^T * lambda = 0
b = matrix(np.zeros((1, 1)))  # b = 0

# Tắt hiển thị quá trình tối ưu hóa
solvers.options['show_progress'] = False

# Giải bài toán tối ưu bậc hai
sol = solvers.qp(K, p, G, h, A, b)
l = np.array(sol['x'])  # Giá trị lambda sau khi tối ưu

# Lấy các chỉ số của lambda lớn hơn epsilon
epsilon = 1e-6
S = np.where(l > epsilon)[0]

# Tính toán các giá trị VS, XS, yS, lS
VS = V[:, S]
XS = X[:, S]
yS = y[:, S]
lS = l[S]

# Tính w và b
w = VS.dot(lS)
b = np.mean(yS.T - w.T.dot(XS))

print('lambda = ')
print(l.T)
print('w = ', w.T)
print('b = ', b)

# Hiển thị các điểm dữ liệu và đường quyết định
def plot_data(X0, X1, w, b):
    plt.plot(X0[:, 0], X0[:, 1], 'ro', label='Class 1')
    plt.plot(X1[:, 0], X1[:, 1], 'bs', label='Class -1')

    # Vẽ đường quyết định
    x_vals = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    y_vals = -(w[0] * x_vals + b) / w[1]
    plt.plot(x_vals, y_vals, 'k-', label='Decision boundary')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.axis('equal')
    plt.show()

# Hiển thị kết quả
plot_data(X0, X1, w, b)
