# Hỗ trợ Python 2 và 3
from __future__ import print_function
# Đảm bảo tương thích giữa Python 2 và 3 cho hàm print

import numpy as np 
# Nhập thư viện NumPy để xử lý mảng và tính toán số học

import matplotlib.pyplot as plt
# Nhập thư viện Matplotlib để vẽ đồ thị

from scipy.spatial.distance import cdist
# Nhập hàm cdist từ SciPy để tính khoảng cách giữa các điểm

from cvxopt import matrix, solvers
# Nhập các công cụ từ CVXOPT để giải bài toán tối ưu hóa lồi

np.random.seed(22)
# Đặt seed cho bộ sinh số ngẫu nhiên để đảm bảo kết quả có thể tái tạo

# Tạo dữ liệu giả lập
means = [[2, 2], [4, 2]]
# Định nghĩa trung bình cho hai lớp

cov = [[.3, .2], [.2, .3]]
# Định nghĩa ma trận hiệp phương sai cho cả hai lớp

N = 10
# Số lượng điểm dữ liệu cho mỗi lớp

X0 = np.random.multivariate_normal(means[0], cov, N) # class 1
# Tạo N điểm dữ liệu cho lớp 1 từ phân phối chuẩn đa biến

X1 = np.random.multivariate_normal(means[1], cov, N) # class -1 
# Tạo N điểm dữ liệu cho lớp -1 từ phân phối chuẩn đa biến

X = np.concatenate((X0.T, X1.T), axis=1) # Tập dữ liệu kết hợp
# Kết hợp dữ liệu từ cả hai lớp thành một ma trận X

y = np.concatenate((np.ones((1, N)), -1 * np.ones((1, N))), axis=1) # Nhãn
# Tạo nhãn tương ứng: 1 cho lớp 1 và -1 cho lớp -1

# Xây dựng ma trận V
V = np.concatenate((X0.T, -X1.T), axis=1)
# Tạo ma trận V bằng cách ghép X0 và -X1

K = matrix(V.T.dot(V)) # Ma trận K = V^T V
# Tính ma trận kernel K = V^T * V và chuyển đổi thành đối tượng matrix của CVXOPT

# Khởi tạo các tham số cho bài toán Quadratic Programming
p = matrix(-np.ones((2 * N, 1)))
# Tạo vector p với tất cả các phần tử là -1

G = matrix(-np.eye(2 * N)) # Điều kiện lambda >= 0
# Tạo ma trận G là ma trận đơn vị âm để đảm bảo lambda >= 0

h = matrix(np.zeros((2 * N, 1)))
# Tạo vector h với tất cả các phần tử là 0

A = matrix(y) # Ràng buộc y^T lambda = 0
# Sử dụng y làm ma trận A để đảm bảo ràng buộc y^T * lambda = 0

b = matrix(np.zeros((1, 1)))
# Tạo ma trận b là một ma trận 1x1 với giá trị 0

# Giải bài toán Quadratic Programming
solvers.options['show_progress'] = False
# Tắt hiển thị tiến trình giải

sol = solvers.qp(K, p, G, h, A, b)
# Giải bài toán Quadratic Programming

# Lấy các giá trị của lambda
l = np.array(sol['x'])
# Chuyển đổi kết quả từ đối tượng matrix của CVXOPT sang mảng NumPy

print('lambda = ')
print(l.T)
# In ra giá trị của lambda

# Chọn các support vectors
epsilon = 1e-6 # Số nhỏ để xác định support vectors
# Định nghĩa một giá trị epsilon nhỏ để so sánh

S = np.where(l > epsilon)[0] # Chỉ số của các support vectors
# Tìm chỉ số của các support vectors (lambda > epsilon)

# Lấy các support vectors và tính toán w, b
VS = V[:, S]
# Lấy các cột của V tương ứng với các support vectors

XS = X[:, S]
# Lấy các cột của X tương ứng với các support vectors

yS = y[:, S]
# Lấy các nhãn tương ứng với các support vectors

lS = l[S]
# Lấy các giá trị lambda tương ứng với các support vectors

# Tính vector w và bias b
w = VS.dot(lS)
# Tính vector w bằng cách nhân VS với lS

b = np.mean(yS.T - w.T.dot(XS))
# Tính bias b bằng cách lấy trung bình của (y - w^T * x) cho các support vectors

print('w = ', w.T)
print('b = ', b)
# In ra giá trị của w và b

# Trực quan hóa dữ liệu và siêu phẳng
def plot_svm(X0, X1, w, b):
    plt.plot(X0[:, 0], X0[:, 1], 'ro', markersize=8, label='Class 1')
    # Vẽ các điểm của lớp 1 bằng màu đỏ

    plt.plot(X1[:, 0], X1[:, 1], 'bs', markersize=8, label='Class -1')
    # Vẽ các điểm của lớp -1 bằng màu xanh

    # Vẽ siêu phẳng
    xx = np.linspace(0, 6, 100)
    # Tạo 100 điểm trên trục x từ 0 đến 6

    yy = -(w[0] * xx + b) / w[1]
    # Tính giá trị y tương ứng cho mỗi x để vẽ đường decision boundary

    plt.plot(xx, yy, 'g-', linewidth=2, label='Decision boundary')
    # Vẽ đường decision boundary

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.axis([0, 6, 0, 6])
    # Thiết lập các nhãn trục, chú thích và giới hạn trục

    plt.show()
    # Hiển thị đồ thị

# Hiển thị đồ thị
plot_svm(X0, X1, w, b)
# Gọi hàm plot_svm để vẽ đồ thị kết quả
