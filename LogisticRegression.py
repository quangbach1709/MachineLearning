# Hỗ trợ Python 2 và 3
from __future__ import division, print_function, unicode_literals
# Đảm bảo tương thích giữa Python 2 và 3 cho phép chia, in và chuỗi Unicode

import numpy as np 
# Nhập thư viện NumPy để xử lý mảng và tính toán số học

import matplotlib.pyplot as plt
# Nhập thư viện Matplotlib để vẽ đồ thị

np.random.seed(2)
# Đặt seed cho bộ sinh số ngẫu nhiên để đảm bảo kết quả có thể tái tạo

# Dữ liệu và nhãn (studying hours và pass/fail)
X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])
# Tạo mảng NumPy chứa dữ liệu đầu vào (số giờ học)

y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
# Tạo mảng NumPy chứa nhãn tương ứng (0: fail, 1: pass)

# Thêm một hàng gồm các số 1 vào X (cho hệ số w0)
X = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)
# Thêm một hàng các số 1 vào đầu X để tính hệ số tự do w0

# Hàm sigmoid
def sigmoid(s):
    return 1 / (1 + np.exp(-s))
# Định nghĩa hàm sigmoid để ánh xạ giá trị thực về khoảng (0, 1)

# Thuật toán Logistic Regression
def logistic_sigmoid_regression(X, y, w_init, eta, tol=1e-4, max_count=10000):
    w = [w_init]  # Khởi tạo danh sách lưu trữ trọng số qua từng lần lặp
    N = X.shape[1]  # Số lượng mẫu dữ liệu
    d = X.shape[0]  # Số đặc trưng (bao gồm cả hệ số b)
    count = 0
    check_w_after = 20  # Số lần lặp kiểm tra điều kiện hội tụ
    
    while count < max_count:
        mix_id = np.random.permutation(N)  # Trộn ngẫu nhiên dữ liệu
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)  # Đặc trưng của mẫu i (cột)
            yi = y[i]  # Nhãn thực tế của mẫu i
            zi = sigmoid(np.dot(w[-1].T, xi))  # Tính xác suất dự đoán
            w_new = w[-1] + eta * (yi - zi) * xi  # Cập nhật trọng số
            count += 1
            if count % check_w_after == 0:  # Kiểm tra điều kiện hội tụ
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w
            w.append(w_new)
    return w
# Định nghĩa hàm huấn luyện mô hình hồi quy logistic

# Tham số ban đầu
eta = .05  # Tốc độ học
d = X.shape[0]  # Số chiều của X (bao gồm cả hệ số b)
w_init = np.random.randn(d, 1)  # Khởi tạo trọng số ngẫu nhiên

# Huấn luyện Logistic Regression
w = logistic_sigmoid_regression(X, y, w_init, eta)
print(w[-1])
# Huấn luyện mô hình và in ra trọng số cuối cùng

# Dự đoán trên tập dữ liệu huấn luyện
print(sigmoid(np.dot(w[-1].T, X)))
# In ra xác suất dự đoán cho tất cả các mẫu trong tập huấn luyện

# Tách dữ liệu thành 2 nhóm dựa trên nhãn
X0 = X[1, np.where(y == 0)][0]
y0 = y[np.where(y == 0)]
X1 = X[1, np.where(y == 1)][0]
y1 = y[np.where(y == 1)]
# Tách dữ liệu thành hai nhóm: fail (0) và pass (1)

# Vẽ dữ liệu và đường sigmoid
plt.plot(X0, y0, 'ro', markersize=8)
plt.plot(X1, y1, 'bs', markersize=8)
# Vẽ các điểm dữ liệu: đỏ cho fail, xanh cho pass

# Tạo đường cong sigmoid
xx = np.linspace(0, 6, 1000)
w0 = w[-1][0][0]
w1 = w[-1][1][0]
threshold = -w0 / w1
yy = sigmoid(w0 + w1 * xx)
# Tạo dữ liệu để vẽ đường cong sigmoid

# Hiển thị đường sigmoid và điểm phân loại
plt.axis([-2, 8, -1, 2])
plt.plot(xx, yy, 'g-', linewidth=2)
plt.plot(threshold, 0.5, 'y^', markersize=8)
plt.xlabel('studying hours')
plt.ylabel('predicted probability of pass')
plt.show()
# Vẽ đồ thị cuối cùng với đường cong sigmoid và điểm phân loại
