# Để hỗ trợ cả Python 2 và 3
from __future__ import division, print_function, unicode_literals  # Đảm bảo tính tương thích giữa Python 2 và 3
import numpy as np  # Nhập thư viện numpy để xử lý mảng và tính toán số học
import matplotlib.pyplot as plt  # Nhập thư viện matplotlib để vẽ đồ thị

# Khởi tạo dữ liệu ngẫu nhiên
np.random.seed(2)  # Đặt hạt giống cho bộ sinh số ngẫu nhiên để có thể tái tạo kết quả
X = np.random.rand(1000, 1)  # Tạo một mảng 1000x1 với các giá trị ngẫu nhiên từ 0 đến 1
y = 4 + 3 * X + 0.2 * np.random.randn(1000, 1)  # Tạo biến mục tiêu y với một hàm tuyến tính cộng thêm nhiễu ngẫu nhiên

# Hàm tính gradient của hàm mất mát
def grad(w):
    N = Xbar.shape[0]  # Lấy số lượng mẫu từ ma trận Xbar
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y)  # Tính gradient của hàm mất mát

# Hàm tính giá trị của hàm mất mát (cost function)
def cost(w):
    N = Xbar.shape[0]  # Lấy số lượng mẫu từ ma trận Xbar
    return 0.5/N * np.linalg.norm(y - Xbar.dot(w), 2)**2  # Tính giá trị hàm mất mát

# Xây dựng ma trận Xbar từ X (thêm cột 1 vào để tính hệ số w_0)
one = np.ones((X.shape[0], 1))  # Tạo một cột chứa toàn bộ giá trị 1 với số hàng bằng số hàng của X
Xbar = np.concatenate((one, X), axis=1)  # Nối cột 1 với ma trận X để tạo ma trận Xbar

# Giải bài toán bằng công thức giải tích
A = np.dot(Xbar.T, Xbar)  # Tính ma trận A là ma trận chuyển vị của Xbar nhân với Xbar
b = np.dot(Xbar.T, y)  # Tính ma trận b là ma trận chuyển vị của Xbar nhân với y
w_lr = np.dot(np.linalg.pinv(A), b)  # Tính trọng số w_lr bằng cách sử dụng ma trận nghịch đảo của A
print('Solution found by formula: w = ', w_lr.T)  # In ra trọng số tìm được bằng công thức

# Hiển thị kết quả
w = w_lr  # Gán trọng số tìm được cho w
w_0 = w[0][0]  # Lấy hệ số w_0
w_1 = w[1][0]  # Lấy hệ số w_1
x0 = np.linspace(0, 1, 2, endpoint=True)  # Tạo một mảng x0 với 2 giá trị từ 0 đến 1
y0 = w_0 + w_1 * x0  # Tính giá trị y0 dựa trên w_0 và w_1

# Vẽ dữ liệu và đường hồi quy
plt.plot(X, y, 'b.')  # Vẽ dữ liệu với màu xanh
plt.plot(x0, y0, 'y', linewidth=2)  # Vẽ đường hồi quy với màu vàng
plt.axis([0, 1, 0, 10])  # Đặt giới hạn cho trục x và y
plt.xlabel('X')  # Đặt nhãn cho trục x
plt.ylabel('y')  # Đặt nhãn cho trục y
plt.title('Linear Regression with Analytical Solution')  # Đặt tiêu đề cho đồ thị
plt.show()  # Hiển thị đồ thị

# Hàm tính gradient theo phương pháp số
def numerical_grad(w, cost):
    eps = 1e-4  # Đặt giá trị epsilon nhỏ để tính gradient
    g = np.zeros_like(w)  # Khởi tạo mảng g với kích thước giống w
    for i in range(len(w)):  # Duyệt qua từng phần tử của w
        w_p = w.copy()  # Sao chép w vào w_p
        w_n = w.copy()  # Sao chép w vào w_n
        w_p[i] += eps  # Tăng phần tử thứ i của w_p lên epsilon
        w_n[i] -= eps  # Giảm phần tử thứ i của w_n xuống epsilon
        g[i] = (cost(w_p) - cost(w_n)) / (2 * eps)  # Tính gradient tại phần tử thứ i
    return g  # Trả về gradient

# Hàm kiểm tra tính đúng của gradient
def check_grad(w, cost, grad):
    w = np.random.rand(w.shape[0], w.shape[1])  # Tạo một mảng w ngẫu nhiên
    grad1 = grad(w)  # Tính gradient bằng hàm grad
    grad2 = numerical_grad(w, cost)  # Tính gradient bằng phương pháp số
    return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False  # So sánh hai gradient

# Kiểm tra gradient
print('Checking gradient...', check_grad(np.random.rand(2, 1), cost, grad))  # In ra kết quả kiểm tra gradient

# Hàm Gradient Descent
def myGD(w_init, grad, eta):
    w = [w_init]  # Khởi tạo danh sách w với giá trị khởi tạo
    for it in range(100):  # Lặp tối đa 100 lần
        w_new = w[-1] - eta * grad(w[-1])  # Cập nhật trọng số mới
        if np.linalg.norm(grad(w_new)) / len(w_new) < 1e-3:  # Kiểm tra điều kiện dừng
            break  # Dừng nếu điều kiện thỏa mãn
        w.append(w_new)  # Thêm trọng số mới vào danh sách
    return (w, it)  # Trả về danh sách trọng số và số lần lặp

# Khởi tạo w và chạy thuật toán Gradient Descent
w_init = np.array([[2], [1]])  # Khởi tạo trọng số ban đầu
(w1, it1) = myGD(w_init, grad, eta=0.1)  # Chạy thuật toán Gradient Descent
print('Solution found by GD: w = ', w1[-1].T, ',\nafter %d iterations.' % (it1 + 1))  # In ra trọng số tìm được và số lần lặp

# Hiển thị kết quả từ Gradient Descent
w_0_gd = w1[-1][0][0]  # Lấy hệ số w_0 từ kết quả GD
w_1_gd = w1[-1][1][0]  # Lấy hệ số w_1 từ kết quả GD
y0_gd = w_0_gd + w_1_gd * x0  # Tính giá trị y0_gd dựa trên w_0_gd và w_1_gd

# Vẽ dữ liệu và đường hồi quy tìm được bằng Gradient Descent
plt.plot(X, y, 'b.')  # Vẽ dữ liệu với màu xanh
plt.plot(x0, y0_gd, 'r', linewidth=2, label='GD solution')  # Vẽ đường hồi quy từ GD với màu đỏ
plt.plot(x0, y0, 'y', linewidth=2, label='Analytical solution')  # Vẽ đường hồi quy từ giải tích với màu vàng
plt.axis([0, 1, 0, 10])  # Đặt giới hạn cho trục x và y
plt.xlabel('X')  # Đặt nhãn cho trục x
plt.ylabel('y')  # Đặt nhãn cho trục y
plt.legend(loc='best')  # Hiển thị chú thích cho đồ thị
plt.title('Linear Regression with Gradient Descent')  # Đặt tiêu đề cho đồ thị
plt.show()  # Hiển thị đồ thị
