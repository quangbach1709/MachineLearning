import numpy as np  # Nhập thư viện numpy, một thư viện cho tính toán số học với mảng n chiều

def predict(w, X):
    ''' Dự đoán nhãn cho mỗi hàng của X, dựa trên w
    X: một mảng numpy 2 chiều có kích thước (N, d), mỗi hàng là một điểm dữ liệu
    w: một mảng numpy 1 chiều có kích thước (d)
    '''
    return np.sign(X.dot(w))  # Tính toán dự đoán bằng cách nhân ma trận X với trọng số w và trả về dấu của kết quả

def perceptron(X, y, w_init):
    ''' Thực hiện thuật toán học perceptron
    X: một mảng numpy 2 chiều có kích thước (N, d), mỗi hàng là một điểm dữ liệu
    y: một mảng numpy 1 chiều có kích thước (N), nhãn của mỗi hàng của X. y[i] = 1/-1
    w_init: một mảng numpy 1 chiều có kích thước (d)
    '''
    w = w_init  # Khởi tạo trọng số w với giá trị ban đầu
    while True:  # Vòng lặp vô hạn cho đến khi không còn điểm dữ liệu nào được phân loại sai
        pred = predict(w, X)  # Dự đoán nhãn cho dữ liệu X bằng trọng số w
        # Tìm các chỉ số của các điểm được phân loại sai
        mis_idxs = np.where(np.equal(pred, y) == False)[0]  # Lấy chỉ số của các điểm mà dự đoán không khớp với nhãn thực tế
        num_mis = mis_idxs.shape[0]  # Đếm số lượng điểm được phân loại sai
        if num_mis == 0:  # Nếu không còn điểm dữ liệu nào được phân loại sai
            return w  # Trả về trọng số w

        # Chọn ngẫu nhiên một điểm dữ liệu được phân loại sai
        random_id = np.random.choice(mis_idxs, 1)[0]  # Chọn một chỉ số ngẫu nhiên từ các chỉ số của điểm phân loại sai
        # Cập nhật trọng số w
        w = w + y[random_id] * X[random_id]  # Cập nhật trọng số bằng cách cộng với sản phẩm của nhãn và điểm dữ liệu

# Tạo dữ liệu mẫu
means = [[-1, 0], [1, 0]]  # Trung bình của hai lớp dữ liệu
cov = [[.3, .2], [.2, .3]]  # Ma trận hiệp phương sai
N = 10  # Số lượng điểm dữ liệu cho mỗi lớp
X0 = np.random.multivariate_normal(means[0], cov, N)  # Tạo N điểm dữ liệu cho lớp 0
X1 = np.random.multivariate_normal(means[1], cov, N)  # Tạo N điểm dữ liệu cho lớp 1
X = np.concatenate((X0, X1), axis=0)  # Kết hợp hai lớp dữ liệu thành một mảng
y = np.concatenate((np.ones(N), -1 * np.ones(N)))  # Tạo nhãn cho dữ liệu, lớp 0 là 1 và lớp 1 là -1

Xbar = np.concatenate((np.ones((2 * N, 1)), X), axis=1)  # Thêm cột 1 vào đầu mảng X để tính toán hệ số chặn
w_init = np.random.randn(Xbar.shape[1])  # Khởi tạo trọng số ngẫu nhiên với kích thước tương ứng

w = perceptron(Xbar, y, w_init)  # Chạy thuật toán perceptron để tìm trọng số

# Dự đoán nhãn cho dữ liệu huấn luyện
y_pred = predict(w, Xbar)  # Dự đoán nhãn cho dữ liệu Xbar

# In ra dự đoán
print("Predictions:")  # In tiêu đề cho phần dự đoán
for i, pred in enumerate(y_pred):  # Duyệt qua từng dự đoán
    print(f"Data point {i + 1}: {'Positive' if pred > 0 else 'Negative'}")  # In ra dự đoán cho từng điểm dữ liệu

# Tính và in ra độ chính xác
accuracy = np.mean(y_pred == y)  # Tính độ chính xác bằng cách so sánh dự đoán với nhãn thực tế
print(f"\nAccuracy: {accuracy * 100:.2f}%")  # In ra độ chính xác dưới dạng phần trăm

import matplotlib.pyplot as plt  # Nhập thư viện matplotlib để vẽ đồ thị

def visualize_perceptron(X, y, w, iterations):
    plt.figure(figsize=(10, 8))  # Tạo một hình với kích thước 10x8
    
    # Vẽ dữ liệu
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='b', label='Positive')  # Vẽ các điểm dữ liệu dương
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c='r', label='Negative')  # Vẽ các điểm dữ liệu âm
    
    # Vẽ đường phân chia
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # Xác định giới hạn x
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # Xác định giới hạn y
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))  # Tạo lưới cho đồ thị
    Z = predict(w, np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()])  # Dự đoán nhãn cho lưới
    Z = Z.reshape(xx.shape)  # Định hình lại dự đoán để phù hợp với lưới
    plt.contour(xx, yy, Z, colors='g', levels=[0], alpha=0.5, linestyles=['--'])  # Vẽ đường phân chia

    plt.xlabel('X1')  # Nhãn cho trục x
    plt.ylabel('X2')  # Nhãn cho trục y
    plt.title(f'Perceptron Decision Boundary (Iterations: {iterations})')  # Tiêu đề cho đồ thị
    plt.legend()  # Hiển thị chú thích
    plt.show()  # Hiển thị đồ thị

# Sửa đổi hàm perceptron để trả về cả w và số lần lặp
def perceptron(X, y, w_init):
    w = w_init  # Khởi tạo trọng số
    iterations = 0  # Khởi tạo số lần lặp
    while True:  # Vòng lặp vô hạn cho đến khi không còn điểm dữ liệu nào được phân loại sai
        iterations += 1  # Tăng số lần lặp
        pred = predict(w, X)  # Dự đoán nhãn cho dữ liệu X
        mis_idxs = np.where(np.equal(pred, y) == False)[0]  # Tìm các chỉ số của các điểm được phân loại sai
        num_mis = mis_idxs.shape[0]  # Đếm số lượng điểm được phân loại sai
        if num_mis == 0:  # Nếu không còn điểm dữ liệu nào được phân loại sai
            return w, iterations  # Trả về trọng số và số lần lặp
        random_id = np.random.choice(mis_idxs, 1)[0]  # Chọn một chỉ số ngẫu nhiên từ các chỉ số của điểm phân loại sai
        w = w + y[random_id] * X[random_id]  # Cập nhật trọng số

# Chạy thuật toán perceptron đã sửa đổi
w, iterations = perceptron(Xbar, y, w_init)  # Chạy thuật toán và nhận trọng số cùng số lần lặp

# Vẽ kết quả
visualize_perceptron(X, y, w, iterations)  # Vẽ đồ thị với dữ liệu và đường phân chia

print(f"\nNumber of iterations: {iterations}")  # In ra số lần lặp
