import numpy as np
import matplotlib.pyplot as plt

# Tạo dữ liệu giả lập (2 lớp)
np.random.seed(1)
X1 = np.random.randn(50, 2) - [2, 2]
X2 = np.random.randn(50, 2) + [2, 2]
X = np.vstack((X1, X2))
y = np.hstack((-1 * np.ones(50), np.ones(50)))  # Nhãn -1 và 1

# Hàm tính dự đoán của mô hình SVM tuyến tính
def predict(X, w, b):
    return np.sign(np.dot(X, w) + b)

# Hàm huấn luyện SVM sử dụng Gradient Descent
def svm_train(X, y, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)  # Vector trọng số
    b = 0  # Bias

    # Gradient Descent để tối ưu hóa w và b
    for _ in range(n_iters):
        for i, x_i in enumerate(X):
            # Kiểm tra điều kiện ràng buộc của SVM
            if y[i] * (np.dot(x_i, w) + b) >= 1:
                # Không vi phạm điều kiện ràng buộc
                w -= learning_rate * (2 * lambda_param * w)
            else:
                # Vi phạm điều kiện ràng buộc, cập nhật w và b
                w -= learning_rate * (2 * lambda_param * w - np.dot(x_i, y[i]))
                b -= learning_rate * y[i]
    
    return w, b

# Huấn luyện mô hình SVM
w, b = svm_train(X, y)

# Dự đoán trên dữ liệu huấn luyện
y_pred = predict(X, w, b)

# Tính độ chính xác
accuracy = np.mean(y_pred == y)
print(f"Độ chính xác của SVM: {accuracy * 100:.2f}%")

# Vẽ đồ thị kết quả
def plot_svm_decision_boundary(X, y, w, b):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50)

    # Tạo đường thẳng biểu diễn siêu phẳng (w.x + b = 0)
    ax = plt.gca()
    x_vals = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
    y_vals = -(w[0] * x_vals + b) / w[1]

    plt.plot(x_vals, y_vals, 'k-')
    plt.title("Siêu phẳng phân chia của SVM")
    plt.xlabel("Đặc trưng 1")
    plt.ylabel("Đặc trưng 2")
    plt.show()

# Hiển thị đồ thị phân chia của SVM
plot_svm_decision_boundary(X, y, w, b)
