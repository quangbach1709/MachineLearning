import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# Tạo dữ liệu mẫu
means = [[-1, 0], [1, 0]]  # Trung bình của hai lớp dữ liệu
cov = [[.3, .2], [.2, .3]]  # Ma trận hiệp phương sai
N = 10  # Số lượng điểm dữ liệu cho mỗi lớp
X0 = np.random.multivariate_normal(means[0], cov, N)  # Tạo N điểm dữ liệu cho lớp 0
X1 = np.random.multivariate_normal(means[1], cov, N)  # Tạo N điểm dữ liệu cho lớp 1
X = np.concatenate((X0, X1), axis=0)  # Kết hợp hai lớp dữ liệu thành một mảng
y = np.concatenate((np.ones(N), -1 * np.ones(N)))  # Tạo nhãn cho dữ liệu, lớp 0 là 1 và lớp 1 là -1

# Khởi tạo và huấn luyện mô hình Perceptron
model = Perceptron()
model.fit(X, y)

# Dự đoán nhãn cho dữ liệu huấn luyện
y_pred = model.predict(X)

# In ra dự đoán
print("Predictions:")
for i, pred in enumerate(y_pred):
	print(f"Data point {i + 1}: {'Positive' if pred > 0 else 'Negative'}")

# Tính và in ra độ chính xác
accuracy = np.mean(y_pred == y)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Hàm hiển thị kết quả Perceptron
def visualize_perceptron(X, y, model):
	plt.figure(figsize=(10, 8))
	
	# Vẽ dữ liệu
	plt.scatter(X[y == 1, 0], X[y == 1, 1], c='b', label='Positive')
	plt.scatter(X[y == -1, 0], X[y == -1, 1], c='r', label='Negative')
	
	# Vẽ đường phân chia
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
	Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	plt.contour(xx, yy, Z, colors='g', levels=[0], alpha=0.5, linestyles=['--'])

	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.title('Perceptron Decision Boundary')
	plt.legend()
	plt.show()

# Vẽ kết quả
visualize_perceptron(X, y, model)

