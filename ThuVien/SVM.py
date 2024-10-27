import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Khởi tạo dữ liệu
np.random.seed(22)
means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N)  # class 1
X1 = np.random.multivariate_normal(means[1], cov, N)  # class -1 
X = np.concatenate((X0, X1), axis=0)  # Tất cả dữ liệu
y = np.concatenate((np.ones(N), -1 * np.ones(N)))  # Nhãn

# Khởi tạo và huấn luyện mô hình SVM
model = SVC(kernel='linear', C=1e5)
model.fit(X, y)

# Lấy các trọng số w và b từ mô hình đã huấn luyện
w = model.coef_[0]
b = model.intercept_[0]

print('w = ', w)
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