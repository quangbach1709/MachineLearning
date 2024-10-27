from __future__ import print_function  # Import để đảm bảo rằng print được sử dụng như một hàm, cho phép sử dụng các chức năng của Python 3.x trong Python 2.x
import numpy as np  # Import thư viện NumPy để sử dụng các chức năng toán học và mảng
import matplotlib.pyplot as plt  # Import thư viện Matplotlib để vẽ đồ thị
from sklearn.cluster import KMeans  # Import KMeans từ thư viện scikit-learn

np.random.seed(11)  # Đặt hạt giống ngẫu nhiên để đảm bảo rằng các kết quả ngẫu nhiên được tái tạo

# Tạo dữ liệu
means = [[2, 2], [8, 3], [3, 6]]  # Định nghĩa các vị trí trung bình của các cụm
cov = [[1, 0], [0, 1]]  # Định nghĩa ma trận hiệp phương sai cho các điểm dữ liệu
N = 500  # Số lượng điểm dữ liệu cho mỗi cụm
X0 = np.random.multivariate_normal(means[0], cov, N)  # Tạo N điểm dữ liệu ngẫu nhiên cho cụm 0
X1 = np.random.multivariate_normal(means[1], cov, N)  # Tạo N điểm dữ liệu ngẫu nhiên cho cụm 1
X2 = np.random.multivariate_normal(means[2], cov, N)  # Tạo N điểm dữ liệu ngẫu nhiên cho cụm 2

X = np.concatenate((X0, X1, X2), axis=0)  # Nối các điểm dữ liệu của các cụm lại để tạo thành một mảng duy nhất
K = 3  # Số lượng cụm

# Khởi tạo và huấn luyện mô hình K-means
kmeans = KMeans(n_clusters=K, random_state=11)
kmeans.fit(X)

# Lấy các nhãn và centroid
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Hàm hiển thị kết quả K-means
def kmeans_display(X, label, centroids):
	K = np.amax(label) + 1  # Tìm số lượng cụm dựa trên giá trị lớn nhất của nhãn
	for k in range(K):
		cluster = X[label == k]
		plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {k}')
	
	plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='*', label='Centroids')
	plt.legend()
	plt.axis('equal')  # Đảm bảo rằng các trục x và y có tỷ lệ giống nhau để hiển thị các điểm dữ liệu một cách chính xác
	plt.show()  # Hiển thị đồ thị

# Hiển thị kết quả
kmeans_display(X, labels, centroids)