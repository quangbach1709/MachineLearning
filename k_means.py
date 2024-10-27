from __future__ import print_function  # Import để đảm bảo rằng print được sử dụng như một hàm, cho phép sử dụng các chức năng của Python 3.x trong Python 2.x
import numpy as np  # Import thư viện NumPy để sử dụng các chức năng toán học và mảng
import matplotlib.pyplot as plt  # Import thư viện Matplotlib để vẽ đồ thị
from scipy.spatial.distance import cdist  # Import hàm cdist từ thư viện SciPy để tính khoảng cách giữa các điểm

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

# Khởi tạo nhãn ban đầu
original_label = np.asarray([0]*N + [1]*N + [2]*N).T  # Tạo một mảng chứa nhãn ban đầu cho các điểm dữ liệu

# Hàm hiển thị kết quả K-means
def kmeans_display(X, label):
    K = np.amax(label) + 1  # Tìm số lượng cụm dựa trên giá trị lớn nhất của nhãn
    X0 = X[label == 0, :]  # Lấy các điểm dữ liệu có nhãn là 0
    X1 = X[label == 1, :]  # Lấy các điểm dữ liệu có nhãn là 1
    X2 = X[label == 2, :]  # Lấy các điểm dữ liệu có nhãn là 2
    
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4, alpha=.8)  # Vẽ các điểm dữ liệu của cụm 0
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize=4, alpha=.8)  # Vẽ các điểm dữ liệu của cụm 1
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize=4, alpha=.8)  # Vẽ các điểm dữ liệu của cụm 2

    plt.axis('equal')  # Đảm bảo rằng các trục x và y có tỷ lệ giống nhau để hiển thị các điểm dữ liệu một cách chính xác
    plt.plot()  # Vẽ đồ thị
    plt.show()  # Hiển thị đồ thị

# Hàm khởi tạo các centroid ngẫu nhiên
def initialize_centroids(X, K):
    return X[np.random.choice(X.shape[0], K, replace=False)]  # Chọn K điểm dữ liệu ngẫu nhiên làm các centroid ban đầu

# Hàm gán nhãn cho các điểm dữ liệu dựa trên khoảng cách đến centroid
def assign_labels(X, centroids):
    distances = cdist(X, centroids)  # Tính khoảng cách từ mỗi điểm dữ liệu đến các centroid
    return np.argmin(distances, axis=1)  # Gán nhãn bằng chỉ số của centroid gần nhất

# Hàm cập nhật các centroid dựa trên các điểm được gán vào từng cụm
def update_centroids(X, labels, K):
    centroids = np.zeros((K, X.shape[1]))  # Khởi tạo một mảng để lưu trữ các centroid mới
    for k in range(K):
        centroids[k, :] = np.mean(X[labels == k, :], axis=0)  # Tính trung bình của các điểm dữ liệu trong mỗi cụm để cập nhật centroid
    return centroids

# Hàm kiểm tra sự hội tụ của các centroid
def has_converged(centroids, new_centroids):
    return set([tuple(c) for c in centroids]) == set([tuple(c) for c in new_centroids])  # Kiểm tra xem các centroid có thay đổi không

# Thuật toán K-means chính
def kmeans(X, K):
    centroids = initialize_centroids(X, K)  # Khởi tạo các centroid ban đầu
    labels = np.zeros(X.shape[0])  # Khởi tạo một mảng để lưu trữ các nhãn

    while True:
        labels = assign_labels(X, centroids)  # Gán nhãn cho các điểm dữ liệu
        new_centroids = update_centroids(X, labels, K)  # Cập nhật các centroid

        # Kiểm tra sự hội tụ
        if has_converged(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, labels

# Thực hiện K-means
centroids, labels = kmeans(X, K)

# Hiển thị kết quả
kmeans_display(X, labels)
