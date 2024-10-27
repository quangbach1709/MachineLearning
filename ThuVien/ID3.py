# Đây là một chương trình Python triển khai thuật toán Cây Quyết Định ID3 sử dụng scikit-learn

# Import các thư viện cần thiết
from __future__ import print_function  # Đảm bảo tương thích với Python 2 và 3 cho hàm print
import numpy as np  # Thư viện để xử lý các phép tính số học và mảng
import pandas as pd  # Thư viện để xử lý và phân tích dữ liệu
from sklearn.tree import DecisionTreeClassifier  # Thư viện cho Cây Quyết Định

# Đoạn code chạy khi file được thực thi trực tiếp
if __name__ == "__main__":
    df = pd.read_csv('weather.csv', index_col=0)  # Đọc dữ liệu từ file CSV
    X = df.iloc[:, :-1]  # Lấy tất cả các cột trừ cột cuối cùng làm đặc trưng
    y = df.iloc[:, -1]  # Lấy cột cuối cùng làm nhãn

    # Chuyển đổi các biến phân loại thành các biến số
    X = pd.get_dummies(X)

    tree = DecisionTreeClassifier(max_depth=3, min_samples_split=2)  # Tạo cây quyết định với độ sâu tối đa là 3
    tree.fit(X, y)  # Huấn luyện cây quyết định
    print(tree.predict(X))  # In ra dự đoán cho dữ liệu huấn luyện