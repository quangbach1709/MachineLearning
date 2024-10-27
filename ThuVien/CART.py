import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Dữ liệu mẫu
dataset = [
	[2.771244718, 1.784783929, 'A'],
	[1.728571309, 1.169761413, 'A'],
	[3.678319846, 2.81281357, 'B'],
	[3.961043357, 2.61995032, 'B'],
	[2.999208922, 2.209014212, 'A'],
	[7.497545867, 3.162953546, 'B'],
	[9.00220326, 3.339047188, 'B'],
	[7.444542326, 0.476683375, 'B'],
	[10.12493903, 3.234550982, 'B'],
	[6.642287351, 3.319983761, 'B']
]

# Chuyển đổi dữ liệu thành numpy array
dataset = np.array(dataset)

# Tách tập dữ liệu thành tập huấn luyện và tập kiểm tra
train_set = dataset[:7]
test_set = dataset[7:]

# Tách thuộc tính và nhãn
X_train = train_set[:, :-1].astype(float)
y_train = train_set[:, -1]
X_test = test_set[:, :-1].astype(float)
y_test = test_set[:, -1]

# Khởi tạo và huấn luyện mô hình cây quyết định
model = DecisionTreeClassifier(max_depth=3, min_samples_split=2)
model.fit(X_train, y_train)

# Dự đoán cho các mẫu trong tập kiểm tra
predictions = model.predict(X_test)

# Tính độ chính xác
accuracy = accuracy_score(y_test, predictions)
print(f'Dự đoán: {predictions}')
print(f'Thực tế: {y_test}')
print(f'Độ chính xác: {accuracy:.2f}%')