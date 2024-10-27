import numpy as np

# Tính chỉ số Gini cho một tập hợp nhãn
def gini_index(groups, classes):
    # Tính tổng số lượng mẫu trong tất cả các nhóm
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0  # Khởi tạo chỉ số Gini là 0
    for group in groups:
        # Tính số lượng mẫu trong mỗi nhóm
        size = float(len(group))
        if size == 0:
            continue  # Nếu nhóm không có mẫu nào, bỏ qua
        score = 0.0  # Khởi tạo điểm số cho mỗi nhóm
        labels = [row[-1] for row in group]  # Lấy nhãn của mỗi mẫu trong nhóm
        for class_val in classes:
            # Tính tỷ lệ của mỗi nhãn trong nhóm
            proportion = labels.count(class_val) / size
            # Cộng thêm tỷ lệ bình phương vào điểm số
            score += proportion * proportion
        # Tính chỉ số Gini cho nhóm và cộng vào tổng chỉ số Gini
        gini += (1.0 - score) * (size / n_instances)
    return gini  # Trả về chỉ số Gini

# Chia dữ liệu thành 2 nhóm theo một giá trị thuộc tính cụ thể
def test_split(index, value, dataset):
    left, right = [], []  # Khởi tạo hai danh sách để lưu trữ các nhóm
    for row in dataset:
        # Nếu giá trị của thuộc tính tại chỉ số cụ thể nhỏ hơn giá trị chia, thêm vào nhóm trái
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)  # Ngược lại, thêm vào nhóm phải
    return left, right  # Trả về hai nhóm

# Chọn điểm chia tốt nhất cho một nút
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))  # Lấy các giá trị nhãn duy nhất
    b_index, b_value, b_score, b_groups = 999, 999, 999, None  # Khởi tạo các biến để lưu trữ giá trị tốt nhất
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)  # Chia dữ liệu thành hai nhóm
            gini = gini_index(groups, class_values)  # Tính chỉ số Gini cho các nhóm
            # Nếu chỉ số Gini nhỏ hơn giá trị tốt nhất hiện tại, cập nhật các biến
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}  # Trả về điểm chia tốt nhất

# Tạo một nút lá (leaf)
def to_terminal(group):
    outcomes = [row[-1] for row in group]  # Lấy các nhãn của các mẫu trong nhóm
    return max(set(outcomes), key=outcomes.count)  # Trả về nhãn phổ biến nhất

# Phân chia nút thành các nhánh con hoặc tạo nút lá
def split(node, max_depth, min_size, depth):
    left, right = node['groups']  # Lấy các nhóm từ nút
    del(node['groups'])  # Xóa các nhóm khỏi nút
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)  # Nếu một trong hai nhóm rỗng, tạo nút lá
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)  # Nếu đã đạt độ sâu tối đa, tạo nút lá
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)  # Nếu nhóm trái có ít hơn min_size mẫu, tạo nút lá
    else:
        node['left'] = get_split(left)  # Ngược lại, chia nhóm trái
        split(node['left'], max_depth, min_size, depth+1)  # Tiếp tục chia nhánh con
    if len(right) <= min_size:
        node['right'] = to_terminal(right)  # Nếu nhóm phải có ít hơn min_size mẫu, tạo nút lá
    else:
        node['right'] = get_split(right)  # Ngược lại, chia nhóm phải
        split(node['right'], max_depth, min_size, depth+1)  # Tiếp tục chia nhánh con

# Xây dựng cây quyết định
def build_tree(train, max_depth, min_size):
    root = get_split(train)  # Tìm điểm chia tốt nhất cho nút gốc
    split(root, max_depth, min_size, 1)  # Bắt đầu chia nhánh từ nút gốc
    return root  # Trả về nút gốc của cây quyết định

# Dự đoán một giá trị dựa trên cây quyết định
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)  # Nếu nhánh trái là một nút, tiếp tục dự đoán
        else:
            return node['left']  # Ngược lại, trả về giá trị của nút lá
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)  # Nếu nhánh phải là một nút, tiếp tục dự đoán
        else:
            return node['right']  # Ngược lại, trả về giá trị của nút lá

# Đánh giá thuật toán CART
def accuracy_metric(actual, predicted):
    correct = sum(1 for i in range(len(actual)) if actual[i] == predicted[i])  # Tính số lượng dự đoán đúng
    return correct / len(actual) * 100.0  # Trả về độ chính xác

# Thuật toán CART
def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)  # Xây dựng cây quyết định
    predictions = [predict(tree, row) for row in test]  # Dự đoán cho các mẫu trong tập kiểm tra
    return predictions  # Trả về các dự đoán

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

# Tách tập dữ liệu thành tập huấn luyện và tập kiểm tra
train_set = dataset[:7]
test_set = dataset[7:]

# Thực hiện huấn luyện và dự đoán
max_depth = 3
min_size = 1
predictions = decision_tree(train_set, test_set, max_depth, min_size)

# Lấy nhãn thực tế của tập kiểm tra
actual = [row[-1] for row in test_set]

# Tính độ chính xác
accuracy = accuracy_metric(actual, predictions)
print(f'Predictions: {predictions}')
print(f'Actual: {actual}')
print(f'Accuracy: {accuracy:.2f}%')
