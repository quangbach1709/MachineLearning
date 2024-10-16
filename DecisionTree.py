# Đây là một chương trình Python triển khai thuật toán Cây Quyết Định ID3

# Import các thư viện cần thiết
from __future__ import print_function  # Đảm bảo tương thích với Python 2 và 3 cho hàm print
import numpy as np  # Thư viện để xử lý các phép tính số học và mảng
import pandas as pd  # Thư viện để xử lý và phân tích dữ liệu

# Định nghĩa lớp TreeNode để biểu diễn một nút trong cây quyết định
class TreeNode(object):
    def __init__(self, ids = None, children = [], entropy = 0, depth = 0):
        self.ids = ids           # Chỉ số của dữ liệu trong nút này
        self.entropy = entropy   # Entropy của nút, sẽ được điền sau
        self.depth = depth       # Khoảng cách đến nút gốc
        self.split_attribute = None # Thuộc tính được chọn để phân chia, nếu không phải là lá
        self.children = children # Danh sách các nút con
        self.order = None       # Thứ tự các giá trị của thuộc tính phân chia trong các nút con
        self.label = None       # Nhãn của nút nếu nó là một lá

    def set_properties(self, split_attribute, order):
        # Phương thức để thiết lập thuộc tính phân chia và thứ tự
        self.split_attribute = split_attribute
        self.order = order

    def set_label(self, label):
        # Phươnga thức để thiết lập nhãn cho nút
        self.label = label

# Hàm tính entropy
def entropy(freq):
    # Loại bỏ các xác suất bằng 0
    freq_0 = freq[np.array(freq).nonzero()[0]]
    prob_0 = freq_0/float(freq_0.sum())
    return -np.sum(prob_0*np.log(prob_0))

# Định nghĩa lớp DecisionTreeID3 đểy dựng và sử dụng cây quyết định
class DecisionTreeID3(object):
    def __init__(self, max_depth= 10, min_samples_split = 2, min_gain = 1e-4):
        self.root = None  # Nút gốc của cây
        self.max_depth = max_depth  # Độ sâu tối đa của cây
        self.min_samples_split = min_samples_split  # Số mẫu tối thiểu để tiếp tục phân chia
        self.Ntrain = 0  # Số lượng mẫu huấn luyện
        self.min_gain = min_gain  # Ngưỡng lợi ích thông tin tối thiểu để tiếp tục phân chia
    
    def fit(self, data, target):
        # Phương thức để huấn luyện cây quyết định
        self.Ntrain = data.shape[0]  # Số lượng mẫu huấn luyện
        self.data = data  # Dữ liệu huấn luyện
        self.attributes = list(data)  # Danh sách các thuộc tính
        self.target = target  # Biến mục tiêu
        self.labels = target.unique()  # Các nhãn duy nhất trong biến mục tiêu
        
        ids = range(self.Ntrain)  # Tạo danh sách chỉ số cho tất cả các mẫu
        self.root = TreeNode(ids = ids, entropy = self._entropy(ids), depth = 0)  # Tạo nút gốc
        queue = [self.root]  # Hàng đợi để xây dựng cây theo chiều rộng
        while queue:
            node = queue.pop()  # Lấy một nút từ hàng đợi
            if node.depth < self.max_depth or node.entropy < self.min_gain:
                node.children = self._split(node)  # Phân chia nút
                if not node.children:  # Nếu là nút lá
                    self._set_label(node)  # Gán nhãn cho nút lá
                queue += node.children  # Thêm các nút con vào hàng đợi
            else:
                self._set_label(node)  # Gán nhãn cho nút nếu đạt độ sâu tối đa hoặc entropy quá nhỏ
                
    def _entropy(self, ids):
        # Tính entropy của một nút với các chỉ số ids
        if len(ids) == 0: return 0
        ids = [i+1 for i in ids]  # Chỉ số của pandas series bắt đầu từ 1
        freq = np.array(self.target[ids].value_counts())  # Đếm tần suất của các nhãn
        return entropy(freq)  # Tính entropy

    def _set_label(self, node):
        # Tìm nhãn cho một nút nếu nó là lá
        # Đơn giản chọn bằng cách bỏ phiếu đa số
        target_ids = [i + 1 for i in node.ids]  # target là một biến series
        node.set_label(self.target[target_ids].mode()[0])  # Chọn nhãn phổ biến nhất

    def _split(self, node):
        # Phương thức để phân chia một nút
        ids = node.ids  # Lấy danh sách các chỉ số của các mẫu trong nút hiện tại
        best_gain = 0  # Khởi tạo giá trị lợi ích thông tin tốt nhất
        best_splits = []  # Danh sách các phân chia tốt nhất
        best_attribute = None  # Thuộc tính tốt nhất để phân chia
        order = None  # Thứ tự các giá trị của thuộc tính tốt nhất
        sub_data = self.data.iloc[ids, :]  # Lấy tập dữ liệu con tương ứng với các chỉ số trong nút
        # Duyệt qua từng thuộc tính
        for i, att in enumerate(self.attributes):
            values = self.data.iloc[ids, i].unique().tolist()  # Lấy danh sách các giá trị duy nhất của thuộc tính
            if len(values) == 1: continue  # Bỏ qua nếu chỉ có một giá trị duy nhất (entropy = 0)
            splits = []  # Danh sách các phân chia cho thuộc tính hiện tại
            for val in values: 
                # Tìm các chỉ số của các mẫu có giá trị thuộc tính bằng val
                sub_ids = sub_data.index[sub_data[att] == val].tolist()
                splits.append([sub_id-1 for sub_id in sub_ids])  # Điều chỉnh chỉ số và thêm vào splits
            # Không phân chia nếu một nút con có quá ít điểm
            if min(map(len, splits)) < self.min_samples_split: continue
            # Tính lợi ích thông tin
            HxS = 0  # Entropy có điều kiện
            for split in splits:
                HxS += len(split) * self._entropy(split) / len(ids)
            gain = node.entropy - HxS  # Tính lợi ích thông tin
            if gain < self.min_gain: continue  # Dừng nếu lợi ích quá nhỏ
            # Cập nhật thuộc tính tốt nhất nếu tìm thấy lợi ích cao hơn
            if gain > best_gain:
                best_gain = gain 
                best_splits = splits
                best_attribute = att
                order = values
        # Đặt thuộc tính và thứ tự cho nút hiện tại
        node.set_properties(best_attribute, order)
        # Tạo các nút con dựa trên phân chia tốt nhất
        child_nodes = [TreeNode(ids = split,
                     entropy = self._entropy(split), depth = node.depth + 1) for split in best_splits]
        return child_nodes  # Trả về danh sách các nút con

    def predict(self, new_data):
        """
        Phương thức để dự đoán nhãn cho dữ liệu mới
        :param new_data: một dataframe mới, mỗi hàng là một điểm dữ liệu
        :return: nhãn dự đoán cho mỗi hàng
        """
        npoints = new_data.shape[0]
        labels = [None]*npoints
        for n in range(npoints):
            x = new_data.iloc[n, :]  # Một điểm dữ liệu
            # Bắt đầu từ gốc và đi xuống cho đến khi gặp một lá
            node = self.root
            while node.children: 
                node = node.children[node.order.index(x[node.split_attribute])]
            labels[n] = node.label
            
        return labels



if __name__ == "__main__":
    # Đoạn code chạy khi file được thực thi trực tiếp
    df = pd.read_csv('weather.csv', index_col = 0)  # Đọc dữ liệu từ file CSV
    X = df.iloc[:, :-1]  # Lấy tất cả các cột trừ cột cuối cùng làm đặc trưng
    y = df.iloc[:, -1]  # Lấy cột cuối cùng làm nhãn
    tree = DecisionTreeID3(max_depth = 3, min_samples_split = 2)  # Tạo cây quyết định với độ sâu tối đa là 3
    tree.fit(X, y)  # Huấn luyện cây quyết định
    print(tree.predict(X))  # In ra dự đoán cho dữ liệu huấn luyện