import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Đặt seed cho bộ sinh số ngẫu nhiên để đảm bảo kết quả có thể tái tạo
np.random.seed(2)

# Dữ liệu và nhãn (studying hours và pass/fail)
X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
			  2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]]).T
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# Khởi tạo và huấn luyện mô hình Logistic Regression
model = LogisticRegression()
model.fit(X, y)

# In ra các trọng số của mô hình
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Dự đoán trên tập dữ liệu huấn luyện
predictions = model.predict_proba(X)[:, 1]
print("Predicted probabilities:", predictions)

# Tách dữ liệu thành 2 nhóm dựa trên nhãn
X0 = X[y == 0]
X1 = X[y == 1]

# Vẽ dữ liệu và đường sigmoid
plt.plot(X0, np.zeros_like(X0), 'ro', markersize=8, label='Fail')
plt.plot(X1, np.ones_like(X1), 'bs', markersize=8, label='Pass')

# Tạo đường cong sigmoid
xx = np.linspace(0, 6, 1000).reshape(-1, 1)
yy = model.predict_proba(xx)[:, 1]

# Hiển thị đường sigmoid và điểm phân loại
plt.plot(xx, yy, 'g-', linewidth=2, label='Sigmoid curve')
plt.axhline(0.5, color='y', linestyle='--', label='Decision boundary')
plt.xlabel('Studying hours')
plt.ylabel('Predicted probability of pass')
plt.legend()
plt.show()