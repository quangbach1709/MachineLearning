import numpy as np
from sklearn.linear_model import LinearRegression

# Training data
X_train = np.array([[60, 2, 10], [40, 2, 5], [100, 3, 7]])
y_train = np.array([10, 12, 20])

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Print the weights
print("Các trọng số của mô hình:")
print("w_0 (hệ số chặn):", model.intercept_)
print("w_1 (hệ số diện tích):", model.coef_[0])
print("w_2 (hệ số số phòng ngủ):", model.coef_[1])
print("w_3 (hệ số khoảng cách đến trung tâm):", model.coef_[2])

# Hàm dự đoán giá nhà
def predict_house_price(area, bedrooms, distance):
	return model.predict([[area, bedrooms, distance]])[0]

# Nhập thông tin ngôi nhà mới
new_area = float(input("Nhập diện tích ngôi nhà (m2): "))
new_bedrooms = int(input("Nhập số phòng ngủ: "))
new_distance = float(input("Nhập khoảng cách đến trung tâm (km): "))

# Dự đoán giá nhà
predicted_price = predict_house_price(new_area, new_bedrooms, new_distance)

print(f"Giá dự đoán của ngôi nhà: {predicted_price:.2f} (đơn vị giá)")