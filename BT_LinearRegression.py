import numpy as np
import matplotlib.pyplot as plt

# #Bài tập 1 cân nặng 
# # Height (cm), input data
# X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T #.T để chuyển thành ma trận chuyển vị
# # Weight (kg), output data
# y = np.array([ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

# # Building Xbar
# one = np.ones((X.shape[0], 1)) # tạo ma trận 1 cột có số hàng bằng số hàng với ma trận X
# Xbar = np.concatenate((one, X), axis = 1) # nối ma trận 1 cột với ma trận X
# # Calculating weights of the fitting line
# A = np.dot(Xbar.T, Xbar) # tính ma trận A là ma trận chuyển vị của Xbar nhân với Xbar X*X^T
# b = np.dot(Xbar.T, y) # tính ma trận b là ma trận chuyển vị của Xbar nhân với y X*Y
# w = np.dot(np.linalg.pinv(A), b) # tính ma trận w là ma trận nghịch đảo của A nhân với b A^(-1)*b
# # weights
# w_0, w_1 = w[0], w[1]

# height = int(input("Nhập chiều cao: "))

# y1 = w_1*height + w_0

# print('Input %.2fcm, predicted output %.2fkg' %(height, y1) )

#Bai tap 2 gia nha voi 3 tham so dien tich, so phong ngu, khoảng cách den trung tam

X_traim= np.array([[60, 2, 10], [40, 2, 5], [100, 3, 7]])
y_train = np.array([10, 12, 20])


# Bài tập 2: Giá nhà với 3 tham số: diện tích, số phòng ngủ, khoảng cách đến trung tâm

# Thêm cột 1 vào đầu ma trận X_train
X_train_bar = np.concatenate((np.ones((X_traim.shape[0], 1)), X_traim), axis=1)

# Tính toán trọng số w
A = np.dot(X_train_bar.T, X_train_bar)
b = np.dot(X_train_bar.T, y_train)
w = np.dot(np.linalg.pinv(A), b)

# In ra các trọng số
print("Các trọng số của mô hình:")
print("w_0 (hệ số chặn):", w[0])
print("w_1 (hệ số diện tích):", w[1])
print("w_2 (hệ số số phòng ngủ):", w[2])
print("w_3 (hệ số khoảng cách đến trung tâm):", w[3])

# Hàm dự đoán giá nhà
def predict_house_price(area, bedrooms, distance):
    return w[0] + w[1]*area + w[2]*bedrooms + w[3]*distance

# Nhập thông tin ngôi nhà mới
new_area = float(input("Nhập diện tích ngôi nhà (m2): "))
new_bedrooms = int(input("Nhập số phòng ngủ: "))
new_distance = float(input("Nhập khoảng cách đến trung tâm (km): "))

# Dự đoán giá nhà
predicted_price = predict_house_price(new_area, new_bedrooms, new_distance)

print(f"Giá dự đoán của ngôi nhà: {predicted_price:.2f} (đơn vị giá)")


