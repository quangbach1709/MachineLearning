import numpy as np
import matplotlib.pyplot as plt

# Height (cm), input data
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T #.T để chuyển thành ma trận chuyển vị
# Weight (kg), output data
y = np.array([ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

# Building Xbar
one = np.ones((X.shape[0], 1)) # tạo ma trận 1 cột có số hàng bằng số hàng với ma trận X
Xbar = np.concatenate((one, X), axis = 1) # nối ma trận 1 cột với ma trận X
# Calculating weights of the fitting line
A = np.dot(Xbar.T, Xbar) # tính ma trận A là ma trận chuyển vị của Xbar nhân với Xbar X*X^T
b = np.dot(Xbar.T, y) # tính ma trận b là ma trận chuyển vị của Xbar nhân với y X*Y
w = np.dot(np.linalg.pinv(A), b) # tính ma trận w là ma trận nghịch đảo của A nhân với b A^(-1)*b
# weights
w_0, w_1 = w[0], w[1]

height = int(input("Nhập chiều cao: "))

y1 = w_1*height + w_0

print('Input %.2fcm, predicted output %.2fkg' %(height, y1) )

