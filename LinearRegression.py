import numpy as np
import matplotlib.pyplot as plt

# A=np.loadtxt('data/univariate.txt',delimiter=',')
# print(A)

# np.savetxt('data/univariate_copy.txt',A,fmt='%.2f',delimiter=',') # luu file

# #import toàn bộ file univariate.txt
# X = np.loadtxt('data/univariate.txt', delimiter = ',')
# #import Theta từ file univariate_theta.txt
# Theta = np.loadtxt('data/univariate_theta.txt')

# #Luu cot y lai
# y=np.copy(X[:,-1])

# #chuyển đổi cột đầu tiên thành cột 2
# X[:,1]=X[:,0]

# #Đổi cột đầu thành 1
# X[:,0]=1

# #tính lợi nhuận 
# predict = X@Theta

# #Chuyen lơi nhận vè $
# predict = predict*10000

# #Tin cập dân số lợi nhuận
# print('%d nguoi: %2f' %(X[0,1]*10000,predict[0]))

# #luu file
# np.savetxt('data/predict.txt',predict ,fmt='%.6f')

# #Plot giá trị thực tế (không lấy cột bias 1 đầu)
# #X[:,1:] là x-axis của biểu đồ, không lấy cột đầu; y là y-axis, rx là red x, plot dữ liệu bằng dấu x màu đỏ
# plt.plot(X[:,1:],y,'rx')

# #Plot dự đoán
# plt.plot(X[:,1:],predict/10000,'-b')#ta dùng đơn vị gốc là 10000$, -b là đường thẳng màu xanh
# #show kết quả
# plt.show()

#import toàn bộ file multivariate.txt
_X = np.loadtxt('data/multivariate.txt', delimiter = ',')

#import Theta từ file multivariate_theta.txt
Theta = np.loadtxt('data/multivariate_theta.txt')

#Tạo ma trận X bằng kích thước của _X
X = np.zeros((np.size(_X,0),np.size(_X,1)))

#Thêm cột đầu bằng 1
X[:,0] = 1

#Thêm các cột x1 -> xn
n = np.size(_X,1) - 1
X[:,1:] = _X[:,0:n]

#Tính giá nhà (đơn vị $)
predict = X @ Theta
#in bộ diện tích-số phòng-giá đầu tiên
print('%.2f feet-vuông, %d phòng ngủ : %.1f$' %(X[0,1], X[0,2], predict[0]))

#Lưu kết quả
np.savetxt('data/predicted_value.txt',predict,fmt = '%.2f')

