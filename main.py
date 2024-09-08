import numpy as np
_A =[[1,2,3],[4,5,6],[7,8,9]]
_B=[[2,3,4],[7,9,10]]
_C=[7,9,21]
_D=[[1,2],[4,5],[7,8]]
A = np.array(_A)
B = np.array(_B)
C = np.array(_C)
D = np.array(_D)
# print(A)
# print('a[0][1] = ',A[0][1])# in ra phan tu o vi tri 0,1
# print('a[:][1] = ',A[:][1])# in ra phan tu o vi tri 1
# print('a[1][:] = ',A[1][:])# in ra phan tu o vi tri 1
# print('A+B = \n',A+B)
# print('A-B = \n',A-B)
# print('A*2 = \n',A*2)
# print('A/2 = \n',A/2)
# nhan voi vector
# print('A*C = \n',A.dot(C)) # nhan voi vector
# print('B*C = \n',B@C) # nhan voi vector
# print('A*D = \n',A@D) # nhan voi ma tran

#ma tran don vi
# I = np.eye(3)
# print(I)
#kiem tra ma tran don vi
# print(I ==1)

#ma tran dao nguoc
_E = [[1,2,3],[0,1,4],[5,6,0]]
E = np.array(_E)
E_inv = np.linalg.inv(E)
print(E_inv)

#ma tran chuyen vi
E_T=np.transpose(E)
print(E_T)

#dua ra so hang va so cot
print(np.size(E,0)) #so hang
print(np.size(E,1)) #so cot

#sum max min voi ma tran
print(np.sum(E)) #tinh tong ma tran
print(np.sum(E,0)) #tinh tong theo hang
print(np.sum(E,1)) #tinh tong theo cot
print(np.max(E)) #tim gia tri lon nhat ma tran
print(np.max(E,0)) #tim gia tri lon nhat theo hang
print(np.max(E,1)) #tim gia tri lon nhat theo cot
print(np.min(E)) #tim gia tri nho nhat ma tran
print(np.min(E,0)) #tim gia tri nho nhat theo hang









