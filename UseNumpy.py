import numpy as np
import pandas as pd
# import matplotlib.pyplot as pyplot
# import seaborn
# import scipy

# a = np.array(1, 2, 3)
# print(a)


# b = np.array([1, 2, 3], dtype=complex)
# print(b)

# student = np.dtype([('name','S20'), ('age', 'i1'), ('marks', 'f4')])
# print(student)

# a = np.array([('liusan', 20, 50), ('wusi', 18, 38)], dtype=student)
# print(a)

# cc = np.arange(36)
# print(cc)
# print(cc.ndim)
# dd = cc.reshape(3, 4, 3)
# print(dd.ndim)
# print(dd)
# print(dd.shape)

# xx = np.empty([2, 3], dtype=int, order='C')
# print(xx)
# yy = np.zeros([2,3], dtype=int)
# print(yy)

# # zz = np.ones(5)

# deffff = [1, 2, 3]
# sss = np.array(deffff)
# print(sss)
# zzzz = np.asarray(deffff)
# print(zzzz)

# # vvvv = np.fromarrays(deffff)
# # print(vvvv)
# # np.frombuffer
# # np.fromfile
# # np.fromfunction
# # np.fromiter

# it = iter(range(5))
# aaaaa = np.fromiter(it, dtype=float)
# print(aaaaa)

# np.arange(10)
# #等差数列
# aaaaaaaa = np.linspace(1, 1, 10)
# print(aaaaaaaa)

# #等比数列
# ss = np.logspace(10, 100, base=2)
# print(ss)

# # ss = np.logspace(0, 9, 10, base=2)
# # print(ss)

# b = np.arange(10)
# a = slice(2, 8, 2)
# print(a)
# print(b[a])
# print(b[2])
# print(b[2:])
# print(b[::-1])

# a = np.array([[1,2,3],[3,4,5],[4,5,6]])
# print(a[..., 1])
# print(a[1,...])
# print(a[:,:])
# print(a[1,:])
# print(a[1])

# x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])  
# print ('我们的数组是：')
# print (x)
# print ('\n')
# # 现在我们会打印出大于 5 的元素  
# print  ('大于 5 的元素是：')
# print (x[x >  5])

# a = np.array([np.NAN, 1, 2, 3, np.nan, 5])
# print(a)
# print(a[~np.isnan(a)])

# x = np.arange(32).reshape(4, 8)
# print(x)
# print(x[[3,2,1,1], [1,1,1,1]])

# # numpy关于矩阵运算的广播规则
# # 自动提示 然后进行运算
# a = np.array([1, 2, 3, 4])
# b = np.array([10, 20, 30, 40])
# print(a + b)
# print(a * b)

# a = np.array([[ 1 , 2 , 3], [11, 12, 13],[21, 22, 23],[31, 32 ,33]])
# # Exception has occurred: ValueError
# # operands could not be broadcast together with shapes (4,3) (4,) 
# # print(a + b)
# # 广播规则
# c = [2, 4, 5]
# print(a + c)
# print(a * c)

# a = np.arange(6).reshape(2, 3)
# for item in np.nditer(a):
#     print(item)


a = np.arange(0,60,5)
a = a.reshape(3,4)
# print(a)

# b = a.T
# print(b)

# c = b.copy(order='C')
# print(c)
# for i in np.nditer(c):
#     print(i)

# d = b.copy(order='F')
# print(d)

# for item in a:
#     print(item)

# for item in a.flat:
#     print(item)

# b = a.flatten()
# print(a)
# print(b)
# b[1] = 20
# print(b)
# print(a)

# b = a.ravel()
# b[1] = 20
# print(a)
# print(b)

# a = np.arange(8).reshape(2,2,2)
# b = np.swapaxes(a, 2, 0)
# print(a)
# print(b)

print(np.bitwise_or(11,22))
print(np.left_shift(10, 2))
print(np.right_shift(20,2))

print(np.char.add(['jjj', 'hhh'], ['rrr', 'ddd']))
print(np.char.capitalize('jjj'))
print(np.char.upper('lll'))
print(np.char.title('i am lazy'))

zz = np.char.split('i am lazy')
print(zz)
print (np.char.strip(['arunooba','admin','java'],'a'))
#使用指定分隔符连接
print (np.char.join([':','-'],['runoob','google']))


a = [1, 2, 3]
b = np.array(a)
print(type(b))
print(np.sin(a))

print(np.std(a))
np.var(a)

c = np.where(b > 3)
print(b[c])


import math
iter = map(math.sqrt, [1,4,9])
for item in iter:
    print(item)

import numpy.matlib
import numpy.matrixlib
import numpy.linalg
#除了ndarray 还有一种矩阵数据类型 numpy.matlib.matrix
aa = np.matlib.empty([2, 3], dtype=int)
print(aa)
print(type(aa))
print (np.matlib.zeros((2,2)))

a = np.array([[1,2],[3,4]])
b = np.array([[11,12],[13,14]])
print(np.dot(a, b))
# np.vdot
# np.inner
# np.outer
# np.matmul

# IO NUMPY进行加载存储 NUMPY序列化反序列化
# numpy.load()
# numpy.save()

