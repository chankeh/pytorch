from __future__ import print_function

import torch


import numpy as np

# x = torch.Tensor(5,3) 
# 随机初始化
# x = torch.rand(5,3)
# print(x)
# # print (x.size())
# y = torch.rand(5,3)
# print(y)

#print(x+y)

# print(x*y)
# result = torch.Tensor(5,3)
# torch.add(x,y,out = result)
# print(result)

# print (x.add_(y))

# print (x[2,:])

# a = torch.ones(5)
# print(a)
# print (type(a))
#
# b = a.numpy()
# print(type(b))


a1 = np.ones(5)
b1 = torch.from_numpy(a1)
np.add(a1,1,out = a1)
print(a1)
print(b1)


