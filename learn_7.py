from __future__ import print_function

import torch

from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2],[3,4]])

var = Variable(tensor,requires_grad = True)

# print(tensor)
# print (var)

t_out = torch.mean(tensor * tensor)
v_out = torch.mean(var *  var)

# print (t_out)
#
# print (v_out)

v_out.backward() # 反向传播

print(var.grad) # var进行梯度求解 ， 计算图迭代反向传播

# t_out.backward() # error
# print(tensor)  # error


# 获取Variable里面的参数
print(var.data)

print(var.data.numpy()) # 转换成numpy

print(torch.from_numpy(var.data.numpy()))
