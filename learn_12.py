import torch
import numpy as np
a = torch.Tensor([[2,3],[4,8],[7,9]])
print('a is :{}'.format(a))
print('a size is {}'.format(a.size()))

b = torch.LongTensor([[2,3],[4,8],[7,9]])
print('b is :{}'.format(b))
print('b size is {}'.format(b.size()))

c = torch.zeros((3,2))
print ('zero tensor c :{}'.format(c))

d = torch.randn((3,2))
print('d size is :{}'.format(d))
a[0,1] = 100
# print('changed a is :{}'.format(a))
# print(a[1,0])

numpy_b = b.numpy()
print(type(numpy_b))

print('numpy_b is :{}'.format(numpy_b))

e = np.array([2,3],[4,5])

torch_e = torch.from_numpy(e)
print(type(torch_e))

print ('torch_e is : {}'.format(torch_e))
