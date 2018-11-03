from __future__ import print_function
import torch

from torch.autograd import Variable

x = Variable(torch.ones(2,2),requires_grad = True)

print (x)
print (type(x))
y = x+2
print(y)

print(x.grad_fn)
print(y.grad_fn)

z = y*y*3

print (z)

out = z.mean() # 取平均数 27/4 = 4.5

print(z,out)

out.backward()

print (x.grad)