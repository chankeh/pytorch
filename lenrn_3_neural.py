from __future__ import print_function

import torch 

from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F  

class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()

		self.conv1 = nn.Conv2d(1,6,5) # 1 input 6 output 5x5 square
		self.conv2 = nn.Conv2d(6,16,5) # 6 input 16 output 5x5 square

		# add affine operation : y = wx+b
		self.fc1 = nn.Linear(16 * 5 * 5 ,120) # input 16 * 5 * 5 output 120
		self.fc2 = nn.Linear(120,84)
		self.fc3 = nn.Linear(84,10)

	def forward(self,x):
		# 进行两次池化
		x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
		x = F.max_pool2d(F.relu(self.conv2(x)),2) # 同(2*2)

		x = x.view(-1,self.num_flat_feature(x))

		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))

		x = self.fc3(x)
		return x
	def num_flat_feature(self,x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features
net = Net()
print(net)

params = list(net.parameters())

print(len(params))

for param in params:
	print(param.size())

input = Variable(torch.randn(1,1,32,32))
out = net(input)
print (out)

net.zero_grad()

out.backward(torch.randn(1,10))

out = net(input)

target  = Variable(torch.arange(1,2))

# criterion = nn.MSELoss()

criterion = nn.CrossEntropyLoss()
loss = criterion(out, target)

print(loss)

print(loss.grad_fn)

