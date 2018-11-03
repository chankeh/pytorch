from __future__ import print_function

import torch

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt 
import numpy as np 

import torch.nn as nn
import torch.nn.functional as F  

import torch.optim as optim


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=4,shuffle=True,num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
	img = img / 2 +0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg,(1,2,0)))
	plt.show()

# dataiter = iter(trainloader)

# images,labels = dataiter.next()

# imshow(torchvision.utils.make_grid(images))

# print(''.join('%5s ' % classes[labels[j]] for j in range(4)))


class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.conv1 = nn.Conv2d(3,6,5) # 3 input 6 output with 5x5
		self.pool = nn.MaxPool2d(2,2)
		self.conv2 = nn.Conv2d(6,16,5) # 6 input 16 output 5x5 square

		# add affine operation : y = wx+b
		self.fc1 = nn.Linear(16 * 5 * 5 ,120) # input 16 * 5 * 5 output 120
		self.fc2 = nn.Linear(120,84)
		self.fc3 = nn.Linear(84,10)

	def forward(self,x):
		# 进行两次池化
		# x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
		# x = F.max_pool2d(F.relu(self.conv2(x)),2) # 同(2*2)

		# x = x.view(-1,self.num_flat_feature(x))

		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))

		x = x.view(-1,16*5*5)

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
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

for epoc in range(5):
	running_loss = 0.0
	for i,data in enumerate(trainloader,0):
		inputs,labels = data 
		optimizer.zero_grad()

		outputs = net(inputs)
		loss = criterion(outputs,labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.data[0]
		if i % 2000 == 1999:
			print('[%d,%5d] loss : %.3f' % (epoc+1,i+1,running_loss/2000))
			running_loss = 0.0

print('Finish training')























