from __future__ import print_function

import torch

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt 
import numpy as np 

import torch.nn as nn
import torch.nn.functional as F  

import torch.optim as optim

from torch.autograd import Variable


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

dataiter = iter(testloader)
images,labels = dataiter.next()

# imshow(torchvision.utils.make_grid(images))

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


outputs = net(Variable(images))

_,pred = torch.max(outputs.data,1)
print('Predicted: ', ' '.join('%5s' % classes[pred[j]] for j in range(4)))

correct = 0
total = 0

for data in testloader:
	images,labels = data
	outputs = net(Variable(images))
	_,pred = torch.max(outputs.data,1)
	total+=labels.size(0)
	correct+=(pred==labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
	images,labels = data 
	outputs = net(Variable(images))
	_,pred = torch.max(outputs.data,1)
	c = (pred==labels).squeeze()
	for i in range(4):
		label = labels[i]
		class_correct[label]+=c[i]
		class_total[label] +=1
for i in range(10):
	print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))












