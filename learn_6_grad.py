from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



# W_h = torch.randn(20,20,requires_grad = True)
# W_x = torch.randn(20,10,requires_grad = True)
#
# x = torch.randn(1,10)
#
# prev_h = torch.randn(1,20)
#
# i2h = torch.mm(W_x,x.t())
# h2h = torch.mm(W_h,prev_h.t())
# next_h = i2h+h2h
# next_h = next_h.tanh()
#
# val = next_h.backward(torch.ones(20,1))
#
# print(type(val))

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)
    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = x.view(-1,320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x,training = self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
input = Variable(torch.randn(10,25))
# output = model(input)


net = Net()
print(net)
dataset = [torch.randn(1,100),torch.randn(1,100)]
print(dataset)
optimizer = torch.optim.SGD(net.parameters(),lr = 0.01,momentum=0.9)
for input,target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = F.cross_entropy(output,target)
    print('loss : %d',loss)
    loss.backward()
    optimizer.step()