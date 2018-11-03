import torch
from torch.functional import F
from torch.autograd import Variable
import matplotlib.pyplot as plt


# y = a * x^2 + b
x = torch.unsqueeze(torch.linspace(-1,1,100),dim = 1)

y = x.pow(2)+0.2*torch.rand(x.size())
#
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

# build a nn

# x, y = Variable(x), Variable(y)
class Net(torch.nn.Module):
    def __init__(self,n_features,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_features,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(n_features=1,n_hidden=10,n_output=1) # 回归问题，只需要输出一个值就可以

print (net) #  打印搭建的神经网络结构

# train the net

optimizer = torch.optim.SGD(net.parameters(),lr = 0.2)
loss_func = torch.nn.MSELoss() # 回归预测一般使用均方误差


plt.ion()   # 画图
# plt.show()
for t in range(200):
    prediction = net(x) # 每次输入一个数，在网络内部 1-10-1
    loss = loss_func(prediction,y)

#    optimizer.zero_grad()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()