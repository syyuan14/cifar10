import torch as t
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torch import optim
from net import Net
show = ToPILImage()


# 定义数据的预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转化为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
])
# 训练集
trainset = tv.datasets.CIFAR10(
    root='./data/',
    train=True,
    transform=transform,
    download=False
)
trainloader = t.utils.data.DataLoader(
    trainset,
    batch_size=5,
    shuffle=True,
    num_workers=2,
)

classes = (
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
)


net = Net()
# 定义优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 训练网络
t.set_num_threads(8)
for epoch in range(2):
    running_loss = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # 梯度清零
        optimizer.zero_grad()

        # forward+backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # 更新参数
        optimizer.step()

        # 打印log信息
        # loss是一个scalar,需要使用loss.item()来获取数值,不能使用loss[0]
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d,%5d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
            running_loss = 0.0
t.save(net.state_dict(), "net.pth")
print("Finished Training")
