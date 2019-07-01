import torch as t
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torch import optim
show = ToPILImage()
from net import Net

#加载测试模型
net = Net()
net.load_state_dict(t.load("net.pth"))
# 测试集
testset = tv.datasets.CIFAR10(
    root='./data',
    train=False,
    download=False,
    transform=transform
)
testloader = t.utils.data.DataLoader(
    testset,
    batch_size=5,
    shuffle=False,
    num_workers=2
)

# 测试网络

correct = 0
total = 0
with t.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = t.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))
