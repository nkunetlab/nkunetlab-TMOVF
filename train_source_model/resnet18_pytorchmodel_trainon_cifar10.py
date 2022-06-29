# top
import torch.nn.utils.prune as prune
from resnet import ResNet_18
import torch
from dataset.cifar10 import getCifar10
from dataset.cifar100 import getCifar100
import itertools
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
import numpy as np
import random
from neural_gas import NeuralGas
import torchvision.transforms as transforms
import torchvision.models as models

'''
注意把models.resnet18()的fc层的输出改为10
'''

transform = transforms.Compose(
    [transforms.Resize([224, 224]),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)


def set_seed(random_seed=1):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = models.resnet18()
network.fc = nn.Linear(network.fc.in_features, 10)
network.to(device)
network.load_state_dict(torch.load('./pretrained/resnet_18_cifar10_pytorchmodel/model.pth'))
optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(trainloader.dataset),
                100. * batch_idx / len(trainloader),
                loss.item())
            )


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(testloader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(testloader.dataset),
        100. * correct / len(testloader.dataset))
    )


test()
for i in range(20):
    train(i)
    torch.save(network.state_dict(), './pretrained/resnet_18_cifar10_pytorchmodel/model.pth')
    test()
