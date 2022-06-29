# top
import torch.nn.utils.prune as prune
from resnet import ResNet_18
import torch
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
from tqdm import tqdm

'''
注意把models.resnet50()的fc层的输出改为100
'''


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
trainloader, testloader = getCifar100(batch_size_train=64, batch_size_test=100)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
network = models.resnet50(pretrained=True)
network.fc = nn.Linear(network.fc.in_features, 100)
network.to(device)
optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    print('Epoch:', epoch)
    network.train()
    for batch_idx, (data, target) in tqdm(enumerate(trainloader)):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


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
for i in range(100):
    train(i)
    torch.save(network.state_dict(), './pretrained/resnet_50_cifar100_pytorchmodel/model.pth')
    test()
