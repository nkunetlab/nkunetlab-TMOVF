# top
import torch.nn.utils.prune as prune
from resnet import ResNet_18
import torch
from dataset.svhn import get_svhn
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
import torchvision.models as models


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
n_epochs = 100
batch_size_train = 64
batch_size_test = 100
learning_rate = 0.01
momentum = 0.5
log_interval = 10
train_loader, test_loader = getCifar100(batch_size_train=batch_size_train, batch_size_test=batch_size_test)
network = models.googlenet(pretrained=True, aux_logits=False)
'''
network.dropout = nn.Sequential(
    nn.Dropout(p=0.2, inplace=False),
    nn.Linear(in_features=1024, out_features=512, bias=True)
)
'''
network.fc = nn.Linear(in_features=1024, out_features=100, bias=True)
network = network.to(device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)


def train(epoch, network=network,
          model_save_dir='./pretrained/googlenet_imagenet_finetune_cifar100_pytorchmodel_unfrozen/model.pth',
          optimizer_save_dir='./pretrained/googlenet_imagenet_finetune_cifar10_pytorchmodel_unfrozen/optimizer.pth'):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = F.log_softmax(network(data))
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item())
            )
            torch.save(network.state_dict(), model_save_dir)
            # torch.save(optimizer.state_dict(), optimizer_save_dir)


def test(network):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            test_loss += F.nll_loss(F.log_softmax(output), target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    )


#test(network)
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test(network)

# endd
