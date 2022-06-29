# top
from tqdm import tqdm
import torch.nn.utils.prune as prune
import torch
from dataset.cifar10 import getCifar10
from dataset.cifar100 import getCifar100
from dataset.mini_imagenet import get_mini_imagenet
from dataset.svhn import get_svhn
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


def get_resnet_18_pytorchmodel(path, n_class):
    model = models.resnet18()
    model.fc = nn.Linear(in_features=512, out_features=n_class, bias=True)
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    return model


def get_resnet_50_pytorchmodel(path, n_class):
    model = models.resnet50()
    model.fc = nn.Linear(in_features=2048, out_features=n_class, bias=True)
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    return model


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
_, test_loader = get_mini_imagenet(batch_size_test=100)


def test(network):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data = data.to(device)
            target = target.to(device)
            output = network(data)
            test_loss += F.nll_loss(F.log_softmax(output), target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print('Accuracy: {}/{} ({:.0f}%)'.format(
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))
    )


model = get_resnet_50_pytorchmodel('./pretrained/resnet_50_mini_imagenet_pytorchmodel/model.pth',100)
# list type is also available
#test(model)
parameters_to_prune = []


def add_paras(parameters_to_prune):
    for i, j in list(model.named_modules()):
        if isinstance(j, nn.BatchNorm2d) or isinstance(j, nn.Conv2d):
            if not j.bias == None:
                parameters_to_prune.append((j, 'bias'))
            if not j.weight == None:
                parameters_to_prune.append((j, 'weight'))


add_paras(parameters_to_prune)
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.5,
)


def remove_paras():
    for i, j in list(model.named_modules()):
        if isinstance(j, nn.BatchNorm2d) or isinstance(j, nn.Conv2d):
            if not j.bias == None:
                prune.remove(j, 'bias')
            if not j.weight == None:
                prune.remove(j, 'weight')


remove_paras()

torch.save(model.state_dict(), './pretrained/resnet_50_mini_imagenet_pruned_0.5_unretrained_pytorchmodel/model.pth')
test(model)
# endd
