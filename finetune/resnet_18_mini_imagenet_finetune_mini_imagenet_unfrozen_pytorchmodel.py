# top
from __future__ import print_function, division
from dataset.mini_imagenet import get_mini_imagenet,get_mini_imagenet2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import copy
import torch.nn.functional as F
from tqdm import tqdm

cudnn.benchmark = True
test_loader_train, test_loader_test = get_mini_imagenet2(batch_size_test_test=100)
train_loader, test_loader = get_mini_imagenet(batch_size_test=100)
dataloaders = {'train': test_loader_train, 'val': test_loader_test}
dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 25


def train_model(model, criterion, optimizer, scheduler):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            for (inputs, labels) in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model


def get_resnet_18_pytorchmodel(path):
    model = models.resnet18()
    model.fc = nn.Linear(in_features=512, out_features=100, bias=True)
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    return model


def test(network, test_loader):
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


model = get_resnet_18_pytorchmodel('./pretrained/resnet_18_mini_imagenet_pytorchmodel/model.pth')
model.fc = nn.Linear(in_features=512, out_features=100, bias=True)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler)
torch.save(model.state_dict(), './pretrained/resnet_18_mini_imagenet_finetune_mini_imagenet_unfrozen_pytorchmodel/model.pth')
test(model, test_loader)
test(model, test_loader_test)
# endd
