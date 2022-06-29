# top
from tqdm import tqdm
import torch.nn.utils.prune as prune
from resnet import ResNet_18
import torch
from dataset.cifar10 import getCifar10
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(random_seed=1):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_resnet_18_pytorchmodel(path):
    model = models.resnet18()
    model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    return model


set_seed()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_epochs = 20
batch_size_train = 32
batch_size_test = 100
learning_rate = 1e-4
train_loader, test_loader = get_svhn(batch_size_train=batch_size_train, batch_size_test=batch_size_test)
teacher_model = get_resnet_18_pytorchmodel('./pretrained/resnet_18_svhn_pytorchmodel/model.pth')
student_model = models.googlenet(pretrained=True)
student_model.fc=nn.Linear(in_features=1024,out_features=10,bias=True)
student_model = student_model.to(device)
optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
temp = 7
hard_loss = nn.CrossEntropyLoss()
alpha = 0.3
soft_loss = nn.KLDivLoss(reduction='batchmean')


def train(teacher_model, student_model):
    teacher_model.eval()
    student_model.train()
    for epoch in range(n_epochs):
        for data, targets in tqdm(train_loader):
            data = data.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                teachers_preds = teacher_model(data)
            students_preds = student_model(data)
            students_loss = hard_loss(students_preds, targets)
            ditillation_loss = soft_loss(
                F.softmax(students_preds / temp, dim=1),
                F.softmax(teachers_preds / temp, dim=1)
            )
            loss = alpha * students_loss + (1 - alpha) * ditillation_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def test(model):
    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            predictions = preds.max(1).indices
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        acc = (num_correct / num_samples).item()
    print(f'Accuracy: {acc}\tNumbers: {num_correct.item()}/{num_samples}')



#test(student_model)
train(teacher_model, student_model)
torch.save(student_model.state_dict(), './pretrained/resnet18_knowledge_distilltion_googlenet_pytorchmodel_svhn/model.pth')
test(student_model)