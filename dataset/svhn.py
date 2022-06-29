import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.utils.data as Data

'''
SVHN训练集73257个数字，测试集26032个数字，额外数据集531131个数字，target范围是0-9
'''


def get_svhn(size=[224, 224], batch_size_train=64, batch_size_test=1000):
    transform = transforms.Compose(
        [transforms.Resize(size),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=0)
    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=0)
    return (trainloader, testloader)


def get_svhnnotShuffle(size=[224, 224], batch_size_train=64, batch_size_test=1000):
    transform = transforms.Compose(
        [transforms.Resize(size),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=False, num_workers=0)
    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=0)
    return (trainloader, testloader)


def get_svhn2(size=[224, 224], batch_size_extra=64):
    transform = transforms.Compose(
        [transforms.Resize(size),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    extraset = torchvision.datasets.SVHN(root='./data', split='extra', download=True, transform=transform)
    extraloader = torch.utils.data.DataLoader(extraset, batch_size=batch_size_extra, shuffle=True, num_workers=0)
    return extraloader


def get_svhn_small_trainset(size=[224, 224], batch_size_train=100):
    train_loader, _ = get_svhnnotShuffle(size=size, batch_size_train=1000)
    train_x = []
    train_y = []
    for idx, (i, j) in tqdm(enumerate(train_loader)):
        if idx == 5:
            break
        train_x.append(i)
        train_y.append(j)
    train_x = torch.cat(train_x, dim=0)
    train_y = torch.cat(train_y, dim=0)
    train_set = Data.TensorDataset(train_x, train_y)
    del train_x, train_y
    train_loader_small = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=False,
                                                     num_workers=0)
    del train_set
    return train_loader_small


def get_svhn_small_testset(size=[224, 224], batch_size_test=100):
    _, test_loader = get_svhnnotShuffle(size=size, batch_size_test=1000)
    test_x = []
    test_y = []
    for idx, (i, j) in tqdm(enumerate(test_loader)):
        if idx == 5:
            break
        test_x.append(i)
        test_y.append(j)
    del test_loader
    test_x = torch.cat(test_x, dim=0)
    test_y = torch.cat(test_y, dim=0)
    test_set = Data.TensorDataset(test_x, test_y)
    del test_x, test_y
    test_loader_small = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=False,
                                                    num_workers=0)
    del test_set
    return test_loader_small


if __name__ == "__main__":
    train, test = get_svhn()
    for i in test:
        break
