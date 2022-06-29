from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import torch.utils.data as Data
import torch

'''
5w train set, 1w test set, targets' values belong to range(100)
'''


def get_new_imagenet(size=[224, 224], batch_size_train=64, batch_size_test=1000):
    transform = transforms.Compose(
        [transforms.Resize(size),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = ImageFolder(root="./data/new_imagenet/train", transform=transform)
    test_dataset = ImageFolder(root="./data/new_imagenet/test", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, num_workers=0)
    return (train_loader, test_loader)


def get_new_imagenetnotShuffle(size=[224, 224], batch_size_train=64, batch_size_test=1000):
    transform = transforms.Compose(
        [transforms.Resize(size),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = ImageFolder(root="./data/new_imagenet/train", transform=transform)
    test_dataset = ImageFolder(root="./data/new_imagenet/test", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, num_workers=0)
    return (train_loader, test_loader)


# don't return original train_loader and test_loader which may lead to wrong
def get_new_imagenet2(size=[224, 224], batch_size_test_train=64, batch_size_test_test=1000):
    # can't change 'batch_size_test=1000'
    _, test_loader = get_new_imagenet(size=size, batch_size_test=1000)
    test_loader_train_x = []
    test_loader_train_y = []
    test_loader_test_x = []
    test_loader_test_y = []
    for idx, (i, j) in tqdm(enumerate(test_loader)):
        if idx < 7:
            test_loader_train_x.append(i)
            test_loader_train_y.append(j)
        else:
            test_loader_test_x.append(i)
            test_loader_test_y.append(j)
    test_loader_train_x = torch.cat(test_loader_train_x, dim=0)
    test_loader_train_y = torch.cat(test_loader_train_y, dim=0)
    trainset = Data.TensorDataset(test_loader_train_x, test_loader_train_y)
    del test_loader_train_x, test_loader_train_y
    test_loader_train = torch.utils.data.DataLoader(trainset, batch_size=batch_size_test_train, shuffle=False,
                                                    num_workers=0)
    del trainset
    test_loader_test_x = torch.cat(test_loader_test_x, dim=0)
    test_loader_test_y = torch.cat(test_loader_test_y, dim=0)
    testset = Data.TensorDataset(test_loader_test_x, test_loader_test_y)
    del test_loader_test_x, test_loader_test_y
    test_loader_test = torch.utils.data.DataLoader(testset, batch_size=batch_size_test_test, shuffle=False,
                                                   num_workers=0)
    del testset
    return (test_loader_train, test_loader_test)


def get_new_imagenet_small_trainset(size=[224, 224], batch_size_train=100):
    train_loader, _ = get_new_imagenetnotShuffle(size=size, batch_size_train=1000)
    train_x = []
    train_y = []
    for idx, (i, j) in tqdm(enumerate(train_loader)):
        train_x.append(i[:50])
        train_x.append(i[500:550])
        train_y.append(j[:50])
        train_y.append(j[500:550])
    del train_loader
    train_x = torch.cat(train_x, dim=0)
    train_y = torch.cat(train_y, dim=0)
    train_set = Data.TensorDataset(train_x, train_y)
    del train_x, train_y
    train_loader_small = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=False,
                                                     num_workers=0)
    del train_set
    return train_loader_small


def get_new_imagenet_small_testset(size=[224, 224], batch_size_test=100):
    _, test_loader = get_new_imagenet(size=size, batch_size_test=1000)
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


if __name__ == '__main__':
    pass
