import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from tqdm import tqdm

'''
10类rgb三通道32*32的照片，数据集中一共有 50000 张训练圄片和 10000 张测试图片
作为main来运行时'./data'路径是错误的，需要'../data'才行，但如果只是import的话前者反而是正确的
'''


def get_cars_unshuffle_trainset(size=[224, 224], batch_size_train=100):
    transform = transforms.Compose(
        [transforms.Resize(size),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.StanfordCars(root='./data', split='train', download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=False, num_workers=0)
    return trainloader


def get_cars_small_trainset(size=[224, 224], batch_size_train=100):
    train_loader = get_cars_unshuffle_trainset(size=size, batch_size_train=1000)
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


if __name__ == "__main__":
    pass
