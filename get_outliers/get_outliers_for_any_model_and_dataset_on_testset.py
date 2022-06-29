# top
import pandas as pd
import torch.nn as nn
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from resnet import ResNet_18
import torch
from dataset.cifar10 import getCifar10,get_cifar10_small_testset
from dataset.cifar100 import getCifar100,get_cifar100_small_testset
from dataset.svhn import get_svhn,get_svhn_small_testset
from dataset.mini_imagenet import get_mini_imagenet,get_mini_imagenet_small_testset
import numpy as np
import random
from neural_gas import NeuralGas
import torchvision.models as models
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from pyod.models.mad import MAD
from pyod.models.rod import ROD
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF


def tocsv(name, i, prefix):
    list0 = name.tolist()
    df = pd.DataFrame(data=list0)
    df.to_csv(prefix + f'/list{i}.csv', mode="a", encoding="utf_8_sig", header=1, index=0)


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
fmap_block = torch.tensor([])


def forward_hook(module, data_input, data_output):
    global fmap_block
    if fmap_block.shape == torch.Size([0]):
        fmap_block = data_output
    else:
        fmap_block = torch.cat((fmap_block, data_output), 0)


def get_resnet_18_pytorchmodel(path, n_classes):
    model = models.resnet18()
    model.fc = nn.Linear(in_features=512, out_features=n_classes, bias=True)
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    model.avgpool.register_forward_hook(forward_hook)
    return model


def get_resnet_50_pytorchmodel(path, n_classes):
    model = models.resnet50()
    model.fc = nn.Linear(in_features=2048, out_features=n_classes, bias=True)
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    model.avgpool.register_forward_hook(forward_hook)
    return model


def get_googlenet_pytorchmodel(path, n_classes):
    model = models.googlenet(pretrained=True, aux_logits=False)
    model.fc = nn.Linear(in_features=1024, out_features=n_classes, bias=True)
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    model.avgpool.register_forward_hook(forward_hook)
    return model


def get_vgg16_pytorchmodel(path, n_classes):
    model = models.vgg16(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=n_classes, bias=True),
    )
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    model.avgpool.register_forward_hook(forward_hook)
    return model


def get_vgg11_pytorchmodel(path, n_classes):
    model = models.vgg11(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=n_classes, bias=True),
    )
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    model.avgpool.register_forward_hook(forward_hook)
    return model


def test(model):
    global fmap_block
    fmap_block = torch.tensor([])
    with torch.no_grad():
        for i, (data, target) in enumerate(testloader):
            if i % 10 == 0:
                print(f'in test(): {i}/{len(testloader)}')
            data = data.to(device)
            target = target.to(device)
            output = model(data)
    fmap_block = fmap_block.view(testloader.dataset.__len__(), -1)
    print('start move fmap_block to cpu')
    fmap_block = fmap_block.cpu().numpy()


# useless for the moment
def get_res(model):
    global fmap_block
    fmap_block = fmap_block.view(10000, -1)
    fmap_block = fmap_block.cpu().numpy()
    NG = NeuralGas(5000, 512, 0.1, 3)
    NG.setup(fmap_block)
    res = NG.train(fmap_block, 500, False)
    return res


def get_train_scores(model):
    clf = ECOD()
    test(model)
    # res = get_res(model)
    global fmap_block
    print('start clf.fit()')
    clf.fit(fmap_block)
    y_train_scores = clf.decision_scores_
    return y_train_scores


# redef for other outlier detection method
def get_train_scores(model):
    clf = COPOD()
    test(model)
    # res = get_res(model)
    global fmap_block
    print('start clf.fit()')
    clf.fit(fmap_block)
    y_train_scores = clf.decision_scores_
    return y_train_scores


# redef for other addr
def tocsv(name, i, prefix):
    list0 = name.tolist()
    df = pd.DataFrame(data=list0)
    df.to_csv('./COPOD' + prefix[1:] + f'/list{i}.csv', mode="a", encoding="utf_8_sig", header=1, index=0)


def get_svhn_small_test(size=[224, 224], batch_size_test=1000):
    _, test_loader = get_svhn(size=size, batch_size_test=100)
    li = []
    lj = []
    for idx, (i, j) in tqdm(enumerate(test_loader)):
        if idx == 100:
            break
        li.append(i)
        lj.append(j)
    dataset_x = torch.cat(li, dim=0)
    dataset_y = torch.cat(lj, dim=0)
    dataset = torch.utils.data.TensorDataset(dataset_x, dataset_y)
    res = torch.utils.data.DataLoader(dataset, batch_size=batch_size_test, shuffle=False, num_workers=0)
    return res


'''
# redef for other outlier detection method
def get_train_scores(model):
    clf = ROD()
    test(model)
    # res = get_res(model)
    global fmap_block
    print('start clf.fit()')
    clf.fit(fmap_block)
    y_train_scores = clf.decision_scores_
    return y_train_scores


# redef for other addr
def tocsv(name, i, prefix):
    list0 = name.tolist()
    df = pd.DataFrame(data=list0)
    df.to_csv('./ROD' + prefix[1:] + f'/list{i}.csv', mode="a", encoding="utf_8_sig", header=1, index=0)



# cifar10+resnet18
_, testloader = getCifar10(batch_size_test=100)
y_train_scores_1 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar10_pytorchmodel/model.pth', 10))
y_train_scores_2 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar10_pruned_0.1_unretrained_pytorchmodel/model.pth', 10))
y_train_scores_3 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar10_pruned_0.3_unretrained_pytorchmodel/model.pth', 10))
y_train_scores_4 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar10_pruned_0.5_unretrained_pytorchmodel/model.pth', 10))
y_train_scores_5 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar10_finetune_cifar10_unfrozen_pytorchmodel/model.pth', 10))
y_train_scores_6 = get_train_scores(
    get_vgg11_pytorchmodel('./pretrained/resnet18_knowledge_distilltion_vgg11_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_1, 1, './results/cifar10/resnet18')
tocsv(y_train_scores_2, 2, './results/cifar10/resnet18')
tocsv(y_train_scores_3, 3, './results/cifar10/resnet18')
tocsv(y_train_scores_4, 4, './results/cifar10/resnet18')
tocsv(y_train_scores_5, 5, './results/cifar10/resnet18')
tocsv(y_train_scores_6, 6, './results/cifar10/resnet18')

# cifar10+resnet50
_, testloader = getCifar10(batch_size_test=100)
y_train_scores_1 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar10_pytorchmodel/model.pth', 10))
y_train_scores_2 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar10_pruned_0.1_unretrained_pytorchmodel/model.pth', 10))
y_train_scores_3 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar10_pruned_0.3_unretrained_pytorchmodel/model.pth', 10))
y_train_scores_4 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar10_pruned_0.5_unretrained_pytorchmodel/model.pth', 10))
y_train_scores_5 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar10_finetune_cifar10_unfrozen_pytorchmodel/model.pth', 10))
y_train_scores_6 = get_train_scores(
    get_vgg11_pytorchmodel('./pretrained/resnet50_knowledge_distilltion_vgg11_pytorchmodel_cifar10/model.pth', 10))
tocsv(y_train_scores_1, 1, './results/cifar10/resnet50')
tocsv(y_train_scores_2, 2, './results/cifar10/resnet50')
tocsv(y_train_scores_3, 3, './results/cifar10/resnet50')
tocsv(y_train_scores_4, 4, './results/cifar10/resnet50')
tocsv(y_train_scores_5, 5, './results/cifar10/resnet50')
tocsv(y_train_scores_6, 6, './results/cifar10/resnet50')

# svhn+resnet18
testloader = get_svhn_small_test(batch_size_test=100)
y_train_scores_1 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_svhn_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_1, 1, './results/svhn/resnet18')
del y_train_scores_1
y_train_scores_2 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_svhn_pruned_0.1_unretrained_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_2, 2, './results/svhn/resnet18')
del y_train_scores_2
y_train_scores_3 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_svhn_pruned_0.3_unretrained_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_3, 3, './results/svhn/resnet18')
del y_train_scores_3
y_train_scores_4 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_svhn_pruned_0.5_unretrained_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_4, 4, './results/svhn/resnet18')
del y_train_scores_4
y_train_scores_5 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_svhn_finetune_svhn_unfrozen_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_5, 5, './results/svhn/resnet18')
del y_train_scores_5

y_train_scores_6 = get_train_scores(
    get_vgg11_pytorchmodel('./pretrained/resnet18_knowledge_distilltion_vgg11_pytorchmodel_svhn/model.pth', 10))
tocsv(y_train_scores_6, 6, './results/svhn/resnet18')
del y_train_scores_6

# svhn+resnet50
testloader = get_svhn_small_test(batch_size_test=100)
y_train_scores_1 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_svhn_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_1, 1, './results/svhn/resnet50')
del y_train_scores_1
y_train_scores_2 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_svhn_pruned_0.1_unretrained_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_2, 2, './results/svhn/resnet50')
del y_train_scores_2
y_train_scores_3 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_svhn_pruned_0.3_unretrained_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_3, 3, './results/svhn/resnet50')
del y_train_scores_3
y_train_scores_4 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_svhn_pruned_0.5_unretrained_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_4, 4, './results/svhn/resnet50')
del y_train_scores_4
y_train_scores_5 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_svhn_finetune_svhn_unfrozen_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_5, 5, './results/svhn/resnet50')
del y_train_scores_5
y_train_scores_6 = get_train_scores(
    get_vgg11_pytorchmodel('./pretrained/resnet50_knowledge_distilltion_vgg11_pytorchmodel_svhn/model.pth', 10))
tocsv(y_train_scores_6, 6, './results/svhn/resnet50')
del y_train_scores_6

# cifar100+resnet18
_, testloader = getCifar100(batch_size_test=100)
y_train_scores_1 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar100_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_1, 1, './results/cifar100/resnet18')
y_train_scores_2 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar100_pruned_0.1_unretrained_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_2, 2, './results/cifar100/resnet18')
y_train_scores_3 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar100_pruned_0.3_unretrained_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_3, 3, './results/cifar100/resnet18')
y_train_scores_4 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar100_pruned_0.5_unretrained_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_4, 4, './results/cifar100/resnet18')
y_train_scores_5 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar100_finetune_cifar100_unfrozen_pytorchmodel/model.pth',
                               100))
tocsv(y_train_scores_5, 5, './results/cifar100/resnet18')
y_train_scores_6 = get_train_scores(
    get_vgg11_pytorchmodel('./pretrained/resnet18_knowledge_distilltion_vgg11_pytorchmodel_cifar100/model.pth', 100))
tocsv(y_train_scores_6, 6, './results/cifar100/resnet18')

# cifar100+resnet50
_, testloader = getCifar100(batch_size_test=100)
y_train_scores_1 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar100_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_1, 1, './results/cifar100/resnet50')
y_train_scores_2 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar100_pruned_0.1_unretrained_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_2, 2, './results/cifar100/resnet50')
y_train_scores_3 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar100_pruned_0.3_unretrained_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_3, 3, './results/cifar100/resnet50')
y_train_scores_4 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar100_pruned_0.5_unretrained_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_4, 4, './results/cifar100/resnet50')
y_train_scores_5 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar100_finetune_cifar100_unfrozen_pytorchmodel/model.pth',
                               100))
tocsv(y_train_scores_5, 5, './results/cifar100/resnet50')
y_train_scores_6 = get_train_scores(
    get_vgg11_pytorchmodel('./pretrained/resnet50_knowledge_distilltion_vgg11_pytorchmodel_cifar100/model.pth', 100))
tocsv(y_train_scores_6, 6, './results/cifar100/resnet50')

# mini_imagenet+resnet18
_, testloader = get_mini_imagenet(batch_size_test=100)
y_train_scores_1 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_mini_imagenet_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_1, 1, './results/mini_imagenet/resnet18')
y_train_scores_2 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_mini_imagenet_pruned_0.1_unretrained_pytorchmodel/model.pth',
                               100))
tocsv(y_train_scores_2, 2, './results/mini_imagenet/resnet18')
y_train_scores_3 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_mini_imagenet_pruned_0.3_unretrained_pytorchmodel/model.pth',
                               100))
tocsv(y_train_scores_3, 3, './results/mini_imagenet/resnet18')
y_train_scores_4 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_mini_imagenet_pruned_0.5_unretrained_pytorchmodel/model.pth',
                               100))
tocsv(y_train_scores_4, 4, './results/mini_imagenet/resnet18')
y_train_scores_5 = get_train_scores(
    get_resnet_18_pytorchmodel(
        './pretrained/resnet_18_mini_imagenet_finetune_mini_imagenet_unfrozen_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_5, 5, './results/mini_imagenet/resnet18')
y_train_scores_6 = get_train_scores(
    get_vgg11_pytorchmodel('./pretrained/resnet18_knowledge_distilltion_vgg11_pytorchmodel_mini_imagenet/model.pth',
                           100))
tocsv(y_train_scores_6, 6, './results/mini_imagenet/resnet18')

# mini_imagenet+resnet50
_, testloader = get_mini_imagenet(batch_size_test=100)
y_train_scores_1 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_mini_imagenet_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_1, 1, './results/mini_imagenet/resnet50')
y_train_scores_2 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_mini_imagenet_pruned_0.1_unretrained_pytorchmodel/model.pth',
                               100))
tocsv(y_train_scores_2, 2, './results/mini_imagenet/resnet50')
y_train_scores_3 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_mini_imagenet_pruned_0.3_unretrained_pytorchmodel/model.pth',
                               100))
tocsv(y_train_scores_3, 3, './results/mini_imagenet/resnet50')
y_train_scores_4 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_mini_imagenet_pruned_0.5_unretrained_pytorchmodel/model.pth',
                               100))
tocsv(y_train_scores_4, 4, './results/mini_imagenet/resnet50')
y_train_scores_5 = get_train_scores(
    get_resnet_50_pytorchmodel(
        './pretrained/resnet_50_mini_imagenet_finetune_mini_imagenet_unfrozen_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_5, 5, './results/mini_imagenet/resnet50')
y_train_scores_6 = get_train_scores(
    get_vgg11_pytorchmodel('./pretrained/resnet50_knowledge_distilltion_vgg11_pytorchmodel_mini_imagenet/model.pth',
                           100))
tocsv(y_train_scores_6, 6, './results/mini_imagenet/resnet50')

# googlenet
_, testloader = getCifar10(batch_size_test=100)
y_train_scores_1 = get_train_scores(
    get_googlenet_pytorchmodel('./pretrained/googlenet_imagenet_finetune_cifar10_pytorchmodel_unfrozen/model.pth', 10))
tocsv(y_train_scores_1, 1, './results/cifar10/googlenet')
_, testloader = getCifar100(batch_size_test=100)
y_train_scores_2 = get_train_scores(
    get_googlenet_pytorchmodel('./pretrained/googlenet_imagenet_finetune_cifar100_pytorchmodel_unfrozen/model.pth',
                               100))
tocsv(y_train_scores_2, 2, './results/cifar100/googlenet')
testloader = get_svhn_small_test(batch_size_test=100)
y_train_scores_3 = get_train_scores(
    get_googlenet_pytorchmodel('./pretrained/googlenet_imagenet_finetune_svhn_pytorchmodel_unfrozen/model.pth', 10))
tocsv(y_train_scores_3, 3, './results/svhn/googlenet')
_, testloader = get_mini_imagenet(batch_size_test=100)
y_train_scores_4 = get_train_scores(
    get_googlenet_pytorchmodel('./pretrained/googlenet_imagenet_finetune_mini_imagenet_pytorchmodel_unfrozen/model.pth',
                               100))
tocsv(y_train_scores_4, 4, './results/mini_imagenet/googlenet')
# end ROD
'''

cifar10_test_loader = get_cifar10_small_testset(batch_size_test=100)
cifar100_test_loader = get_cifar100_small_testset(batch_size_test=100)
svhn_test_loader = get_svhn_small_testset(batch_size_test=100)
mini_imagenet_test_loader = get_mini_imagenet_small_testset(batch_size_test=100)
'''

# redef for other outlier detection method
def get_train_scores(model):
    clf = CBLOF()
    test(model)
    # res = get_res(model)
    global fmap_block
    print('start clf.fit()')
    clf.fit(fmap_block)
    y_train_scores = clf.decision_scores_
    return y_train_scores


# redef for other addr
def tocsv(name, i, prefix):
    list0 = name.tolist()
    df = pd.DataFrame(data=list0)
    df.to_csv('./CBLOF' + prefix[1:] + f'/list{i}.csv', mode="a", encoding="utf_8_sig", header=1, index=0)


# cifar10+resnet18
testloader = cifar10_test_loader
y_train_scores_1 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar10_pytorchmodel/model.pth', 10))
y_train_scores_2 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar10_pruned_0.1_unretrained_pytorchmodel/model.pth', 10))
y_train_scores_3 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar10_pruned_0.3_unretrained_pytorchmodel/model.pth', 10))
y_train_scores_4 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar10_pruned_0.5_unretrained_pytorchmodel/model.pth', 10))
y_train_scores_5 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar10_finetune_cifar10_unfrozen_pytorchmodel/model.pth', 10))
y_train_scores_6 = get_train_scores(
    get_vgg11_pytorchmodel('./pretrained/resnet18_knowledge_distilltion_vgg11_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_1, 1, './results/cifar10/resnet18')
tocsv(y_train_scores_2, 2, './results/cifar10/resnet18')
tocsv(y_train_scores_3, 3, './results/cifar10/resnet18')
tocsv(y_train_scores_4, 4, './results/cifar10/resnet18')
tocsv(y_train_scores_5, 5, './results/cifar10/resnet18')
tocsv(y_train_scores_6, 6, './results/cifar10/resnet18')

# cifar10+resnet50
testloader = cifar10_test_loader
y_train_scores_1 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar10_pytorchmodel/model.pth', 10))
y_train_scores_2 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar10_pruned_0.1_unretrained_pytorchmodel/model.pth', 10))
y_train_scores_3 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar10_pruned_0.3_unretrained_pytorchmodel/model.pth', 10))
y_train_scores_4 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar10_pruned_0.5_unretrained_pytorchmodel/model.pth', 10))
y_train_scores_5 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar10_finetune_cifar10_unfrozen_pytorchmodel/model.pth', 10))
y_train_scores_6 = get_train_scores(
    get_vgg11_pytorchmodel('./pretrained/resnet50_knowledge_distilltion_vgg11_pytorchmodel_cifar10/model.pth', 10))
tocsv(y_train_scores_1, 1, './results/cifar10/resnet50')
tocsv(y_train_scores_2, 2, './results/cifar10/resnet50')
tocsv(y_train_scores_3, 3, './results/cifar10/resnet50')
tocsv(y_train_scores_4, 4, './results/cifar10/resnet50')
tocsv(y_train_scores_5, 5, './results/cifar10/resnet50')
tocsv(y_train_scores_6, 6, './results/cifar10/resnet50')

# svhn+resnet18
testloader = svhn_test_loader
y_train_scores_1 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_svhn_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_1, 1, './results/svhn/resnet18')
del y_train_scores_1
y_train_scores_2 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_svhn_pruned_0.1_unretrained_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_2, 2, './results/svhn/resnet18')
del y_train_scores_2
y_train_scores_3 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_svhn_pruned_0.3_unretrained_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_3, 3, './results/svhn/resnet18')
del y_train_scores_3
y_train_scores_4 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_svhn_pruned_0.5_unretrained_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_4, 4, './results/svhn/resnet18')
del y_train_scores_4
y_train_scores_5 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_svhn_finetune_svhn_unfrozen_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_5, 5, './results/svhn/resnet18')
del y_train_scores_5

y_train_scores_6 = get_train_scores(
    get_vgg11_pytorchmodel('./pretrained/resnet18_knowledge_distilltion_vgg11_pytorchmodel_svhn/model.pth', 10))
tocsv(y_train_scores_6, 6, './results/svhn/resnet18')
del y_train_scores_6

# svhn+resnet50
testloader = svhn_test_loader
y_train_scores_1 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_svhn_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_1, 1, './results/svhn/resnet50')
del y_train_scores_1
y_train_scores_2 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_svhn_pruned_0.1_unretrained_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_2, 2, './results/svhn/resnet50')
del y_train_scores_2
y_train_scores_3 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_svhn_pruned_0.3_unretrained_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_3, 3, './results/svhn/resnet50')
del y_train_scores_3
y_train_scores_4 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_svhn_pruned_0.5_unretrained_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_4, 4, './results/svhn/resnet50')
del y_train_scores_4
y_train_scores_5 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_svhn_finetune_svhn_unfrozen_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_5, 5, './results/svhn/resnet50')
del y_train_scores_5
y_train_scores_6 = get_train_scores(
    get_vgg11_pytorchmodel('./pretrained/resnet50_knowledge_distilltion_vgg11_pytorchmodel_svhn/model.pth', 10))
tocsv(y_train_scores_6, 6, './results/svhn/resnet50')
del y_train_scores_6

# cifar100+resnet18
testloader = cifar100_test_loader
y_train_scores_1 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar100_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_1, 1, './results/cifar100/resnet18')
y_train_scores_2 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar100_pruned_0.1_unretrained_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_2, 2, './results/cifar100/resnet18')
y_train_scores_3 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar100_pruned_0.3_unretrained_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_3, 3, './results/cifar100/resnet18')
y_train_scores_4 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar100_pruned_0.5_unretrained_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_4, 4, './results/cifar100/resnet18')
y_train_scores_5 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar100_finetune_cifar100_unfrozen_pytorchmodel/model.pth',
                               100))
tocsv(y_train_scores_5, 5, './results/cifar100/resnet18')
y_train_scores_6 = get_train_scores(
    get_vgg11_pytorchmodel('./pretrained/resnet18_knowledge_distilltion_vgg11_pytorchmodel_cifar100/model.pth', 100))
tocsv(y_train_scores_6, 6, './results/cifar100/resnet18')

# cifar100+resnet50
testloader = cifar100_test_loader
y_train_scores_1 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar100_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_1, 1, './results/cifar100/resnet50')
y_train_scores_2 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar100_pruned_0.1_unretrained_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_2, 2, './results/cifar100/resnet50')
y_train_scores_3 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar100_pruned_0.3_unretrained_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_3, 3, './results/cifar100/resnet50')
y_train_scores_4 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar100_pruned_0.5_unretrained_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_4, 4, './results/cifar100/resnet50')
y_train_scores_5 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar100_finetune_cifar100_unfrozen_pytorchmodel/model.pth',
                               100))
tocsv(y_train_scores_5, 5, './results/cifar100/resnet50')
y_train_scores_6 = get_train_scores(
    get_vgg11_pytorchmodel('./pretrained/resnet50_knowledge_distilltion_vgg11_pytorchmodel_cifar100/model.pth', 100))
tocsv(y_train_scores_6, 6, './results/cifar100/resnet50')

# mini_imagenet+resnet18
testloader = mini_imagenet_test_loader
y_train_scores_1 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_mini_imagenet_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_1, 1, './results/mini_imagenet/resnet18')
y_train_scores_2 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_mini_imagenet_pruned_0.1_unretrained_pytorchmodel/model.pth',
                               100))
tocsv(y_train_scores_2, 2, './results/mini_imagenet/resnet18')
y_train_scores_3 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_mini_imagenet_pruned_0.3_unretrained_pytorchmodel/model.pth',
                               100))
tocsv(y_train_scores_3, 3, './results/mini_imagenet/resnet18')
y_train_scores_4 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_mini_imagenet_pruned_0.5_unretrained_pytorchmodel/model.pth',
                               100))
tocsv(y_train_scores_4, 4, './results/mini_imagenet/resnet18')
y_train_scores_5 = get_train_scores(
    get_resnet_18_pytorchmodel(
        './pretrained/resnet_18_mini_imagenet_finetune_mini_imagenet_unfrozen_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_5, 5, './results/mini_imagenet/resnet18')
y_train_scores_6 = get_train_scores(
    get_vgg11_pytorchmodel('./pretrained/resnet18_knowledge_distilltion_vgg11_pytorchmodel_mini_imagenet/model.pth',
                           100))
tocsv(y_train_scores_6, 6, './results/mini_imagenet/resnet18')

# mini_imagenet+resnet50
testloader = mini_imagenet_test_loader
y_train_scores_1 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_mini_imagenet_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_1, 1, './results/mini_imagenet/resnet50')
y_train_scores_2 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_mini_imagenet_pruned_0.1_unretrained_pytorchmodel/model.pth',
                               100))
tocsv(y_train_scores_2, 2, './results/mini_imagenet/resnet50')
y_train_scores_3 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_mini_imagenet_pruned_0.3_unretrained_pytorchmodel/model.pth',
                               100))
tocsv(y_train_scores_3, 3, './results/mini_imagenet/resnet50')
y_train_scores_4 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_mini_imagenet_pruned_0.5_unretrained_pytorchmodel/model.pth',
                               100))
tocsv(y_train_scores_4, 4, './results/mini_imagenet/resnet50')
y_train_scores_5 = get_train_scores(
    get_resnet_50_pytorchmodel(
        './pretrained/resnet_50_mini_imagenet_finetune_mini_imagenet_unfrozen_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_5, 5, './results/mini_imagenet/resnet50')
y_train_scores_6 = get_train_scores(
    get_vgg11_pytorchmodel('./pretrained/resnet50_knowledge_distilltion_vgg11_pytorchmodel_mini_imagenet/model.pth',
                           100))
tocsv(y_train_scores_6, 6, './results/mini_imagenet/resnet50')

# googlenet
testloader = cifar10_test_loader
y_train_scores_1 = get_train_scores(
    get_googlenet_pytorchmodel('./pretrained/googlenet_imagenet_finetune_cifar10_pytorchmodel_unfrozen/model.pth', 10))
tocsv(y_train_scores_1, 1, './results/cifar10/googlenet')
testloader = cifar100_test_loader
y_train_scores_2 = get_train_scores(
    get_googlenet_pytorchmodel('./pretrained/googlenet_imagenet_finetune_cifar100_pytorchmodel_unfrozen/model.pth',
                               100))
tocsv(y_train_scores_2, 2, './results/cifar100/googlenet')
testloader = svhn_test_loader
y_train_scores_3 = get_train_scores(
    get_googlenet_pytorchmodel('./pretrained/googlenet_imagenet_finetune_svhn_pytorchmodel_unfrozen/model.pth', 10))
tocsv(y_train_scores_3, 3, './results/svhn/googlenet')
testloader = mini_imagenet_test_loader
y_train_scores_4 = get_train_scores(
    get_googlenet_pytorchmodel('./pretrained/googlenet_imagenet_finetune_mini_imagenet_pytorchmodel_unfrozen/model.pth',
                               100))
tocsv(y_train_scores_4, 4, './results/mini_imagenet/googlenet')


# end CBLOF
'''
# redef for other outlier detection method
def get_train_scores(model):
    clf = COF()
    test(model)
    # res = get_res(model)
    global fmap_block
    print('start clf.fit()')
    clf.fit(fmap_block)
    y_train_scores = clf.decision_scores_
    return y_train_scores


# redef for other addr
def tocsv(name, i, prefix):
    list0 = name.tolist()
    df = pd.DataFrame(data=list0)
    df.to_csv('./COF' + prefix[1:] + f'/list{i}.csv', mode="a", encoding="utf_8_sig", header=1, index=0)


# cifar10+resnet18
testloader = cifar10_test_loader
y_train_scores_1 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar10_pytorchmodel/model.pth', 10))
y_train_scores_2 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar10_pruned_0.1_unretrained_pytorchmodel/model.pth', 10))
y_train_scores_3 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar10_pruned_0.3_unretrained_pytorchmodel/model.pth', 10))
y_train_scores_4 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar10_pruned_0.5_unretrained_pytorchmodel/model.pth', 10))
y_train_scores_5 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar10_finetune_cifar10_unfrozen_pytorchmodel/model.pth', 10))
y_train_scores_6 = get_train_scores(
    get_vgg11_pytorchmodel('./pretrained/resnet18_knowledge_distilltion_vgg11_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_1, 1, './results/cifar10/resnet18')
tocsv(y_train_scores_2, 2, './results/cifar10/resnet18')
tocsv(y_train_scores_3, 3, './results/cifar10/resnet18')
tocsv(y_train_scores_4, 4, './results/cifar10/resnet18')
tocsv(y_train_scores_5, 5, './results/cifar10/resnet18')
tocsv(y_train_scores_6, 6, './results/cifar10/resnet18')

# cifar10+resnet50
testloader = cifar10_test_loader
y_train_scores_1 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar10_pytorchmodel/model.pth', 10))
y_train_scores_2 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar10_pruned_0.1_unretrained_pytorchmodel/model.pth', 10))
y_train_scores_3 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar10_pruned_0.3_unretrained_pytorchmodel/model.pth', 10))
y_train_scores_4 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar10_pruned_0.5_unretrained_pytorchmodel/model.pth', 10))
y_train_scores_5 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar10_finetune_cifar10_unfrozen_pytorchmodel/model.pth', 10))
y_train_scores_6 = get_train_scores(
    get_vgg11_pytorchmodel('./pretrained/resnet50_knowledge_distilltion_vgg11_pytorchmodel_cifar10/model.pth', 10))
tocsv(y_train_scores_1, 1, './results/cifar10/resnet50')
tocsv(y_train_scores_2, 2, './results/cifar10/resnet50')
tocsv(y_train_scores_3, 3, './results/cifar10/resnet50')
tocsv(y_train_scores_4, 4, './results/cifar10/resnet50')
tocsv(y_train_scores_5, 5, './results/cifar10/resnet50')
tocsv(y_train_scores_6, 6, './results/cifar10/resnet50')

# svhn+resnet18
testloader = svhn_test_loader
y_train_scores_1 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_svhn_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_1, 1, './results/svhn/resnet18')
del y_train_scores_1
y_train_scores_2 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_svhn_pruned_0.1_unretrained_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_2, 2, './results/svhn/resnet18')
del y_train_scores_2
y_train_scores_3 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_svhn_pruned_0.3_unretrained_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_3, 3, './results/svhn/resnet18')
del y_train_scores_3
y_train_scores_4 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_svhn_pruned_0.5_unretrained_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_4, 4, './results/svhn/resnet18')
del y_train_scores_4
y_train_scores_5 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_svhn_finetune_svhn_unfrozen_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_5, 5, './results/svhn/resnet18')
del y_train_scores_5

y_train_scores_6 = get_train_scores(
    get_vgg11_pytorchmodel('./pretrained/resnet18_knowledge_distilltion_vgg11_pytorchmodel_svhn/model.pth', 10))
tocsv(y_train_scores_6, 6, './results/svhn/resnet18')
del y_train_scores_6

# svhn+resnet50
testloader = svhn_test_loader
y_train_scores_1 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_svhn_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_1, 1, './results/svhn/resnet50')
del y_train_scores_1
y_train_scores_2 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_svhn_pruned_0.1_unretrained_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_2, 2, './results/svhn/resnet50')
del y_train_scores_2
y_train_scores_3 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_svhn_pruned_0.3_unretrained_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_3, 3, './results/svhn/resnet50')
del y_train_scores_3
y_train_scores_4 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_svhn_pruned_0.5_unretrained_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_4, 4, './results/svhn/resnet50')
del y_train_scores_4
y_train_scores_5 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_svhn_finetune_svhn_unfrozen_pytorchmodel/model.pth', 10))
tocsv(y_train_scores_5, 5, './results/svhn/resnet50')
del y_train_scores_5
y_train_scores_6 = get_train_scores(
    get_vgg11_pytorchmodel('./pretrained/resnet50_knowledge_distilltion_vgg11_pytorchmodel_svhn/model.pth', 10))
tocsv(y_train_scores_6, 6, './results/svhn/resnet50')
del y_train_scores_6

# cifar100+resnet18
testloader = cifar100_test_loader
y_train_scores_1 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar100_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_1, 1, './results/cifar100/resnet18')
y_train_scores_2 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar100_pruned_0.1_unretrained_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_2, 2, './results/cifar100/resnet18')
y_train_scores_3 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar100_pruned_0.3_unretrained_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_3, 3, './results/cifar100/resnet18')
y_train_scores_4 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar100_pruned_0.5_unretrained_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_4, 4, './results/cifar100/resnet18')
y_train_scores_5 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_cifar100_finetune_cifar100_unfrozen_pytorchmodel/model.pth',
                               100))
tocsv(y_train_scores_5, 5, './results/cifar100/resnet18')
y_train_scores_6 = get_train_scores(
    get_vgg11_pytorchmodel('./pretrained/resnet18_knowledge_distilltion_vgg11_pytorchmodel_cifar100/model.pth', 100))
tocsv(y_train_scores_6, 6, './results/cifar100/resnet18')

# cifar100+resnet50
testloader = cifar100_test_loader
y_train_scores_1 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar100_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_1, 1, './results/cifar100/resnet50')
y_train_scores_2 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar100_pruned_0.1_unretrained_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_2, 2, './results/cifar100/resnet50')
y_train_scores_3 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar100_pruned_0.3_unretrained_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_3, 3, './results/cifar100/resnet50')
y_train_scores_4 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar100_pruned_0.5_unretrained_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_4, 4, './results/cifar100/resnet50')
y_train_scores_5 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_cifar100_finetune_cifar100_unfrozen_pytorchmodel/model.pth',
                               100))
tocsv(y_train_scores_5, 5, './results/cifar100/resnet50')
y_train_scores_6 = get_train_scores(
    get_vgg11_pytorchmodel('./pretrained/resnet50_knowledge_distilltion_vgg11_pytorchmodel_cifar100/model.pth', 100))
tocsv(y_train_scores_6, 6, './results/cifar100/resnet50')

# mini_imagenet+resnet18
testloader = mini_imagenet_test_loader
y_train_scores_1 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_mini_imagenet_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_1, 1, './results/mini_imagenet/resnet18')
y_train_scores_2 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_mini_imagenet_pruned_0.1_unretrained_pytorchmodel/model.pth',
                               100))
tocsv(y_train_scores_2, 2, './results/mini_imagenet/resnet18')
y_train_scores_3 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_mini_imagenet_pruned_0.3_unretrained_pytorchmodel/model.pth',
                               100))
tocsv(y_train_scores_3, 3, './results/mini_imagenet/resnet18')
y_train_scores_4 = get_train_scores(
    get_resnet_18_pytorchmodel('./pretrained/resnet_18_mini_imagenet_pruned_0.5_unretrained_pytorchmodel/model.pth',
                               100))
tocsv(y_train_scores_4, 4, './results/mini_imagenet/resnet18')
y_train_scores_5 = get_train_scores(
    get_resnet_18_pytorchmodel(
        './pretrained/resnet_18_mini_imagenet_finetune_mini_imagenet_unfrozen_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_5, 5, './results/mini_imagenet/resnet18')
y_train_scores_6 = get_train_scores(
    get_vgg11_pytorchmodel('./pretrained/resnet18_knowledge_distilltion_vgg11_pytorchmodel_mini_imagenet/model.pth',
                           100))
tocsv(y_train_scores_6, 6, './results/mini_imagenet/resnet18')

# mini_imagenet+resnet50
testloader = mini_imagenet_test_loader
y_train_scores_1 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_mini_imagenet_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_1, 1, './results/mini_imagenet/resnet50')
y_train_scores_2 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_mini_imagenet_pruned_0.1_unretrained_pytorchmodel/model.pth',
                               100))
tocsv(y_train_scores_2, 2, './results/mini_imagenet/resnet50')
y_train_scores_3 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_mini_imagenet_pruned_0.3_unretrained_pytorchmodel/model.pth',
                               100))
tocsv(y_train_scores_3, 3, './results/mini_imagenet/resnet50')
y_train_scores_4 = get_train_scores(
    get_resnet_50_pytorchmodel('./pretrained/resnet_50_mini_imagenet_pruned_0.5_unretrained_pytorchmodel/model.pth',
                               100))
tocsv(y_train_scores_4, 4, './results/mini_imagenet/resnet50')
y_train_scores_5 = get_train_scores(
    get_resnet_50_pytorchmodel(
        './pretrained/resnet_50_mini_imagenet_finetune_mini_imagenet_unfrozen_pytorchmodel/model.pth', 100))
tocsv(y_train_scores_5, 5, './results/mini_imagenet/resnet50')
y_train_scores_6 = get_train_scores(
    get_vgg11_pytorchmodel('./pretrained/resnet50_knowledge_distilltion_vgg11_pytorchmodel_mini_imagenet/model.pth',
                           100))
tocsv(y_train_scores_6, 6, './results/mini_imagenet/resnet50')

# googlenet
testloader = cifar10_test_loader
y_train_scores_1 = get_train_scores(
    get_googlenet_pytorchmodel('./pretrained/googlenet_imagenet_finetune_cifar10_pytorchmodel_unfrozen/model.pth', 10))
tocsv(y_train_scores_1, 1, './results/cifar10/googlenet')
testloader = cifar100_test_loader
y_train_scores_2 = get_train_scores(
    get_googlenet_pytorchmodel('./pretrained/googlenet_imagenet_finetune_cifar100_pytorchmodel_unfrozen/model.pth',
                               100))
tocsv(y_train_scores_2, 2, './results/cifar100/googlenet')
testloader = svhn_test_loader
y_train_scores_3 = get_train_scores(
    get_googlenet_pytorchmodel('./pretrained/googlenet_imagenet_finetune_svhn_pytorchmodel_unfrozen/model.pth', 10))
tocsv(y_train_scores_3, 3, './results/svhn/googlenet')
testloader = mini_imagenet_test_loader
y_train_scores_4 = get_train_scores(
    get_googlenet_pytorchmodel('./pretrained/googlenet_imagenet_finetune_mini_imagenet_pytorchmodel_unfrozen/model.pth',
                               100))
tocsv(y_train_scores_4, 4, './results/mini_imagenet/googlenet')

# end COF
