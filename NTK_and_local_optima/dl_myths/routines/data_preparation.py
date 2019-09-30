"""Repeatable code parts."""


import torch
import torchvision.transforms as transforms
import torchvision


def get_loaders(dataset='CIFAR10', batch_size=64, augmentations=False, shuffle=True, normalize=True):
    """Return a dataloader with given dataset and augmentation, normalize data."""
    # Compute mean, std:
    if dataset == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='~/data', train=True,
                                                 download=True, transform=transforms.ToTensor())
        cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
        data_mean = torch.mean(cc, dim=1).tolist()
        data_std = torch.std(cc, dim=1).tolist()
    elif dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='~/data', train=True,
                                                download=True, transform=transforms.ToTensor())
        cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
        data_mean = torch.mean(cc, dim=1).tolist()
        data_std = torch.std(cc, dim=1).tolist()
    elif dataset == 'MNIST':
        trainset = torchvision.datasets.MNIST(root='~/data', train=True,
                                              download=True, transform=transforms.ToTensor())
        cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
        data_mean = (torch.mean(cc, dim=0).item(),)
        data_std = (torch.std(cc, dim=0).item(),)

    print(f'Data mean is {data_mean}, data std is  {data_std}') if normalize else print('Normalization disabled.')
    # Setup data
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])

    if dataset == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='~/data', train=True,
                                                 download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='~/data', train=False, download=True, transform=transform_test)
    elif dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='~/data', train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='~/data', train=False, download=True, transform=transform_test)
    elif dataset == 'MNIST':
        trainset = torchvision.datasets.MNIST(root='~/data', train=True,
                                              download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root='~/data', train=False, download=True, transform=transform_test)

    num_workers = torch.get_num_threads() if torch.get_num_threads() > 0 else 0
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=min(batch_size, len(trainset)),
                                              shuffle=shuffle, drop_last=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=min(batch_size, len(testset)),
                                             shuffle=False, drop_last=True, num_workers=num_workers)

    return trainloader, testloader


class TensorDatasetAugmented(torch.utils.data.Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.

    """

    def __init__(self, *tensors, crop=False, flip=False, padding=4):
        """Init :>."""
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

        self.padding = padding
        self.crop = crop
        self.flip = flip
        self.shape = self.tensors[0].shape
        self.pad = torch.nn.ZeroPad2d(self.padding)

    def __getitem__(self, index):
        """As normal :> ."""
        # crop:
        if self.crop:
            idx = torch.randint(0, self.padding * 2, [2])
            out = self.pad(self.tensors[0][index])
            out = out[:, idx[0]:idx[0] + 32, idx[1]:idx[1] + 32]
        else:
            out = self.tensors[0][index]
        # flip
        if torch.rand(1) > 0.5 and self.flip:
            out = torch.flip(out, (1,))
        if torch.rand(1) > 0.5 and self.flip:
            out = torch.flip(out, (2,))
        return (out, self.tensors[1][index])

    def __len__(self):
        """Len of dataset."""
        return self.tensors[0].size(0)
