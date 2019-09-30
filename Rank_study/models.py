############################################################
#
# models.py
# model class definitions for rank study
# 
# August 2019
#
############################################################

import torch.nn as nn
import torch.nn.functional as F


class FullyConnectedBlockNonlinear(nn.Module):
    def __init__(self, width):
        super(FullyConnectedBlockNonlinear, self).__init__()
        self.linear = nn.Linear(width, width)
        self.bn = nn.BatchNorm1d(width)

    def forward(self, x):
        return F.relu(self.bn(self.linear(x)))


class FullyConnectedBlockNonlinearNoBN(nn.Module):
    def __init__(self, width):
        super(FullyConnectedBlockNonlinearNoBN, self).__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, x):
        return F.relu(self.linear(x))


class Linear(nn.Module):
    def __init__(self, channels=3, image_dim=32, width=100, depth=3, num_classes=10):
        super(Linear, self).__init__()
        self.depth = depth
        self.linear_first = nn.Linear(image_dim*image_dim*channels, width)
        self.bn_first = nn.BatchNorm1d(width)
        self.layers = self._make_layers(width, depth-2)
        self.linear_last = nn.Linear(width, num_classes)

    def _make_layers(self, width, depth):
        layers = []
        for i in range(depth):
            layers.append(FullyConnectedBlockNonlinear(width))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.bn_first(self.linear_first(out)))
        out = self.layers(out)
        out = self.linear_last(out)
        return out


class LinearNoBN(nn.Module):
    def __init__(self, channels=3, image_dim=32, width=100, depth=3, num_classes=10):
        super(LinearNoBN, self).__init__()
        self.depth = depth
        self.linear_first = nn.Linear(image_dim*image_dim*channels, width)
        self.layers = self._make_layers(width, depth-2)
        self.linear_last = nn.Linear(width, num_classes)

    def _make_layers(self, width, depth):
        layers = []
        for i in range(depth):
            layers.append(FullyConnectedBlockNonlinearNoBN(width))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.linear_first(out))
        out = self.layers(out)
        out = self.linear_last(out)
        return out


class LinearSingleLayer(nn.Module):
    def __init__(self, norm='None', channels=3, image_dim=32, num_classes=10):
        super(LinearSingleLayer, self).__init__()
        self.norm = norm
        if norm == 'BN':
            self.linear_layer = nn.Linear(image_dim*image_dim*channels, num_classes, bias=False)
            self.bn = nn.BatchNorm1d(num_classes)
        else:
            self.linear_layer = nn.Linear(image_dim*image_dim*channels, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.linear_layer(out)
        if self.norm == 'BN':
            out = self.bn(out)
        return out


class ConvBlockNonlinearBN(nn.Module):
    def __init__(self, num_filters):
        super(ConvBlockNonlinearBN, self).__init__()
        self.conv = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ConvBlockNonlinearNoBN(nn.Module):
    def __init__(self, num_filters):
        super(ConvBlockNonlinearNoBN, self).__init__()
        self.conv = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        return F.relu(self.conv(x))


class CNN(nn.Module):
    def __init__(self, block, channels=3, image_dim=32, num_classes=10, depth=8, num_filters=64):
        super(CNN, self).__init__()
        self.block = block
        self.conv_first = nn.Conv2d(channels, num_filters, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_first = nn.BatchNorm2d(num_filters)
        self.layers = self._make_layer(block, num_filters, depth-2)
        self.conv_last = nn.Conv2d(num_filters, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn_last = nn.BatchNorm2d(1)
        self.linear = nn.Linear(1*image_dim*image_dim, num_classes)

    def _make_layer(self, block, num_filters, depth):
        layers = []
        for i in range(depth):
            layers.append(block(num_filters))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_first(x)
        if self.block == ConvBlockNonlinearBN:
            out = self.bn_first(out)
        out = F.relu(out)
        out = self.layers(out)
        out = self.conv_last(out)
        if self.block == ConvBlockNonlinearBN:
            out = self.bn_last(out)
        out = F.relu(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def single_layer_no_bn(channels=1, image_dim=28, num_classes=10):
    return LinearSingleLayer(channels=channels, image_dim=image_dim, num_classes=num_classes)


def single_layer_bn(channels=1, image_dim=28, num_classes=10):
    return LinearSingleLayer(norm='BN', channels=channels, image_dim=image_dim, num_classes=num_classes)


def mlp(depth, width, channels=1, image_dim=28, num_classes=10):
    return Linear(channels=channels, image_dim=image_dim, width=width, depth=depth, num_classes=num_classes)


def mlp_no_bn(depth, width, channels=1, image_dim=28, num_classes=10):
    return LinearNoBN(channels=channels, image_dim=image_dim, width=width, depth=depth, num_classes=num_classes)


def cnn(channels=3, image_dim=32, num_classes=10, depth=5, num_filters=16):
    return CNN(block=ConvBlockNonlinearBN, channels=channels, image_dim=image_dim, num_classes=num_classes,
               depth=depth, num_filters=num_filters)


def cnn_no_bn(channels=3, image_dim=32, num_classes=10, depth=5, num_filters=16):
    return CNN(block=ConvBlockNonlinearNoBN, channels=channels, image_dim=image_dim, num_classes=num_classes,
               depth=depth, num_filters=num_filters)
