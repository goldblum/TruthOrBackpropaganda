'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG11-1': [64 * 1, 'M', 128 * 1, 'M', 256 * 1 , 256 * 1, 'M', 512 * 1, 512 * 1, 'M', 512 * 1, 512 * 1, 'M'],
    'VGG11-2': [64 * 2, 'M', 128 * 2, 'M', 256 * 2 , 256 * 2, 'M', 512 * 2, 512 * 2, 'M', 512 * 2, 512 * 2, 'M'],
    'VGG11-3': [64 * 3, 'M', 128 * 3, 'M', 256 * 3 , 256 * 3, 'M', 512 * 3, 512 * 3, 'M', 512 * 3, 512 * 3, 'M'],
    'VGG11-4': [64 * 4, 'M', 128 * 4, 'M', 256 * 4 , 256 * 4, 'M', 512 * 4, 512 * 4, 'M', 512 * 4, 512 * 4, 'M'],
    'VGG11-5': [64 * 5, 'M', 128 * 5, 'M', 256 * 5 , 256 * 5, 'M', 512 * 5, 512 * 5, 'M', 512 * 5, 512 * 5, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(cfg[vgg_name][-2], 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())

# test()
