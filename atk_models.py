import torch
import torch.nn as nn
import torch.nn.functional as F

# GEN_CLASSES = {}
# DIS_CLASSES = {}
# def _register(name, obj, dct):
#     dct[name] = obj
#
# def register_gen(cls):
#     _register(cls.__name__, cls, GEN_CLASSES)
#     return cls
#
# def register_dis(cls):
#     _register(cls.__name__, cls, DIS_CLASSES)
#     return cls

# @register_dis
def load_liner_net(dims, _activation, _batchnorm, mode='Enc'): # 'Enc', 'Dec', 'Dis'
    layers = []
    for i in range(len(dims) - 1):
        layers.append(
            nn.Linear(dims[i], dims[i + 1]))
        if i < (len(dims) - 1) - 1 or mode == 'Enc':
            if _batchnorm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            if _activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif _activation == 'leakyrelu':
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif _activation == 'tanh':
                layers.append(nn.Tanh())
            elif _activation == 'relu':
                layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % _activation)

    # 对于解码器，添加一个tanh
    if mode == 'Dec':
        layers.append(nn.Tanh())
    elif mode == 'Dis':
        layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)

class ClippingLayer(nn.Module):
    def __init__(self, min_value, max_value):
        super(ClippingLayer, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, input):
        return torch.clamp(input, self.min_value, self.max_value)


class noisymnist_dis(nn.Module):
    def __init__(self):
        super(noisymnist_dis, self).__init__()
        # MNIST: 1*28*28
        self.models = []
        for i in range(2):
            model = [
                nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=0, bias=True),
                nn.LeakyReLU(0.2),
                # 8*13*13
                nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.2),

                # 16*5*5
                nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 1, 1),
                nn.Sigmoid()
                # 32*1*1
            ]
            self.models.append(nn.Sequential(*model))
        self.models = nn.ModuleList(self.models)
    def forward(self, views):
        outputs = []
        for i, x in enumerate(views):
            outputs.append(self.models[i](x).squeeze())

        return outputs

class noisymnist_gen(nn.Module):
    def __init__(self,
                 ):
        super(noisymnist_gen, self).__init__()

        self.encoders = []
        self.bottle_necks = []
        self.decoders = []
        for i in range(2):
            encoder_lis = [
                # MNIST:1*28*28
                nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0, bias=True),
                nn.InstanceNorm2d(8),
                nn.ReLU(),
                # 8*26*26
                nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
                nn.InstanceNorm2d(16),
                nn.ReLU(),
                # 16*12*12
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
                nn.InstanceNorm2d(32),
                nn.ReLU(),
                # 32*5*5
            ]

            # bottle_neck_lis = [ResnetBlock(32),
            #                ResnetBlock(32),
            #                ResnetBlock(32),
            #                ResnetBlock(32),]

            decoder_lis = [
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
                nn.InstanceNorm2d(16),
                nn.ReLU(),
                # state size. 16 x 11 x 11
                nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
                nn.InstanceNorm2d(8),
                nn.ReLU(),
                # state size. 8 x 23 x 23
                nn.ConvTranspose2d(8, 1, kernel_size=6, stride=1, padding=0, bias=False),
                nn.Tanh()
                # state size. image_nc x 28 x 28
            ]

            self.encoders.append(nn.Sequential(*encoder_lis))
            # self.bottle_necks.append(nn.Sequential(*bottle_neck_lis))
            self.decoders.append(nn.Sequential(*decoder_lis))

        self.encoders = nn.ModuleList(self.encoders)
        # self.bottle_necks = nn.ModuleList(self.bottle_necks)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, views):
        views_ = []
        for i, x in enumerate(views):
            x = self.encoders[i](x)
            # x = self.bottle_necks[i](x)
            x = self.decoders[i](x)
            views_.append(x)

        return views_

noisyfashionmnist_dis = noisymnist_dis
noisyfashionmnist_gen = noisymnist_gen

edgefashionmnist_dis = noisymnist_dis
edgefashionmnist_gen = noisymnist_gen

class patchedmnist_dis(nn.Module):
    def __init__(self):
        super(patchedmnist_dis, self).__init__()
        # MNIST: 1*28*28
        self.models = []
        for i in range(3):
            model = [
                nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=0, bias=True),
                nn.LeakyReLU(0.2),
                # 8*13*13
                nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.2),

                # 16*5*5
                nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),

                nn.Conv2d(32, 1, 1),
                nn.Sigmoid()
                # 32*1*1
            ]
            self.models.append(nn.Sequential(*model))
        self.models = nn.ModuleList(self.models)
    def forward(self, views):
        outputs = []
        for i, x in enumerate(views):
            outputs.append(self.models[i](x).squeeze())

        return outputs

# @register_gen
class patchedmnist_gen(nn.Module):
    def __init__(self
                 ):
        super(patchedmnist_gen, self).__init__()

        self.encoders = []
        self.bottle_necks = []
        self.decoders = []
        for i in range(3):
            encoder_lis = [
                # MNIST:1*28*28
                nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0, bias=True),
                nn.InstanceNorm2d(8),
                nn.ReLU(),
                # 8*26*26
                nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
                nn.InstanceNorm2d(16),
                nn.ReLU(),
                # 16*12*12
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
                nn.InstanceNorm2d(32),
                nn.ReLU(),
                # 32*5*5
            ]

            # bottle_neck_lis = [ResnetBlock(32),
            #                ResnetBlock(32),
            #                ResnetBlock(32),
            #                ResnetBlock(32),]

            decoder_lis = [
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
                nn.InstanceNorm2d(16),
                nn.ReLU(),
                # state size. 16 x 11 x 11
                nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
                nn.InstanceNorm2d(8),
                nn.ReLU(),
                # state size. 8 x 23 x 23
                nn.ConvTranspose2d(8, 1, kernel_size=6, stride=1, padding=0, bias=False),
                nn.Tanh()
                # state size. image_nc x 28 x 28
            ]

            self.encoders.append(nn.Sequential(*encoder_lis))
            # self.bottle_necks.append(nn.Sequential(*bottle_neck_lis))
            self.decoders.append(nn.Sequential(*decoder_lis))

        self.encoders = nn.ModuleList(self.encoders)
        # self.bottle_necks = nn.ModuleList(self.bottle_necks)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, views):
        views_ = []
        for i, x in enumerate(views):
            x = self.encoders[i](x)
            # x = self.bottle_necks[i](x)
            x = self.decoders[i](x)
            views_.append(x)

        return views_



class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out