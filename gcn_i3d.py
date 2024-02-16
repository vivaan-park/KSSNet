import os
import math
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter

class Unit3D(nn.Module):
    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):

        super().__init__()
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0,
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        (_, _, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super().__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name+'/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0,b1,b2,b3], dim=1)

class InceptionI3d(nn.Module):
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super().__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3, 3, 3), name=name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point:
            return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                            name=name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                            name=name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Logits'
        self.avg_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.build()

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)  # use _modules to work with dataparallel

        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
        else:
            logits = x
        # logits is batch X time X classes, which is what we want to work with
        logits = logits.view(logits.size(0), -1)
        return logits

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        middle_features = max(32, int((in_features + out_features) / 8))
        self.weight1 = Parameter(torch.Tensor(in_features, middle_features))
        self.weight2 = Parameter(torch.Tensor(middle_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight1)
        support = torch.matmul(support, self.weight2)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            output = output + self.bias

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


def merge_gcn_residual(feature, x, merge_conv):
    feature_raw = feature
    feature = feature_raw.transpose(1, 2)
    feature = feature.transpose(2, 3)
    feature = feature.transpose(3, 4).contiguous()
    feature = feature.view(-1, feature.shape[-1])
    reshape_x = x.transpose(0, 1)
    feature = torch.matmul(feature, reshape_x)
    feature = feature.view(feature_raw.shape[0], feature_raw.shape[2], feature_raw.shape[3], feature_raw.shape[4], -1)
    feature = feature.transpose(3, 4)
    feature = feature.transpose(2, 3)
    feature = feature.transpose(1, 2)

    feature = merge_conv(feature)

    return feature_raw + feature


class GCNI3D(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0, adj_file=None, word_file=''):
        super().__init__()
        self.features = nn.Sequential(
            model._modules['Conv3d_1a_7x7'],
            model._modules['MaxPool3d_2a_3x3'],
            model._modules['Conv3d_2b_1x1'],
            model._modules['Conv3d_2c_3x3'],
            model._modules['MaxPool3d_3a_3x3'],
            model._modules['Mixed_3b'],
            model._modules['Mixed_3c'],
            model._modules['MaxPool3d_4a_3x3'],
            model._modules['Mixed_4b'],
            model._modules['Mixed_4c'],
            model._modules['Mixed_4d'],
            model._modules['Mixed_4e'],
            model._modules['Mixed_4f'],
            model._modules['MaxPool3d_5a_2x2'],
            model._modules['Mixed_5b'],
            model._modules['Mixed_5c']
        )

        self.num_classes = num_classes
        self.avg_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.dropout = nn.Dropout3d(0.5)

        self.gc1 = GraphConvolution(in_channel, 192, bias=True)
        self.gc2 = GraphConvolution(192, 480, bias=True)
        self.gc3 = GraphConvolution(480, 832, bias=True)
        self.gc4 = GraphConvolution(832, 1024, bias=True)
        self.relu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

        with open(adj_file, 'rb') as point:
            _adj = pickle.load(point)
        _adj[_adj < 0.03] = 0
        neighbor_rate = 0.4
        _adj = neighbor_rate * _adj + (1 - neighbor_rate) * np.eye(self.num_classes)
        D = _adj.sum(0)
        half_D = D ** (-0.5)
        half_D = np.diag(half_D)
        _adj_norm = np.matmul(np.matmul(half_D, _adj), half_D)
        self.adj = torch.from_numpy(_adj_norm).float()
        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

        if not os.path.exists(word_file):
            word = np.random.randn(num_classes, 300)
            print('graph input: random')
        else:
            with open(word_file, 'rb') as point:
                word = pickle.load(point)
                print('graph input: loaded from {}'.format(word_file))
        self.word = torch.from_numpy(word).float()

        self.merge_conv1 = nn.Conv3d(num_classes, 192, kernel_size=1, stride=1, bias=False)
        self.merge_conv2 = nn.Conv3d(num_classes, 480, kernel_size=1, stride=1, bias=False)
        self.merge_conv3 = nn.Conv3d(num_classes, 832, kernel_size=1, stride=1, bias=False)

        self.logits = Unit3D(in_channels=1024, output_channels=self.num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

    def forward(self, feature):
        adj = self.adj.cuda().detach()
        word = self.word.cuda().detach()

        feature = self.features[0](feature)  # [2, 64, 32, 112, 112]
        feature = self.features[1](feature)  # [2, 64, 32, 56, 56]
        feature = self.features[2](feature)  # [2, 64, 32, 56, 56]
        feature = self.features[3](feature)  # [2, 192, 32, 56, 56]

        x_raw = self.gc1(word, adj)
        x = self.tanh(x_raw)
        feature = merge_gcn_residual(feature, x, self.merge_conv1)

        feature = self.features[4](feature)  # [2, 192, 32, 28, 28]
        feature = self.features[5](feature)  # [2, 256, 32, 28, 28]
        feature = self.features[6](feature)  # [2, 480, 32, 28, 28]

        x = self.relu(x_raw)
        x_raw = self.gc2(x, adj)
        x = self.tanh(x_raw)
        feature = merge_gcn_residual(feature, x, self.merge_conv2)

        feature = self.features[7](feature)  # [2, 480, 16, 14, 14]
        feature = self.features[8](feature)  # [2, 512, 16, 14, 14]
        feature = self.features[9](feature)  # [2, 512, 16, 14, 14]
        feature = self.features[10](feature)  # [2, 512, 16, 14, 14]
        feature = self.features[11](feature)  # [2, 528, 16, 14, 14]
        feature = self.features[12](feature)  # [2, 832, 16, 14, 14]

        x = self.relu(x_raw)
        x_raw = self.gc3(x, adj)
        x = self.tanh(x_raw)
        feature = merge_gcn_residual(feature, x, self.merge_conv3)

        feature = self.features[13](feature)  # [2, 832, 8, 7, 7]
        feature = self.features[14](feature)  # [2, 832, 8, 7, 7]
        feature = self.features[15](feature)  # [2, 1024, 8, 7, 7]

        feature_raw = self.avg_pool(feature)
        feature_raw = self.dropout(feature_raw)
        feature = feature_raw.view(feature_raw.size(0), -1)  # [2, 1024]

        x = self.relu(x_raw)
        x = self.gc4(x, adj)
        x = self.tanh(x)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        y = self.logits(feature_raw)
        y = y.view(y.size(0), -1)
        x = x + y
        return x


def pretrained_gcn_i3d(num_class, t, pretrained=True, adj_file=None, word_file='', in_channel=300):
    model = InceptionI3d(num_class, in_channels = 3)
    if pretrained:
        pretrained_path = './pretrained_model/i3d_imagenet.pth'
        pretrained_model = torch.load(pretrained_path)
        if 'state_dict' in pretrained_model:
            pretrained_model = pretrained_model['state_dict']
        total_num = len(pretrained_model)
        model_state = model.state_dict()
        load_parameters = {key: value for key, value in pretrained_model.items()
            if key in model_state and value.shape == model_state[key].shape}
        load_num = len(load_parameters)
        model_state.update(load_parameters)
        model.load_state_dict(model_state)

    return GCNI3D(model, num_class, t=t, adj_file=adj_file, in_channel=in_channel, word_file=word_file)
