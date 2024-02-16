import numbers
import random

import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image, ImageOps

from params import *
from dataset import TSNDataSet
from gcn_i3d import pretrained_gcn_i3d
from engine import GCNMultiLabelMAPEngine
import datasets_video


class GroupMultiScaleCrop(object):
    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):

        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                         for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))
        ret.append((4 * w_step, 0))
        ret.append((0, 4 * h_step))
        ret.append((4 * w_step, 4 * h_step))
        ret.append((2 * w_step, 2 * h_step))

        if more_fix_crop:
            ret.append((0, 2 * h_step))
            ret.append((4 * w_step, 2 * h_step))
            ret.append((2 * w_step, 4 * h_step))
            ret.append((2 * w_step, 0 * h_step))

            ret.append((1 * w_step, 1 * h_step))
            ret.append((3 * w_step, 1 * h_step))
            ret.append((1 * w_step, 3 * h_step))
            ret.append((3 * w_step, 3 * h_step))

        return ret
    
    
class GroupRandomHorizontalFlip(object):
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group
        

class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)
            

class ToTorchFormatTensor(object):
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()
    

class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        rep_std = self.std * (tensor.size()[0]//len(self.std))

        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


class ChangeToCTHW(object):
    def __init__(self, modality='RGB'):
        self.modality = modality

    def __call__(self, pic):
        if self.modality == 'RGB':
            channel = 3
        elif self.modality == 'flow':
            channel = 2
        else:
            print('modality:{} is not designed!'.format(self.modality))

        reshaped_pic = pic.view(-1, channel, pic.shape[-2], pic.shape[-1])
        reshaped_pic = reshaped_pic.transpose(0, 1).contiguous()

        return reshaped_pic


class GroupScale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Scale(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


def get_config_optim(model, lr, weight_decay):
    params_dict = dict(model.named_parameters())
    params = []

    for key, value in params_dict.items():
        decay_mult = 0.0 if 'bias' in key or 'adj' == key else 1.0

        if key.startswith('conv1') or key.startswith('bn1'):
            lr_mult = 0.1
        elif 'fc' in key:
            lr_mult = 1.0
        elif 'adj' == key:
            lr_mult = 0.0
        elif 'gc' in key:
            lr_mult = 1.0
        else:
            lr_mult = 0.1

        params.append({'params': value,
                    'lr': lr,
                    'lr_mult': lr_mult,
                    'weight_decay': weight_decay,
                    'decay_mult': decay_mult})

    return params

    
def main():
    device = (
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'>>> Using {device} device <<<')

    print('>>> Loading datasets <<<')
    categories, train_list, val_list, train_num_list, val_num_list, root_path, prefix = \
        datasets_video.return_dataset(dataset, modality, path)

    num_class = len(categories)
    train_dataset = TSNDataSet(root_path, train_list, train_num_list,
                               num_class=num_class, num_segments=num_segments,
                               new_length=data_length, modality=modality, image_tmpl=prefix,
                               transform=torchvision.transforms.Compose([
                                  GroupMultiScaleCrop(crop_size, [1.0, 0.875, 0.75, 0.66, 0.5], max_distort=2),
                                  GroupRandomHorizontalFlip(is_flow=False),
                                  Stack(roll=False),
                                  ToTorchFormatTensor(div=True),
                                  GroupNormalize(input_mean, input_std),
                                  ChangeToCTHW(modality=modality)
                              ]))

    val_dataset = TSNDataSet(root_path, val_list, val_num_list,
                             num_class=num_class, num_segments=num_segments,
                             new_length=data_length, modality=modality,
                             image_tmpl=prefix, random_shift=False,
                             transform=torchvision.transforms.Compose([
                                GroupScale(int(scale_size)),
                                GroupCenterCrop(crop_size),
                                Stack(roll=False),
                                ToTorchFormatTensor(div=True),
                                GroupNormalize(input_mean, input_std),
                                ChangeToCTHW(modality=modality)
                            ]))

    print('>>> Building model <<<')
    gcn_i3d = pretrained_gcn_i3d(num_class=num_class, t=0.4,
                                 adj_file='./data/Charades_v1/gcn_info/class_graph_conceptnet_context_0.8.pkl',
                                 word_file='./data/Charades_v1/gcn_info/class_word.pkl')
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(get_config_optim(gcn_i3d, learning_rate, weight_decay))

    state = {        
        'batch_size': batch_size,
        'val_batch_size': val_batch_size,
        'image_size': image_size,
        'max_epochs': epochs,
        'evaluate': evaluate,
        'resume': resume,
        'num_classes': num_class,
        'difficult_examples': False,
        'print_freq': print_freq,
        'save_model_path': save_model_path,
        'log_path': log_path,
        'logname': logname,
        'workers': workers,
        'epoch_step': epoch_step,
        'lr': learning_rate,
        'device_ids': list(range(torch.cuda.device_count())),
    }

    mapengine = GCNMultiLabelMAPEngine(state, inp_file='./data/Charades_v1/gcn_info/class_word.pkl')
    mapengine.learning(gcn_i3d, criterion, train_dataset, val_dataset, optimizer)


if __name__ == '__main__':
    main()