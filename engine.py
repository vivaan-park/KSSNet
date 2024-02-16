import os
import shutil
import pickle

import torch
import torch.nn as nn
import torchnet
import numpy as np


class Engine(object):
    def __init__(self, state={}):
        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('image_size') is None:
            self.state['image_size'] = 224

        if self._state('batch_size') is None:
            self.state['batch_size'] = 64

        if self._state('workers') is None:
            self.state['workers'] = 25

        if self._state('device_ids') is None:
            self.state['device_ids'] = None

        if self._state('evaluate') is None:
            self.state['evaluate'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 90

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []

        # meters
        self.state['meter_loss'] = torchnet.meter.AverageValueMeter()
        # time measure
        self.state['batch_time'] = torchnet.meter.AverageValueMeter()
        self.state['data_time'] = torchnet.meter.AverageValueMeter()
        # display parameters
        if self._state('use_pb') is None:
            self.state['use_pb'] = True
        if self._state('print_freq') is None:
            self.state['print_freq'] = 0

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        loss = self.state['meter_loss'].value()[0]
        if display:
            if training:
                info = 'Epoch: [{0}]\t' \
                       'Loss {loss:.4f}'.format(self.state['epoch'], loss=loss)
                print(info)
                with open(os.path.join(self.state['log_path'], self.state['logname']), 'a') as f:
                    f.write(info + '\n')
            else:
                info = 'Test: \t Loss {loss:.4f}'.format(loss=loss)
                print(info)
                with open(os.path.join(self.state['log_path'], self.state['logname']), 'a') as f:
                    f.write(info + '\n')
        return loss

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        # record loss
        self.state['loss_batch'] = self.state['loss'].cpu().data
        self.state['meter_loss'].add(self.state['loss_batch'])

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                info = 'Epoch: [{0}][{1}/{2}]\t' \
                       'Time {batch_time_current:.3f} ({batch_time:.3f})\t' \
                       'Data {data_time_current:.3f} ({data_time:.3f})\t' \
                       'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss)

                print(info)
                with open(os.path.join(self.state['log_path'], self.state['logname']), 'a') as f:
                    f.write(info + '\n')
            else:
                info = 'Test: [{0}/{1}]\t' \
                       'Time {batch_time_current:.3f} ({batch_time:.3f})\t' \
                       'Data {data_time_current:.3f} ({data_time:.3f})\t' \
                       'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss)

                print(info)
                with open(os.path.join(self.state['log_path'], self.state['logname']), 'a') as f:
                    f.write(info + '\n')

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):

        input_var = torch.autograd.Variable(self.state['input'])
        target_var = torch.autograd.Variable(self.state['target'])

        if not training:
            input_var.volatile = True
            target_var.volatile = True

        # compute output
        self.state['output'] = model(input_var)

        self.state['loss'] = criterion(self.state['output'], target_var)

        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            optimizer.step()

    def init_learning(self, model, criterion):

        self.state['best_score'] = 0

    def learning(self, model, criterion, train_dataset, val_dataset, optimizer=None):

        self.init_learning(model, criterion)

        # data loading code
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=max(self.state['batch_size'], 1), shuffle=True,
                                                   num_workers=self.state['workers'])

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=max(self.state['val_batch_size'], 1), shuffle=False,
                                                 num_workers=self.state['workers'])

        # optionally resume from a checkpoint
        if self._state('resume') is not None:
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                checkpoint = torch.load(self.state['resume'])
                self.state['start_epoch'] = checkpoint['epoch']
                self.state['best_score'] = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.state['evaluate'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))

        if self.state['use_gpu']:
            train_loader.pin_memory = True
            val_loader.pin_memory = True

            model = torch.nn.DataParallel(model, device_ids=self.state['device_ids']).cuda()
            criterion = criterion.cuda()

        if self.state['evaluate']:
            self.validate(val_loader, model, criterion)
            return

        # define optimizer
        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            lr = self.adjust_learning_rate(optimizer)
            print('lr:{:.5f}'.format(lr))

            # train for one epoch
            self.train(train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set
            # prec1 = self.validate(val_loader, model, criterion)
            prec1 = 1.0
            # remember best prec@1 and save checkpoint
            is_best = prec1 > self.state['best_score']
            self.state['best_score'] = max(prec1, self.state['best_score'])
            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self._state('arch'),
                'state_dict': model.module.state_dict() if self.state['use_gpu'] else model.state_dict(),
                'best_score': self.state['best_score'],
            }, is_best)

            info = ' *** best={best:.3f}'.format(best=self.state['best_score'])
            print(info)
            with open(os.path.join(self.state['log_path'], 'train_charades.log'), 'a') as f:
                f.write(info + '\n')

            if epoch == 4:
                exit()
        return self.state['best_score']

    def train(self, data_loader, model, criterion, optimizer, epoch):
        model.train()

        self.on_start_epoch(True, model, criterion, data_loader, optimizer)

        for i, (input, target) in enumerate(data_loader):
            self.state['iteration'] = i

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(True, model, criterion, data_loader, optimizer)
            self.on_forward(True, model, criterion, data_loader, optimizer)
            self.on_end_batch(True, model, criterion, data_loader, optimizer)

        self.on_end_epoch(True, model, criterion, data_loader, optimizer)

    def validate(self, data_loader, model, criterion):
        model.eval()

        self.on_start_epoch(False, model, criterion, data_loader)

        for i, (input, target) in enumerate(data_loader):
            self.state['iteration'] = i

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(False, model, criterion, data_loader)
            self.on_forward(False, model, criterion, data_loader)
            self.on_end_batch(False, model, criterion, data_loader)

        score = self.on_end_epoch(False, model, criterion, data_loader)

        return score

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if self._state('save_model_path') is not None:
            filename_ = filename
            filename = os.path.join(self.state['save_model_path'], filename_)
            if not os.path.exists(self.state['save_model_path']):
                os.makedirs(self.state['save_model_path'])
        print('save model {filename}'.format(filename=filename))
        torch.save(state, filename)
        if is_best:
            filename_best = 'model_best.pth.tar'
            if self._state('save_model_path') is not None:
                filename_best = os.path.join(self.state['save_model_path'], filename_best)
            shutil.copyfile(filename, filename_best)
            if self._state('save_model_path') is not None:
                if self._state('filename_previous_best') is not None:
                    os.remove(self._state('filename_previous_best'))
                filename_best = os.path.join(self.state['save_model_path'],
                                             'model_best_{score:.4f}.pth.tar'.format(score=state['best_score']))
                shutil.copyfile(filename, filename_best)
                self.state['filename_previous_best'] = filename_best

    def adjust_learning_rate(self, optimizer):
        # lr = args.lr * (0.1 ** (epoch // 30))
        decay = 0.1 ** (sum(self.state['epoch'] >= np.array(self.state['epoch_step'])))
        lr = self.state['lr'] * decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class AveragePrecisionMeter(object):
    def __init__(self, difficult_examples=False, n_class=157):
        super(AveragePrecisionMeter, self).__init__()
        self.n_class = n_class
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        self.sample_num = 0.
        self.ap = 0
        self.Nc = np.zeros(self.n_class)
        self.Np = np.zeros(self.n_class)
        self.Ng = np.zeros(self.n_class)

        self.ap_topk = 0
        self.Nc_topk = np.zeros(self.n_class)
        self.Np_topk = np.zeros(self.n_class)
        self.Ng_topk = np.zeros(self.n_class)

    def add(self, output, target):
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if output.numel() > 0:
            assert target.size(1) == output.size(1), \
                'dimensions for output should match target.'

        for k in range(output.size(0)):
            # sort scores
            outputk = output[k, :]
            targetk = target[k, :]
            # compute average precision
            self.ap += AveragePrecisionMeter.average_precision(outputk, targetk, self.difficult_examples)
            self.sample_num += 1

        Nc = np.zeros(self.n_class)
        Np = np.zeros(self.n_class)
        Ng = np.zeros(self.n_class)
        for i in range(self.n_class):
            outputk = output[:, i]
            targetk = target[:, i]
            Ng[i] = sum(targetk == 1)
            Np[i] = sum(outputk >= 0)
            Nc[i] = sum(targetk * (outputk >= 0).type(torch.FloatTensor))

        self.Nc += Nc
        self.Np += Np
        self.Ng += Ng

        n = output.shape[0]
        output_topk = torch.zeros((n, self.n_class)) - 1
        index = output.topk(3, 1, True, True)[1].cpu().numpy()
        tmp = output.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                output_topk[i, ind] = 1 if tmp[i, ind] >= 0 else - 1
        for k_topk in range(output_topk.shape[0]):
            # sort scores
            outputk_topk = output_topk[k_topk, :]
            targetk_topk = target[k_topk, :]
            # compute average precision
            self.ap_topk += AveragePrecisionMeter.average_precision(outputk_topk, targetk_topk, self.difficult_examples)

        Nc_topk = np.zeros(self.n_class)
        Np_topk = np.zeros(self.n_class)
        Ng_topk = np.zeros(self.n_class)
        for i in range(self.n_class):
            outputk_topk = output_topk[:, i]
            targetk_topk = target[:, i]
            Ng_topk[i] = sum(targetk_topk == 1)
            Np_topk[i] = sum(outputk_topk >= 0)
            Nc_topk[i] = sum(targetk_topk * (outputk_topk >= 0).type(torch.FloatTensor))

        self.Nc_topk += Nc_topk
        self.Np_topk += Np_topk
        self.Ng_topk += Ng_topk

    def value(self):
        ap = self.ap / self.sample_num
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=False):

        # sort examples
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        _, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        precision_at_i_list = torch.zeros(int(target.sum()))
        for i in indices:
            label = target[i]
            total_count += 1
            if label == 1:
                pos_count += 1
                precision_at_i_list[int(pos_count) - 1] = pos_count / total_count

        for i in range(len(precision_at_i_list) - 1, 0, -1):
            if precision_at_i_list[i - 1] < precision_at_i_list[i]:
                precision_at_i_list[i - 1] = precision_at_i_list[i]

        precision_at_i = precision_at_i_list.mean()
        return precision_at_i

    def overall(self):
        if self.sample_num == 0:
            return 0

        OP = sum(self.Nc) / max(sum(self.Np), 1e-8)
        OR = sum(self.Nc) / max(sum(self.Ng), 1e-8)
        OF1 = (2 * OP * OR) / max(OP + OR, 1e-8)

        clip_Np = self.Np.copy()
        for i in range(len(self.Np)):
            if self.Np[i] < 1e-8:
                clip_Np[i] = 1e-8

        clip_Ng = self.Ng.copy()
        for i in range(len(self.Ng)):
            if self.Ng[i] < 1e-8:
                clip_Ng[i] = 1e-8

        CP = sum(self.Nc / clip_Np) / self.n_class
        CR = sum(self.Nc / clip_Ng) / self.n_class
        CF1 = (2 * CP * CR) / max(CP + CR, 1e-8)

        return OP, OR, OF1, CP, CR, CF1

    def overall_topk(self, k):
        '''
        k is useless
        '''
        if self.sample_num == 0:
            return 0

        OP = sum(self.Nc_topk) / max(sum(self.Np_topk), 1e-8)
        OR = sum(self.Nc_topk) / max(sum(self.Ng_topk), 1e-8)
        OF1 = (2 * OP * OR) / max(OP + OR, 1e-8)

        clip_Np_topk = self.Np_topk.copy()
        for i in range(len(self.Np_topk)):
            if self.Np_topk[i] < 1e-8:
                clip_Np_topk[i] = 1e-8

        clip_Ng_topk = self.Ng_topk.copy()
        for i in range(len(self.Ng_topk)):
            if self.Ng_topk[i] < 1e-8:
                clip_Ng_topk[i] = 1e-8

        CP = sum(self.Nc_topk / clip_Np_topk) / self.n_class
        CR = sum(self.Nc_topk / clip_Ng_topk) / self.n_class
        CF1 = (2 * CP * CR) / max(CP + CR, 1e-8)

        return OP, OR, OF1, CP, CR, CF1


class MultiLabelMAPEngine(Engine):
    def __init__(self, state):
        super(MultiLabelMAPEngine, self).__init__(state)
        if self._state('difficult_examples') is None:
            self.state['difficult_examples'] = False
        self.state['ap_meter'] = AveragePrecisionMeter(self.state['difficult_examples'])

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        super(MultiLabelMAPEngine, self).on_start_epoch(training, model, criterion, data_loader, optimizer)
        self.state['ap_meter'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        map = 100 * self.state['ap_meter'].value().mean()
        loss = self.state['meter_loss'].value()[0]
        if display:
            if training:
                info = 'Epoch: [{0}]\t' \
                       'Loss {loss:.4f}\t' \
                       'mAP {map:.3f}'.format(self.state['epoch'], loss=loss, map=map)
                print(info)
                with open(os.path.join(self.state['log_path'], self.state['logname']), 'a') as f:
                    f.write(info + '\n')

            else:
                info = 'Test: \t Loss {loss:.4f}\t mAP {map:.3f}'.format(loss=loss, map=map)
                print(info)
                with open(os.path.join(self.state['log_path'], self.state['logname']), 'a') as f:
                    f.write(info + '\n')

        return map

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        self.state['target_gt'] = self.state['target'].clone()

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        Engine.on_end_batch(self, training, model, criterion, data_loader, optimizer, display=False)

        # measure mAP
        self.state['ap_meter'].add(self.state['output'].data, self.state['target_gt'])
        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                info = 'Epoch: [{0}][{1}/{2}]\t' \
                       'Time {batch_time_current:.3f} ({batch_time:.3f})\t' \
                       'Data {data_time_current:.3f} ({data_time:.3f})\t' \
                       'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss)

                print(info)
                with open(os.path.join(self.state['log_path'], self.state['logname']), 'a') as f:
                    f.write(info + '\n')

            else:
                info = 'Test: [{0}/{1}]\t' \
                       'Time {batch_time_current:.3f} ({batch_time:.3f})\t' \
                       'Data {data_time_current:.3f} ({data_time:.3f})\t' \
                       'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss)

                print(info)
                with open(os.path.join(self.state['log_path'], self.state['logname']), 'a') as f:
                    f.write(info + '\n')


class GCNMultiLabelMAPEngine(MultiLabelMAPEngine):
    def __init__(self, state, inp_file=None, num_class=157):
        super().__init__(state)

        if inp_file is not None:
            with open(inp_file, 'rb') as f:
                self.inp = pickle.load(f)
        else:
            self.inp = np.identity(num_class)
        self.inp = torch.from_numpy(self.inp)

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        if training:
            feature_var = torch.autograd.Variable(self.state['feature']).float()
            target_var = torch.autograd.Variable(self.state['target']).float()
            inp_var = torch.autograd.Variable(self.state['input']).float().detach()  # one hot

            # compute output
            self.state['output'] = model(feature_var)
            self.state['loss'] = criterion(self.state['output'], target_var)

            optimizer.zero_grad()
            self.state['loss'].backward()
            nn.utils.clip_grad_norm(model.parameters(), max_norm=10.0)
            optimizer.step()

        else:
            with torch.no_grad():
                feature_var = torch.autograd.Variable(self.state['feature']).float()
                target_var = torch.autograd.Variable(self.state['target']).float()
                inp_var = torch.autograd.Variable(self.state['input']).float().detach()  # one hot

                # compute output
                self.state['output'] = model(feature_var)
                self.state['loss'] = criterion(self.state['output'], target_var)