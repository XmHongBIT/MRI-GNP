import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import Module
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils import data
from typing import Union, List

from torch.utils.data.dataset import Dataset
from digicare.utilities.base_trainer import ModelTrainer_PyTorch
from digicare.utilities.misc import minibar, Timer, ignore_print
from scipy.ndimage import gaussian_filter
from digicare.utilities.file_ops import mkdir, gd, join_path
from digicare.utilities.plot import multi_curve_plot

def mse_loss(pred, true):
    assert pred.shape == true.shape
    numel = pred.numel()
    return torch.sum(torch.pow(pred - true, 2)) / numel

def masked_mse_loss(pred, true, mask):
    assert pred.shape == true.shape == mask.shape
    mask = torch.where(mask > 0.5, 1.0, 0.0)
    return torch.sum(torch.pow(pred * mask - true * mask, 2)) / torch.sum(mask)

class wmh_subtype_diagnosis_nll_trainer(ModelTrainer_PyTorch):
    def __init__(self, output_folder, gpu_index, 
                model, optim, lr_scheduler, 
                train_loader, val_loader, test_loader,
                all_class_names):
        super().__init__(output_folder, gpu_index, model, optim, 
                         lr_scheduler, train_loader, val_loader, test_loader)
        self.all_class_names = all_class_names
        self._initialize_class_name_id_mapping()

    def _initialize_class_name_id_mapping(self):
        self._class_name_to_id_map = {}
        self._id_to_class_name_map = {}
        i = 0
        for class_name in self.all_class_names:
            self._class_name_to_id_map[class_name] = i
            self._id_to_class_name_map[i] = class_name
            i += 1
        print('class_name_to_id:', self._class_name_to_id_map)
        print('class_id_to_name:', self._id_to_class_name_map)
    def _get_class_name_from_class_id(self, class_id):
        return self._id_to_class_name_map[class_id]
    def _get_class_id_from_class_name(self, class_name):
        return self._class_name_to_id_map[class_name]
    
    def _on_epoch(self, epoch:int, msg: str = '', data_loader: Union[data.DataLoader, None] = None , phase: str=''):

        assert data_loader.batch_size == 1, 'Only support batch size = 1, got %d.' % data_loader.batch_size

        self.model.train() if phase == 'train' else self.model.eval()

        loss_, acc_ = [], []
        timer = Timer()

        for batch_idx, (record, x_DS, m_DS, a_DS, _) in enumerate(data_loader):
            x_DS: List[torch.Tensor]
            m_DS: List[torch.Tensor]
            a_DS: List[torch.Tensor]
            
            # further process data sent from data loader.
            for i in range(len(x_DS)):
                x_DS[i] = x_DS[i].float().cuda(self.gpu)
                a_DS[i] = a_DS[i].float().cuda(self.gpu)
                m_DS[i] = m_DS[i].float().cuda(self.gpu)

            def _feed_forward(*args, **kwargs):
                if 'is_training' not in kwargs:
                    raise RuntimeError('you need to set "is_training=True" or '
                                       '"is_training=False" when calling this function.')
                if kwargs['is_training']:
                    self.model.train()
                    loss = nn.CrossEntropyLoss()
                    logits: torch.Tensor = self.model(*args)
                    self.optim.zero_grad()
                    real_class = torch.tensor(self._get_class_id_from_class_name(record['diagnosis'][0])).long().cuda(self.gpu).reshape(1)
                    loss1 = loss(logits, real_class)
                    loss1.backward() 
                    self.optim.step()
                    loss1 = loss1.item()

                    pred_class = int(torch.argmax(logits, dim=1).long().cpu().numpy())
                    real_class = int(real_class.cpu().numpy())
                    acc = (1 if pred_class == real_class else 0)
                else:
                    with torch.no_grad():
                        self.model.eval()
                        logits = self.model(*args)
                        pred_class = int(torch.argmax(logits, dim=1).long().cpu().numpy())
                        real_class = self._get_class_id_from_class_name(record['diagnosis'][0])
                        loss1 = (-1 if pred_class == real_class else 0)
                        acc = (1 if pred_class == real_class else 0)

                return loss1, acc

            loss1, acc = _feed_forward(x_DS[0], a_DS[0], is_training=(phase=='train'))
            loss_.append(loss1)
            acc_.append(acc)

            minibar(msg, batch_idx+1, len(data_loader), timer.elapsed())

        return np.mean(loss_), np.mean(acc_)

class nll_predictor_trainer(ModelTrainer_PyTorch):
    def __init__(self, output_folder, gpu_index, 
                model, optim, lr_scheduler, 
                train_loader, val_loader, test_loader):
        super().__init__(output_folder, gpu_index, model, optim, 
                         lr_scheduler, train_loader, val_loader, test_loader)
    
        self._forward_cache = None # cached temporary results in each forward pass, 
                                   # will be overwritten in every batch
        self._store_cache_to_folder = ''
    
    def _on_train_start(self):
        self._store_cache_to_folder = join_path(self.output_folder, 'nll_map_regression')

    def _on_plot_training_progress(self, output_image):
        train_loss = [(float(s) if s != None else np.nan) for s in self.log['epoch_train_loss']]
        val_loss   = [(float(s) if s != None else np.nan) for s in self.log['epoch_val_loss']]
        test_loss  = [(float(s) if s != None else np.nan) for s in self.log['epoch_test_loss']]
        epochs = [i for i in range(1, len(train_loss)+1)]
        curves = {
            'training'  : { 'x': epochs, 'y': train_loss, 'color': [0.0, 0.5, 0.0], 'label': True },
            'validation': { 'x': epochs, 'y': val_loss,   'color': [0.0, 0.0, 1.0], 'label': True },
            'test'      : { 'x': epochs, 'y': test_loss,  'color': [1.0, 0.0, 0.0], 'label': True },
        }
        def _discard_nan(arr):
            return [item for item in arr if not np.isnan(item)]
        mkdir(gd(output_image))
        multi_curve_plot(curves, output_image, dpi=150, title='Training Progress', xlabel='Epoch', ylabel='Loss',
                         xlim=[1, self.num_epochs], ylim=[0, np.max(_discard_nan(train_loss + val_loss + test_loss))])

    def _on_batch_finished(self, phase, batch_idx):
        if phase == 'val' and self._store_cache_to_folder != '':
            # mask nll map by brain mask
            subject_name = self._forward_cache['subject_name']
            brain_mask   = self._forward_cache['brain_mask']
            nll_map      = self._forward_cache['nll_map']
            nll_map_gt   = self._forward_cache['nll_map_gt']
            assert brain_mask.shape == nll_map.shape, 'brain_mask: %s, nll_map: %s' % (str(brain_mask.shape), str(nll_map.shape))
            masked_nll_map = nll_map * gaussian_filter(brain_mask, sigma=5)
            with ignore_print():
                self.dump_tensor_as_nifti(masked_nll_map, self._store_cache_to_folder, name_prefix=subject_name)
                self.dump_tensor_as_nifti(    nll_map_gt, self._store_cache_to_folder, name_prefix=subject_name+'_gt')
            
    def _on_epoch(self, 
        epoch:int,
        msg: str = '', 
        data_loader: Union[data.DataLoader, None] = None , 
        phase: str=''
    ):

        assert data_loader.batch_size == 1, 'Only support batch size = 1, got %d.' % data_loader.batch_size

        self.model.train() if phase == 'train' else self.model.eval()

        loss_ = []
        timer = Timer()

        for batch_idx, (record, x_DS, m_DS, a_DS, _) in enumerate(data_loader):
            x_DS: List[torch.Tensor]
            m_DS: List[torch.Tensor]
            a_DS: List[torch.Tensor]
            
            # further process data sent from data loader.
            for i in range(len(x_DS)):
                x_DS[i] = x_DS[i].float().cuda(self.gpu)
                a_DS[i] = a_DS[i].float().cuda(self.gpu)
                m_DS[i] = m_DS[i].float().cuda(self.gpu)

            def _feed_forward(*args, **kwargs):
                if 'is_training' not in kwargs:
                    raise RuntimeError('you need to set "is_training=True" or '
                                       '"is_training=False" when calling this function.')
                if kwargs['is_training']:
                    self.model.train()
                    a_hat: torch.Tensor = self.model(*args)
                    self.optim.zero_grad()
                    loss1 = masked_mse_loss(a_hat, a_DS[0], m_DS[0])
                    loss1.backward() 
                    self.optim.step()
                else:
                    with torch.no_grad():
                        self.model.eval()
                        a_hat = self.model(*args)
                        loss1 = masked_mse_loss(a_hat, a_DS[0], m_DS[0])

                self._forward_cache = {
                    'nll_map': a_hat.detach().cpu().numpy()[0,0],
                    'nll_map_gt': a_DS[0].detach().cpu().numpy()[0,0],
                    'brain_mask': m_DS[0].detach().cpu().numpy()[0,0],
                    'subject_name': record['subject_name'][0],
                }

                return loss1

            input_tensor = x_DS[0]
            loss1 = _feed_forward(input_tensor, is_training=(phase=='train'))
            loss_.append(loss1.item())

            minibar(msg, batch_idx+1, len(data_loader), timer.elapsed())

            self._on_batch_finished(phase, batch_idx)

        return np.mean(loss_)
