from digicare.runtime.wmh_diagnosis.image_loader import ImageLoader_Patch3D_Base, ImageLoader_Patch3D_Cube128_1p5mm
from digicare.utilities.database import Database
from digicare.utilities.base_trainer import ModelTrainer_PyTorch
from typing import Union
import torch
import torch.nn as nn
from torch.utils import data
from digicare.utilities.misc import minibar, Timer
from typing import List
from digicare.utilities.file_ops import join_path
import numpy as np
from digicare.utilities.data_io import save_nifti_simple

def vanilla_mse(pred, true):
    assert pred.shape == true.shape
    numel = pred.numel()
    return torch.sum(torch.pow(pred - true, 2)) / numel

def masked_mse(pred, true, mask):
    assert pred.shape == true.shape == mask.shape
    mask = torch.where(mask > 0.5, 1.0, 0.0)
    return torch.sum(torch.pow(pred * mask - true * mask, 2)) / torch.sum(mask)

class ModelTrainer_Multimodality_2Stages(ModelTrainer_PyTorch):
    def __init__(self, output_folder, gpu_index, 
                 model, optim, lr_scheduler, 
                 train_loader, val_loader, test_loader):
        super().__init__(output_folder, gpu_index, model, optim, lr_scheduler, train_loader, val_loader, test_loader)
    
    def _stage_1_training(self, 
        epoch:int,
        msg: str = '', 
        data_loader: Union[data.DataLoader, None] = None , 
        phase: str='') -> float:

        self.model.train() if phase == 'train' else self.model.eval()
        image_loader: ImageLoader_Patch3D_Base = data_loader.dataset
        class_weights = torch.Tensor(image_loader.get_weight_of_each_class()).float().cuda(self.gpu)

        loss_, acc_ = [], []
        timer = Timer()

        for batch_idx, (record, x_DS, m_DS, a_DS, diagnosis) in enumerate(data_loader):
            x_DS: List[torch.Tensor]
            m_DS: List[torch.Tensor]
            a_DS: List[torch.Tensor]

            for i in range(len(x_DS)):
                assert x_DS[i].shape[0] == 1, 'Only support batch_size=1.'
                assert a_DS[i].shape[0] == 1, 'Only support batch_size=1.'
                assert m_DS[i].shape[0] == 1, 'Only support batch_size=1.'

            # further process data sent from data loader.
            for i in range(len(x_DS)):
                x_DS[i] = x_DS[i].float().cuda(self.gpu)
                a_DS[i] = a_DS[i].float().cuda(self.gpu)
                m_DS[i] = m_DS[i].float().cuda(self.gpu)

            diagnosis = diagnosis[0] # pytorch wraps a batch dimension for us, here we unpack the dim
            true_class_id = torch.Tensor([image_loader.map_class_name_to_id(diagnosis)]).long().cuda(self.gpu)

            def _regression_loss_DS(pred_DS, true_DS, mask_DS):
                assert len(pred_DS) == len(true_DS), 'Deep supervision resolution # stages not equal (%d vs %s).' % (len(pred_DS), len(true_DS))
                losses = None
                for pred, true, mask in zip(pred_DS, true_DS, mask_DS):
                    loss = masked_mse(pred, true, mask)
                    #loss = vanilla_mse(pred, true)
                    if losses is None:
                        losses = loss
                    else:
                        losses += loss
                return losses / len(pred_DS)

            def _simply_feed_forward_once(*args, **kwargs):
                if 'is_training' not in kwargs:
                    raise RuntimeError('you need to set "is_training=True" or "is_training=False" when calling this function.')
                ce_loss = nn.CrossEntropyLoss(weight=class_weights)
                if kwargs['is_training']:
                    self.model.train()
                    fetch = self.model(*args)
                    self.optim.zero_grad()
                    c, y = fetch[0], fetch[1:]
                    loss0 = ce_loss(c, true_class_id)
                    loss1 = masked_mse(y[0], a_DS[0], m_DS[0])
                    loss = loss1
                    loss.backward() 
                    self.optim.step()
                else:
                    with torch.no_grad():
                        self.model.eval()
                        fetch = self.model(*args)
                        c, y = fetch[0], fetch[1:]
                        loss0 = ce_loss(c, true_class_id)
                        loss1 = masked_mse(y[0], a_DS[0], m_DS[0])
                        loss = loss1
                return fetch, loss0, loss1

            input_tensor = torch.cat([x_DS[0], m_DS[0]], dim=1)

            fetch, loss0, loss1 = _simply_feed_forward_once(input_tensor, is_training=(phase=='train'))
            loss_.append(loss1.item())

            # calculate acc metric
            c, y = fetch[0], fetch[1:]
            pred_class_id = int(torch.argmax(c).long().detach().cpu().item())
            true_class_id = int(true_class_id.detach().cpu().item())
            acc_.append(1 if pred_class_id == true_class_id else 0)

            minibar(msg, batch_idx+1, len(data_loader), timer.elapsed(), last='%.4f|%s' % 
                    (loss1.item(), 
                     str(pred_class_id)+str(true_class_id)
                    ))
        
        return np.mean(loss_), 'w=%s, acc=%.2f%%, samp=%s' % \
            (
                str(image_loader.get_weight_of_each_class()), 
                100.0 * np.mean(acc_),
                ''.join([str(item) for item in acc_])
            )

    def _stage_2_training(self, 
        epoch:int,
        msg: str = '', 
        data_loader: Union[data.DataLoader, None] = None , 
        phase: str='') -> float:

        self.model.train() if phase == 'train' else self.model.eval()
        image_loader: ImageLoader_Patch3D_Base = data_loader.dataset
        class_weights = torch.Tensor(image_loader.get_weight_of_each_class()).float().cuda(self.gpu)

        # # freeze parameters
        # for name, param in self.model.named_parameters():
        #     param.requires_grad = False
        # for name, param in self.model.named_parameters():
        #     if name.startswith('fc_6_a') or name.startswith('fc_6_b'):
        #         param.requires_grad = True

        loss_, acc_ = [], []
        timer = Timer()

        for batch_idx, (record, x_DS, m_DS, a_DS, diagnosis) in enumerate(data_loader):
            x_DS: List[torch.Tensor]
            m_DS: List[torch.Tensor]
            a_DS: List[torch.Tensor]
            
            # further process data sent from data loader.
            for i in range(len(a_DS)):
                x_DS[i] = x_DS[i].float().cuda(self.gpu)
                a_DS[i] = a_DS[i].float().cuda(self.gpu)
                m_DS[i] = m_DS[i].float().cuda(self.gpu)
            diagnosis = diagnosis[0] # pytorch wraps a batch dimension for us, here we unpack the dim
            true_class_id = torch.Tensor([image_loader.map_class_name_to_id(diagnosis)]).long().cuda(self.gpu)

            def _simply_feed_forward_once(*args, **kwargs):
                if 'is_training' not in kwargs:
                    raise RuntimeError('you need to set "is_training=True" or "is_training=False" when calling this function.')
                ce_loss = nn.CrossEntropyLoss(weight=class_weights)
                if kwargs['is_training']:
                    self.model.train()
                    fetch = self.model(*args)
                    self.optim.zero_grad()
                    c, y = fetch[0], fetch[1:]
                    loss0 = ce_loss(c, true_class_id)
                    loss0.backward() 
                    self.optim.step()
                else:
                    with torch.no_grad():
                        self.model.eval()
                        fetch = self.model(*args)
                        c, y = fetch[0], fetch[1:]
                        #self.dump_tensor_as_nifti(y[0], './dump/', str(batch_idx)+'_y_')
                        loss0 = ce_loss(c, true_class_id)
                return fetch, loss0

            input_tensor = torch.cat([x_DS[0], m_DS[0]], dim=1)

            fetch, loss0 = _simply_feed_forward_once(input_tensor, is_training=(phase=='train'))
            loss_.append(loss0.item())

            # calculate acc metric
            c, y = fetch[0], fetch[1:]
            pred_class_id = int(torch.argmax(c).long().detach().cpu().item())
            true_class_id = int(true_class_id.detach().cpu().item())
            acc_.append(1 if pred_class_id == true_class_id else 0)

            minibar(msg, batch_idx+1, len(data_loader), timer.elapsed(), last='%.4f|%s' % 
                    (loss0.item(), 
                     str(pred_class_id)+str(true_class_id)
                    ))
        
        return -np.mean(acc_), 'w=%s, acc=%.2f%%, samp=%s' % \
            (
                str(image_loader.get_weight_of_each_class()), 
                100.0 * np.mean(acc_),
                ''.join([str(item) for item in acc_])
            )

    def _on_epoch(self, 
        epoch: int,
        msg: str = '', 
        data_loader: Union[data.DataLoader, None] = None , 
        phase: str='') -> float:

        if epoch<=100:
            return self._stage_1_training(epoch, msg, data_loader, phase)
        else:
            return self._stage_2_training(epoch, msg, data_loader, phase)

class ModelTrainer_Baseline_2Stages(ModelTrainer_PyTorch):
    def __init__(self, output_folder, gpu_index, 
                 model, optim, lr_scheduler, 
                 train_loader, val_loader, test_loader):
        super().__init__(output_folder, gpu_index, model, optim, lr_scheduler, train_loader, val_loader, test_loader)
    
    def _stage_1_training(self, 
        epoch:int,
        msg: str = '', 
        data_loader: Union[data.DataLoader, None] = None , 
        phase: str='') -> float:

        self.model.train() if phase == 'train' else self.model.eval()
        image_loader: ImageLoader_Patch3D_Base = data_loader.dataset
        class_weights = torch.Tensor(image_loader.get_weight_of_each_class()).float().cuda(self.gpu)

        loss_, acc_ = [], []
        timer = Timer()

        for batch_idx, (record, x_DS, m_DS, a_DS, diagnosis) in enumerate(data_loader):
            x_DS: List[torch.Tensor]
            m_DS: List[torch.Tensor]
            a_DS: List[torch.Tensor]

            for i in range(len(x_DS)):
                assert x_DS[i].shape[0] == 1, 'Only support batch_size=1.'
                assert a_DS[i].shape[0] == 1, 'Only support batch_size=1.'
                assert m_DS[i].shape[0] == 1, 'Only support batch_size=1.'

            # further process data sent from data loader.
            for i in range(len(x_DS)):
                x_DS[i] = x_DS[i].float().cuda(self.gpu)
                a_DS[i] = a_DS[i].float().cuda(self.gpu)
                m_DS[i] = m_DS[i].float().cuda(self.gpu)

            diagnosis = diagnosis[0] # pytorch wraps a batch dimension for us, here we unpack the dim
            true_class_id = torch.Tensor([image_loader.map_class_name_to_id(diagnosis)]).long().cuda(self.gpu)

            def _regression_loss_DS(pred_DS, true_DS, mask_DS):
                assert len(pred_DS) == len(true_DS), 'Deep supervision resolution # stages not equal (%d vs %s).' % (len(pred_DS), len(true_DS))
                losses = None
                for pred, true, mask in zip(pred_DS, true_DS, mask_DS):
                    loss = masked_mse(pred, true, mask)
                    #loss = vanilla_mse(pred, true)
                    if losses is None:
                        losses = loss
                    else:
                        losses += loss
                return losses / len(pred_DS)

            def _simply_feed_forward_once(*args, **kwargs):
                if 'is_training' not in kwargs:
                    raise RuntimeError('you need to set "is_training=True" or "is_training=False" when calling this function.')
                ce_loss = nn.CrossEntropyLoss(weight=class_weights)
                if kwargs['is_training']:
                    self.model.train()
                    fetch = self.model(*args)
                    self.optim.zero_grad()
                    c, y = fetch[0], fetch[1:]
                    loss0 = ce_loss(c, true_class_id)
                    loss1 = masked_mse(y[0], x_DS[0], m_DS[0])
                    loss = loss0 + loss1
                    loss.backward() 
                    self.optim.step()
                else:
                    with torch.no_grad():
                        self.model.eval()
                        fetch = self.model(*args)
                        c, y = fetch[0], fetch[1:]
                        loss0 = ce_loss(c, true_class_id)
                        loss1 = masked_mse(y[0], x_DS[0], m_DS[0])
                        loss = loss0 + loss1
                return fetch, loss0, loss1

            input_tensor = torch.cat([x_DS[0], m_DS[0]], dim=1)

            fetch, loss0, loss1 = _simply_feed_forward_once(input_tensor, is_training=(phase=='train'))
            loss_.append(loss1.item())

            # calculate acc metric
            c, y = fetch[0], fetch[1:]
            pred_class_id = int(torch.argmax(c).long().detach().cpu().item())
            true_class_id = int(true_class_id.detach().cpu().item())
            acc_.append(1 if pred_class_id == true_class_id else 0)

            minibar(msg, batch_idx+1, len(data_loader), timer.elapsed(), last='%.4f|%s' % 
                    (loss1.item(), 
                     str(pred_class_id)+str(true_class_id)
                    ))
        
        return np.mean(loss_), 'w=%s, acc=%.2f%%, samp=%s' % \
            (
                str(image_loader.get_weight_of_each_class()), 
                100.0 * np.mean(acc_),
                ''.join([str(item) for item in acc_])
            )

    def _stage_2_training(self, 
        epoch:int,
        msg: str = '', 
        data_loader: Union[data.DataLoader, None] = None , 
        phase: str='') -> float:

        self.model.train() if phase == 'train' else self.model.eval()
        image_loader: ImageLoader_Patch3D_Base = data_loader.dataset
        class_weights = torch.Tensor(image_loader.get_weight_of_each_class()).float().cuda(self.gpu)

        # freeze parameters
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        for name, param in self.model.named_parameters():
            if name.startswith('fc_6_a') or name.startswith('fc_6_b'):
                param.requires_grad = True

        loss_, acc_ = [], []
        timer = Timer()

        for batch_idx, (record, x_DS, m_DS, a_DS, diagnosis) in enumerate(data_loader):
            x_DS: List[torch.Tensor]
            m_DS: List[torch.Tensor]
            a_DS: List[torch.Tensor]
            
            # further process data sent from data loader.
            for i in range(len(a_DS)):
                x_DS[i] = x_DS[i].float().cuda(self.gpu)
                a_DS[i] = a_DS[i].float().cuda(self.gpu)
                m_DS[i] = m_DS[i].float().cuda(self.gpu)
            diagnosis = diagnosis[0] # pytorch wraps a batch dimension for us, here we unpack the dim
            true_class_id = torch.Tensor([image_loader.map_class_name_to_id(diagnosis)]).long().cuda(self.gpu)

            def _simply_feed_forward_once(*args, **kwargs):
                if 'is_training' not in kwargs:
                    raise RuntimeError('you need to set "is_training=True" or "is_training=False" when calling this function.')
                ce_loss = nn.CrossEntropyLoss(weight=class_weights)
                if kwargs['is_training']:
                    self.model.train()
                    fetch = self.model(*args)
                    self.optim.zero_grad()
                    c, y = fetch[0], fetch[1:]
                    loss0 = ce_loss(c, true_class_id)
                    loss0.backward() 
                    self.optim.step()
                else:
                    with torch.no_grad():
                        self.model.eval()
                        fetch = self.model(*args)
                        c, y = fetch[0], fetch[1:]
                        #self.dump_tensor_as_nifti(y[0], './dump/', str(batch_idx)+'_y_')
                        loss0 = ce_loss(c, true_class_id)
                return fetch, loss0

            input_tensor = torch.cat([x_DS[0], m_DS[0]], dim=1)
            
            fetch, loss0 = _simply_feed_forward_once(input_tensor, is_training=(phase=='train'))
            loss_.append(loss0.item())

            # calculate acc metric
            c, y = fetch[0], fetch[1:]
            pred_class_id = int(torch.argmax(c).long().detach().cpu().item())
            true_class_id = int(true_class_id.detach().cpu().item())
            acc_.append(1 if pred_class_id == true_class_id else 0)

            minibar(msg, batch_idx+1, len(data_loader), timer.elapsed(), last='%.4f|%s' % 
                    (loss0.item(), 
                     str(pred_class_id)+str(true_class_id)
                    ))
        
        return -np.mean(acc_), 'w=%s, acc=%.2f%%, samp=%s' % \
            (
                str(image_loader.get_weight_of_each_class()), 
                100.0 * np.mean(acc_),
                ''.join([str(item) for item in acc_])
            )

    def _on_epoch(self, 
        epoch: int,
        msg: str = '', 
        data_loader: Union[data.DataLoader, None] = None , 
        phase: str='') -> float:

        if epoch<=100:
            return self._stage_1_training(epoch, msg, data_loader, phase)
        else:
            return self._stage_2_training(epoch, msg, data_loader, phase)
