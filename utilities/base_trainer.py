import torch
import warnings
import numpy as np
import torch.nn as nn
from digicare.utilities.file_ops import file_exist, mkdir, join_path, gd
from digicare.utilities.data_io import load_pkl, save_pkl, save_nifti_simple
from digicare.utilities.misc import printx, Timer, format_sec
from torch.utils import data
from typing import Union
from digicare.utilities.plot import multi_curve_plot
try:
    from torch.optim.lr_scheduler import LRScheduler as LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler # for torch <= 1.13.5

class UserNotImplemented(RuntimeError):
    pass

class ModelTrainer_PyTorch:
    '''
    Description
    -----------
    Simple utility class for training a model under PyTorch deep learning framework with
    simple error checking mechanics. This also helps users to avoid some common errors 
    when designing a network trainer from scratch.
    This template class also implements the following features:
        * Automatic model loading/saving
        * Automatic adjust learning rate after every epoch (if lr_scheduler is given)
        * Saving the current best model using validation set (if given)
        * Plot network training process
        * Event driven design
            - override _on_epoch(...) in derived class to define how the model is trained
              in train/validation/test phase.
            - override _on_load_model(...) in derived class when model is being loaded
            - override _on_save_model(...) in derived class when model is being saved
            - ...
    Distributed parallel is not supported for now.
            
    Note
    -----------
    This class is a template class. Inherit this class and at least override & implement 
    ModelTrainer_PyTorch::_on_epoch(...) in derived class to make it fully usable. See 
    docstring of _on_epoch(...) for more info.

    * I wrote lots of model trainers and found that most of their code are the same (
      such as model loading/saving, lr scheduling, logging, ...). So I decided to make 
      a template trainer class like this, only to focus on the difference of different
      model trainers (by inherit this class and implement _on_*** functions in child 
      class). In this way I can avoid some common mistakes when writing a model trainer
      from scratch and improve my coding efficiency...
    '''
    def __init__(self, 
            output_folder: str                                     = './out',
            gpu_index:     int                                     = 0, 
            model:         Union[torch.nn.Module, None]            = None,
            optim:         Union[torch.optim.Optimizer, str, None] = 'default',
            lr_scheduler:  Union[LRScheduler, str, None]           = 'default',
            train_loader:  Union[data.Dataset, None]               = None, 
            val_loader:    Union[data.Dataset, None]               = None, 
            test_loader:   Union[data.Dataset, None]               = None,
        ):

        # initialize dataloaders
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader

        # other settings
        self.output_folder = mkdir(output_folder)
        self.gpu = gpu_index
        self.current_epoch = 1
        self.best_loss = None
        self.log = {'epoch_train_loss':[], 'epoch_val_loss':[], 'epoch_test_loss':[]}

        # initialize model
        assert model is not None, '"model" should not be None.'
        assert isinstance(model, nn.Module), '"model" should inherit from "torch.nn.Module", but got "%s".' % type(model).__name__
        self.model = model
        cur_device = str(next(self.model.parameters()).device) # "cpu", "cuda:x"
        if cur_device.startswith('cpu'):
            # automatically transfer to cuda if model is on cpu
            self.model = self.model.cuda(device=self.gpu) 
        elif cur_device.startswith('cuda'):
            if cur_device != 'cuda:%d' % self.gpu:
                raise RuntimeError('model is already on "%s", but gpu_index is set to "%d".' % (cur_device, gpu_index))

        # initialize optimizer
        if optim == 'default' or optim == None:
            print('using default Adam optimizer with initial lr=0.01, betas=(0.9,0.999).')
            self.optim = torch.optim.Adam(self.model.parameters(), lr=0.01, betas=(0.9, 0.999))
        elif isinstance(optim, str) and optim != 'default':
            raise RuntimeError('unknown optimizer setting "%s".' % optim)
        else:
            print('using user-defined optimizer "%s".' % type(optim).__name__)
            self.optim = optim
        
        # initialize lr scheduler
        if lr_scheduler == 'default':
            print('using default lr scheduler (lambda lr with decay=0.97)')
            lambda1 = lambda epoch: 0.97 ** epoch
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lambda1)
        elif lr_scheduler == None:
            print('lr scheduler disabled.')
        elif isinstance(lr_scheduler, LRScheduler):
            print('using user-defined lr scheduler "%s", make sure lr scheduler is linked to the optimizer!' % type(lr_scheduler).__name__)
        else:
            raise RuntimeError('unknown lr scheduler "%s".' % str(lr_scheduler))
        self.lr_scheduler = lr_scheduler

    def load_model(self, model_path):
        '''
        load model state, along with optimizer and lr scheduler (if exist) states.
        '''
        if file_exist(model_path) == False:
            raise RuntimeError('cannot find model "%s".' % model_path)
        model_dict : dict = torch.load(model_path, map_location = 'cuda:%d' % self.gpu)
        self.current_epoch = model_dict['current_epoch']
        self.best_loss = model_dict['best_loss'] if 'best_loss' in model_dict else None
        model_dict.pop('current_epoch')
        model_dict.pop('best_loss') if 'best_loss' in model_dict else None
        self._on_load_model(model_dict)
        self.model.load_state_dict(model_dict)
        print('Loaded model "%s".' % model_path)

        # load optimizer
        if file_exist(model_path + '.optim'):
            model_dict = torch.load(model_path + '.optim', map_location='cuda:%d' % self.gpu)
            self.optim.load_state_dict(model_dict)
            print('Optimizer state successfully loaded.')
        else:
            warnings.warn('Cannot recover optimizer state. Optimizer state will be reset to default and may '
                'cause inaccuracy in training.')

        # load lr scheduler
        if file_exist(model_path + '.lr_scheduler'):
            lr_state = torch.load(model_path + '.lr_scheduler', map_location='cuda:%d' % self.gpu)
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(lr_state)
                print('LR scheduler state successfully loaded.')
            else:
                warnings.warn('Found saved LR scheduler state but LR scheduler is not initialized.')
        else:
            if self.lr_scheduler is not None:
                warnings.warn('Cannot recover LR scheduler state. LR scheduler state will be reset to default and may '
                    'cause inaccuracy in training.')

    def save_model(self, model_path):
        '''
        save model state, along with optimizer and lr scheduler (if exist) states.
        '''
        model_dict = self.model.state_dict()
        model_dict['current_epoch'] = self.current_epoch
        model_dict['best_loss'] = self.best_loss
        self._on_save_model(model_dict)
        torch.save(model_dict, model_path)
        optim_dict = self.optim.state_dict()
        torch.save(optim_dict, model_path + '.optim')
        if self.lr_scheduler is not None:
            lr_state = self.lr_scheduler.state_dict()
            torch.save(lr_state, model_path + '.lr_scheduler')

    def plot_training_progress(self, output_image):
        # If user implemented _on_plot_training_progress(...) in child class, then we use the
        # custom implementation instead of the default implementation below.
        try:
            self._on_plot_training_progress(output_image)
        except UserNotImplemented:
            train_loss = [(float(s) if s != None else np.nan) for s in self.log['epoch_train_loss']]
            val_loss   = [(float(s) if s != None else np.nan) for s in self.log['epoch_val_loss']]
            test_loss  = [(float(s) if s != None else np.nan) for s in self.log['epoch_test_loss']]
            epochs = [i for i in range(1, len(train_loss)+1)]
            curves = {
                'training'  : { 'x': epochs, 'y': train_loss, 'color': [0.0, 0.5, 0.0], 'label': True },
                'validation': { 'x': epochs, 'y': val_loss,   'color': [0.0, 0.0, 1.0], 'label': True },
                'test'      : { 'x': epochs, 'y': test_loss,  'color': [1.0, 0.0, 0.0], 'label': True },
            }
            mkdir(gd(output_image))
            multi_curve_plot(curves, output_image, dpi=150, title='Training Progress', xlabel='Epoch', ylabel='Loss')

    def get_current_lr(self):
        return self.optim.param_groups[0]["lr"]

    def clear_best(self):
        self.best_loss = None

    def train(self, num_epochs=1000):
        
        model_latest = join_path(self.output_folder, 'model_latest.model')
        model_best = join_path(self.output_folder, 'model_best.model')
        log_file = join_path(self.output_folder, 'log.pkl')

        self.model_latest = model_latest
        self.model_best = model_best
        self.num_epochs = num_epochs

        if file_exist(model_latest) and file_exist(log_file):
            self.load_model(model_latest)
            self.log = load_pkl(log_file)
        else:
            if file_exist(model_latest) and not file_exist(log_file):
                print('* Found model "%s", but log file "%s" is missing.' % (model_latest, log_file))
            print('Training model from scratch.')

        start_epoch, end_epoch = self.current_epoch, num_epochs+1

        timer = Timer()

        print('%s: Training from epoch %d to %d.' % (timer.now(), start_epoch, num_epochs))

        self._on_train_start()

        def _table_style():
            return [
                '====================================================',
                ' EPOCH    TRAIN     VALIDATION     TEST     ELAPSED ',
                '----------------------------------------------------',
                ' *        *           *           *         *       ',
            ]
        
        print(_table_style()[0])
        print(_table_style()[1])
        print(_table_style()[2])

        printx('Trainer is launching, please wait...')

        for epoch in range(start_epoch, end_epoch):
            timer.tick()

            if self.lr_scheduler is not None:
                lr_before_epoch_start = self.get_current_lr()
            self._on_epoch_start(epoch)
            train_fetch = self._on_epoch(epoch, msg='Epoch %d/%d 1/3' % (epoch, num_epochs), data_loader=self.train_loader, phase='train') if self.train_loader is not None else None
            val_fetch   = self._on_epoch(epoch, msg='Epoch %d/%d 2/3' % (epoch, num_epochs), data_loader=self.val_loader,   phase='val')   if self.val_loader   is not None else None
            test_fetch  = self._on_epoch(epoch, msg='Epoch %d/%d 3/3' % (epoch, num_epochs), data_loader=self.test_loader,  phase='test')  if self.test_loader  is not None else None
            if self.lr_scheduler is not None:
                # Check if lr scheduler actually take effect, if not, print a warning to let user know the situation.
                # Because sometimes the user may forget to link the optimizer to lr scheduler during initialization.
                lr_after_epoch_end = self.get_current_lr()
                if lr_before_epoch_start != lr_after_epoch_end:
                    # This means user accidentally changes the learning rate during model training. Users do not need
                    # to care about learning rate scheduling as it can be handled properly by the model trainer.
                    printx('')
                    print('Warning: lr is changed by user! Normally you don\'t need to care about lr scheduling as '
                          'it will be handled properly by the model trainer.')
                self.lr_scheduler.step()
                lr_after_updated = self.get_current_lr()
                if lr_after_epoch_end == lr_after_updated:
                    printx('')
                    print('Warning: lr is not changed after epoch %d (=%f).' % (epoch, lr_after_updated))

            def _check_fetch(any_fetch, phase):
                if any_fetch is None:
                    return None
                assert isinstance(any_fetch, (float, tuple, list)), 'The return value of "self._epoch(...)" in %s phase should be of type "float", "tuple", or "list". ' \
                                                                    'Got an instance of type "%s" instead.' % (phase, type(any_fetch).__name__)
                if isinstance(any_fetch, float):
                    any_fetch = [any_fetch]
                assert isinstance(any_fetch[0], float), 'The FIRST return value of "self._epoch(...)" in %s phase should be a FLOAT scalar, which will be used in ' \
                                                        'model selection if validation set is available. Got an instance of type "%s" instead.' % (phase, type(any_fetch).__name__)
                return any_fetch

            train_fetch = _check_fetch(train_fetch, 'train')
            val_fetch   = _check_fetch(val_fetch,   'val')
            test_fetch  = _check_fetch(test_fetch,  'test')

            train_loss = train_fetch[0] if train_fetch is not None else None
            val_loss   = val_fetch[0]   if val_fetch   is not None else None
            test_loss  = test_fetch[0]  if test_fetch  is not None else None

            if val_loss is None:
                is_best = True
            else:
                is_best = self.best_loss is None or val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            def _print_one_line_epoch_summary():
                def _left_replace_placeholder_once(str_with_placeholder, content, cell_size):
                    if len(content)<cell_size: content += ' '*(cell_size-len(content))
                    elif len(content)>cell_size: content = content[:cell_size-3]+'...'
                    return str_with_placeholder.replace('*'+' '*(cell_size-1), content, 1)

                oneline_summary = _table_style()[3]
                oneline_summary = _left_replace_placeholder_once(oneline_summary, '%4d%s' % (epoch, '*' if is_best else ' '), 8)
                oneline_summary = _left_replace_placeholder_once(oneline_summary, ('%1.4f' % train_loss) if train_loss is not None else ' n/a ', 8)
                oneline_summary = _left_replace_placeholder_once(oneline_summary, ('%1.4f' % val_loss)   if val_loss   is not None else ' n/a ', 8)
                oneline_summary = _left_replace_placeholder_once(oneline_summary, ('%1.4f' % test_loss)  if test_loss  is not None else ' n/a ', 8)
                oneline_summary = _left_replace_placeholder_once(oneline_summary, '%s' % format_sec( int(timer.tick()) ), 8)

                printx('')
                print(oneline_summary)
            
            # if user returns additional info, print it
            def _print_additional_info_if_required(any_fetch, note):
                if any_fetch is None or len(any_fetch) < 2: return
                formatted_info = '  * %s: ' % note
                for item in any_fetch[1:]:
                    formatted_info += str(item)  + ' '
                print(formatted_info)

            _print_one_line_epoch_summary()
            _print_additional_info_if_required(train_fetch, 'train')
            _print_additional_info_if_required(val_fetch,   '  val')
            _print_additional_info_if_required(test_fetch,  ' test')

            self.current_epoch += 1
            self.log['epoch_train_loss'].append(train_loss)
            self.log['epoch_val_loss'].append(val_loss)
            self.log['epoch_test_loss'].append(test_loss)
            
            printx('Saving model, please do not kill the program...')
            self.save_model(model_latest)
            if is_best:
                printx('Saving best model, please do not kill the program...')
                self.save_model(model_best)
                self._on_best_epoch(epoch, model_best)
            save_pkl(self.log, log_file)
            printx('Model(s) saved. Visualizing training progress...')
            self.plot_training_progress(join_path(self.output_folder, 'progress.png'))
            self._on_epoch_end(epoch)
            printx('Preparing for next epoch...')

        printx('')
        print(_table_style()[0])

        self._on_train_end()

        print('%s: Training finished.' % timer.now())
    
    def _on_plot_training_progress(self, output_image):
        '''
        When an epoch is finished, this function will be called to
        plot model training progress. Plotted image should be saved
        to the path indicated by output_image.
        '''
        raise UserNotImplemented('not implemented.')

    def _on_load_model(self, model_dict: dict):
        '''
        Add additional operations after the model_dict is loaded.
        For example, extract additional info from model_dict
        >>> ... = model_dict['some_info']
        >>> model_dict.pop['some_info'] # <- don't forget to pop!
        NOTE: you need to manually pop info out of the dict once
        the info is extracted, since model_dict will then be passed
        to PyTorch, if it encounters some unrecognized keys it 
        will raise an error.
        '''
        pass

    def _on_save_model(self, model_dict: dict):
        '''
        add additional operations when the model_dict is about to be saved.
        for example, you can add additional info into model_dict, such as:
        >>> model_dict['some_info'] = ...
        literally you can add whatever you like into model_dict and let trainer
        base class save the model dict for you.
        '''
        pass

    def _on_train_start(self):
        '''
        add additional operations when the training process is about to start.
        '''
        pass

    def _on_train_end(self):
        '''
        add additional operations when the training process is ended.
        '''
        pass

    def _on_epoch_start(self, epoch: int):
        '''
        called when current epoch is about to start training.
        '''
        pass

    def _on_epoch_end(self, epoch: int):
        '''
        called when current epoch training is finished.
        '''
        pass

    def _on_epoch(self, 
            epoch:       int,
            msg:         str                          = '', 
            data_loader: Union[data.DataLoader, None] = None,
            phase:       str                          = '',
        ) -> Union[ float, tuple, list]:
        '''
        phase: can be one of "train", "val", or "test" indicating if model is currently under "training", "validation" or "test" mode.
        NOTE: self._on_epoch() can return multiple items, but only the first item will be used for model selection (it is
            treated as the most important metric and the rest are just additional info, also be aware that the first item
            should be a scalar value (float).
        '''
        raise UserNotImplemented('Unimplemented virtual function "ModelTrainer_PyTorch::_on_epoch(...)" called. Please implement it in child class.')

    def _on_best_epoch(self, best_epoch: int, best_model_path: str):
        '''
        will be called when a current best model is produced and saved.
        '''
        pass


    # utility function(s)
    @staticmethod
    def dump_tensor_as_nifti(tensor: Union[torch.Tensor, np.ndarray], output_folder: str, name_prefix: str = 'dump_'):
        '''
        Description
        -----------
        Save tensor/ndarray data as nifti image. 
        Supports saving 2D, 3D, 4D or 5D tensor or ndarray.        

        Parameters
        -----------
        @param tensor: torch.Tensor | np.ndarray 
            Tensor or ndarray that is about to be saved. 
            Its data format can be:
            * 2D tensor: [X, Y]
            * 3D tensor: [X, Y, Z]
            * 4D tensor: [B, C, X, Y]
            * 5D tensor: [B, C, X, Y, Z]

        @param output_folder: str
            Nifti file output folder.
        
        @param name_prefix: str
            File name prefix for the output Nifti file(s).
        '''
        assert isinstance(tensor, (torch.Tensor, np.ndarray)), \
            '"tensor" should be of type "torch.Tensor" or "numpy.ndarray", ' \
            'but got "%s".' % (type(tensor).__name__)
        if isinstance(tensor, torch.Tensor):
            ndarr = tensor.detach().cpu().numpy().astype('float32')
        else:
            ndarr = tensor
        ndarr_dim = len(ndarr.shape)
        assert ndarr_dim in [2, 3, 4, 5], 'Only support dumping 2D~5D tensors, got %dD tensor with shape: %s.' % (ndarr_dim, str(ndarr.shape))
        mkdir(output_folder)

        def _save_4D_or_5D_ndarr(tensor):
            B, C = tensor.shape[0], tensor.shape[1]
            for b in range(B):
                for c in range(C):
                    path = join_path(output_folder, '%sb%d_ch%d.nii.gz' % (name_prefix, b, c))
                    save_nifti_simple(tensor[b,c], path)        
            print('Tensor saved to "%s".' % join_path(output_folder, '%s*.nii.gz' % name_prefix))

        def _save_2D_or_3D_ndarr(tensor):
            path = join_path(output_folder, '%s.nii.gz' % name_prefix)
            save_nifti_simple(tensor, path)
            print('Tensor saved to "%s".' % path)

        if ndarr_dim in [4, 5]:
            _save_4D_or_5D_ndarr(ndarr)
        elif ndarr_dim in [2, 3]:
            _save_2D_or_3D_ndarr(ndarr)
