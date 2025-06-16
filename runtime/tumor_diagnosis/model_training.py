from digicare.utilities.database import Database
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from typing import List, Union
from digicare.utilities.misc import minibar, Timer, printv, printx
from digicare.utilities.data_io import load_pkl
from digicare.utilities.file_ops import cp, file_exist, mkdir, join_path, gd, mv, laf, gn, rm
from digicare.runtime.tumor_diagnosis.data_processing import load_images_from_record
from digicare.utilities.base_trainer import ModelTrainer_PyTorch, LRScheduler
from digicare.utilities.data_io import save_nifti_simple
from scipy.ndimage import zoom
from digicare.runtime.tumor_diagnosis.metrics import PerformanceStatistics
from digicare.utilities.data_io import SimpleExcelWriter

# defines data loader, model trainer, and model statistics

class DataLoader(data.Dataset):
    def __init__(self, 
        database: Database, 
        num_iters: Union[int, None] = 200, 
        loading_configs = {
            # default loading configurations given here
            'enable_data_augmentation': False,
            'augmentation_params': {
                'enable_flip'   : False,
                'enable_rotate' : True,
                'rotate_angle_range' : 10, # -10~+10 degrees
                'rotate_plane'       : [(0,1), (0,2), (1,2)],
                'enable_noise'  : True,
                'noise_std_percent'  : 0.1
            },
            'with_image': True,
            'with_seg': True,
            'with_lesion_volume': False,
            'with_sex': False,
            'with_age': False,
            'with_posvec': False,
            'with_radiomics':False,
            'load_seg_from_which_key': 'autoseg',
            'use_which_channels_in_seg': [1], # channel id starts with 0, so 1 means second channel by default.
            'posvec_len': 0,      # must be set properly
            'radiomics_len': 0,   # must be set properly
            'image_loading_mode': '2D',
            'class_key_name': '', # must be set properly
            '2.5D_slice_step': 5, # only valid in 2.5D data loading mode
        }):
        assert loading_configs['posvec_len'] > 0, 'invalid positional vector length setting.'
        assert loading_configs['radiomics_len'] > 0, 'invalid radiomics feature vector length setting.'
        assert loading_configs['class_key_name'] != '', 'invalid class_key_name.'
        assert loading_configs['image_loading_mode'] in ['2D', '2.5D', '3D'], \
            'unsupported image_loading_mode: "%s".' % self.loading_configs['image_loading_mode']

        print('dataloader loading configs:', loading_configs)

        super().__init__()
        self.num_iters = num_iters # number of iterations per epoch
        self.database = database   # database object
        self.loading_configs = loading_configs
        self.slice_offset = 0 # only for test time augmentation

    def __len__(self):
        # if num_iters is None, then use all the samples in database 
        return self.num_iters if self.num_iters is not None else self.database.num_records()
    
    def __getitem__(self, index):
        i = index if self.num_iters is None else np.random.randint(0, self.database.num_records())
        record = self.database.get_record(i)
        if self.slice_offset != 0:
            printv(2,'note: loading data with slice offset=%d' % self.slice_offset)
        
        # x: input images, y: one-hot
        x, y, _, v = load_images_from_record(record, 
            data_augmentation=self.loading_configs['enable_data_augmentation'], 
            aug_params=self.loading_configs['augmentation_params'],
            load_seg_key=self.loading_configs['load_seg_from_which_key'], 
            load_mode=self.loading_configs['image_loading_mode'],
            slice_step=self.loading_configs['2.5D_slice_step'],
            slice_offset=self.slice_offset,
            load_ki67_estimated_deform_key='ki67_deform')
        gt_target = record[self.loading_configs['class_key_name']]
        assert y is not None, 'training sample does not have segmentation! \n %s' % str(record)
        x, y = x.astype('float32'), y.astype('float32')
        
        # assemble network inputs (images and segmentation)
        if self.loading_configs['with_image'] == False:
            packed_images = np.zeros([1, *x.shape[1:]]).astype('float32')
            if self.loading_configs['with_seg']:
                for channel_id in self.loading_configs['use_which_channels_in_seg']:
                    if channel_id < y.shape[0]:
                        packed_images[0] += y[channel_id]
                packed_images[0] = (packed_images[0] > 0.5).astype('float32')
        else:
            packed_images = np.zeros([x.shape[0]+1, *x.shape[1:]]).astype('float32')
            packed_images[0:x.shape[0]] = x
            if self.loading_configs['with_seg']:
                for channel_id in self.loading_configs['use_which_channels_in_seg']:
                    if channel_id < y.shape[0]:
                        packed_images[x.shape[0]] += y[channel_id]
                packed_images[x.shape[0]] = (packed_images[x.shape[0]] > 0.5).astype('float32')
        
        # assemble additional inputs (some scalar values, lesion volume, sex, age, lesion positional vector)
        lesion_volume = (float(v) if self.loading_configs['with_lesion_volume'] else float(0.0)) / 1000.0 # divide volume by 1000.0
        sex = (0.0 if record['sex'] == 'F' else 1.0) if self.loading_configs['with_sex'] else -1.0
        age = (float(record['age']) if self.loading_configs['with_age'] else 0.0) / 10.0 # we divided age by 10
        if self.loading_configs['with_posvec']:
            posvec = np.array([float(item) for item in record['autoseg_posvec'].split(',')])
            assert len(posvec) == self.loading_configs['posvec_len']
        else:
            posvec = np.zeros([self.loading_configs['posvec_len']]).astype('float32')
        if self.loading_configs['with_radiomics']:
            radiomics_vec = load_pkl(record['radiomics_vec'])
            assert len(radiomics_vec) == self.loading_configs['radiomics_len'], \
                'wrong radiomics feature vector length, expected %d, but got %d.' % \
                (self.loading_configs['radiomics_len'], len(radiomics_vec))
            radiomics_vec = np.array(radiomics_vec)
        else:
            radiomics_vec = np.zeros([self.loading_configs['radiomics_len']]).astype('float32')
        return record, packed_images, lesion_volume, sex, age, posvec, radiomics_vec, gt_target

class GenericClassifierTrainer(ModelTrainer_PyTorch):
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
    def _initialize_trainer_statistics(self, all_class_names):
        self.stats = PerformanceStatistics(all_class_names)

    def _initialize_class_weights(self, data_loader: data.DataLoader):
        # NOTE: calculate class weights if all samples were used for training
        def _sample_ratio_to_sample_weight(ratios):
            ratios = np.array(ratios).astype('float32')
            ratios = ratios / ratios.sum()
            weights = 1.0 / ratios
            weights = weights / weights.sum()
            return weights
        image_loader : DataLoader = data_loader.dataset
        database_obj : Database   = data_loader.dataset.database
        if image_loader.num_iters != None:
            self.all_class_weights = [1.0] * len(self.all_class_names)
        else:
            sample_num_in_each_class = {}
            for class_name in self.all_class_names:
                sample_num_in_each_class[class_name] = 0
            for class_name in database_obj.data_dict[self.class_key_name]:
                if class_name in self.all_class_names:
                    sample_num_in_each_class[class_name] += 1
            l = [ 0 for class_name in self.all_class_names ]
            for class_name in self.all_class_names:
                l[self._get_class_id_from_class_name(class_name)] = sample_num_in_each_class[class_name]
            l = _sample_ratio_to_sample_weight( l )
            sample_weight_in_each_class = {}
            for class_name in self.all_class_names:
                sample_weight_in_each_class[class_name] = l[ self._get_class_id_from_class_name(class_name) ]
            self.all_class_weights = l
            printx('')
            print('* [%s] sample_num_in_each_class:' % self.current_phase, sample_num_in_each_class)
            print('* [%s] sample_weight_in_each_class:' % self.current_phase, sample_weight_in_each_class)
    
    def __init__(self, 
        output_folder: str                                     = './out',
        gpu_index:     int                                     = 0, 
        model:         Union[torch.nn.Module, None]            = None,
        optim:         Union[torch.optim.Optimizer, str, None] = 'default',
        lr_scheduler:  Union[LRScheduler, str, None]           = 'default',
        train_loader:  Union[data.Dataset, None]               = None, 
        val_loader:    Union[data.Dataset, None]               = None, 
        test_loader:   Union[data.Dataset, None]               = None,
        #
        all_class_names:        Union[List[str], None]         = None,
        class_key_name:               Union[str, None]         = None,
        pretrained_model:       Union[List[str], None]         = None,
    ):
        super().__init__(output_folder, gpu_index, model, optim, lr_scheduler, train_loader, val_loader, test_loader)
        self.all_class_names = all_class_names
        self.class_key_name = class_key_name
        assert len(self.all_class_names) > 0, 'invalid class name configuration.'
        self._initialize_class_name_id_mapping()
        self._initialize_trainer_statistics(self.all_class_names)
        self.pretrained_model = pretrained_model

        self._is_in_grad_cam_trace_mode = False
        self._grad_cam_output_dir = ''

    def _on_epoch_start(self, epoch_num):
        self.stats.reset()
        if epoch_num == 1 and self.pretrained_model is not None:
            print('* Loading pretrained model from path "%s"...' % self.pretrained_model)
            if hasattr(self.model, "on_load_model_weights") == False:
                raise RuntimeError('Expecting model class implementing a member function named "on_load_model_weights" '
                                   'in order to handle custom model loading event, but this function is not found. '
                                   'Implement this function to enable pretrained weights loading if you want to train '
                                   'the model based on a pretrained model.')
            self.model.on_load_model_weights(self.pretrained_model)
            
    def _on_epoch(self, epoch, msg, data_loader, phase):
        self.current_phase = phase
        self._initialize_class_weights(data_loader)
        self.model.train() if phase=='train' else self.model.eval()
        self.stats.reset()
        timer = Timer()
        losses = []

        for batch_idx, (sample) in enumerate(data_loader):
            record, packed_image, lesion_volume, sex, age, posvec, radiomics_vec, real_class_name = sample # unpack
            assert packed_image.shape[0] == 1, 'Only support batch_size=1.'
            packed_image, lesion_volume, sex, age, posvec, radiomics_vec = \
                packed_image.cuda(self.gpu), lesion_volume.float().cuda(self.gpu), \
                sex.float().cuda(self.gpu), age.float().cuda(self.gpu), \
                posvec.float().cuda(self.gpu), radiomics_vec.float().cuda(self.gpu)
            
            
            self.dump_tensor_as_nifti(packed_image[0], "shit", name_prefix="123")
            print(record)
            
            
            class_id = torch.tensor(self._get_class_id_from_class_name(real_class_name[0])).long().cuda(self.gpu).reshape(1)
            printv(4, "MODEL INPUTS:\nvol:%s, sex:%s, age:%s, pv:%s, rv[:10]:%s, clsid:%s" % \
                (str(lesion_volume),str(sex),str(age),str(posvec),str(radiomics_vec[:10]), str(class_id)))
            if phase=='train':
                Y_hat = self.model(packed_image, 
                              sex = sex,
                              age = age,
                    lesion_volume = lesion_volume,
                           posvec = posvec,
                    radiomics_vec = radiomics_vec)
                # Y_hat: tensor(batch_size * classes), where batch_size=1 for simplicity
                self.optim.zero_grad()
                loss = nn.CrossEntropyLoss(weight=torch.Tensor(self.all_class_weights).float().cuda(self.gpu))
                L = loss(Y_hat, class_id)
                L.backward() 
                self.optim.step()
                losses.append(L.item())
                # dump tensor
                #self.dump_tensor_as_nifti(packed_image, './dump/', prefix='dump_%d_' % batch_idx)
            else:
                with torch.no_grad():
                    # b,c,x,y,z (b=1)
                    Y_hat = self.model(packed_image, 
                                  sex = sex,
                                  age = age,
                        lesion_volume = lesion_volume,
                               posvec = posvec,
                        radiomics_vec = radiomics_vec)
                    # Y_hat: tensor(batch_size * classes), where batch_size=1 for simplicity
                    class_id_pred = int(torch.argmax(Y_hat, dim=1).long().cpu().numpy())
                    pred_class = self._get_class_name_from_class_id(class_id_pred)
                    real_class = real_class_name[0]
                    L = 1 if real_class == pred_class else 0

                    probs = np.reshape(torch.softmax(Y_hat, dim=1).detach().cpu().numpy(), [len(self.all_class_names)])
                    class_probs = [(self._get_class_name_from_class_id(class_id), probs[class_id]) \
                        for class_id in range(len(self.all_class_names))]
                    class_probs_as_dict = {}
                    for clsname, clsprob in class_probs:
                        class_probs_as_dict[clsname] = clsprob
                    subject_name = record['subject_name'][0]
                    data_source = record['data_source'][0]
                    # save grad cam if needed
                    def _calculate_grad_cam_from_feature(raw_feature: np.ndarray, grad_cam_strategy = 'max'):
                        assert raw_feature.shape[0] == 1, 'batch size should be 1.'
                        assert grad_cam_strategy in ['min', 'max', 'average']
                        raw_feature_no_batch_dim = raw_feature.reshape(raw_feature.shape[1:]) # remove batch dim
                        grad_cam_feature = None
                        if grad_cam_strategy == 'max':
                            grad_cam_feature = np.max(raw_feature_no_batch_dim, axis=0, keepdims=False)
                        elif grad_cam_strategy == 'min':
                            grad_cam_feature = np.min(raw_feature_no_batch_dim, axis=0, keepdims=False)
                        elif grad_cam_strategy == 'average':
                            grad_cam_feature = np.mean(raw_feature_no_batch_dim, axis=0, keepdims=False)
                        assert grad_cam_feature is not None and grad_cam_feature.shape == raw_feature.shape[2:]
                        return grad_cam_feature
                    grad_cam_output = ''
                    if self._is_in_grad_cam_trace_mode and hasattr(self.model, "get_feature_for_grad_cam"):
                        feature_for_grad_cam: np.ndarray = self.model.get_feature_for_grad_cam()
                        if feature_for_grad_cam is not None:
                            grad_cam_feature = _calculate_grad_cam_from_feature(feature_for_grad_cam, 'max')
                            grad_cam_output_folder = mkdir(self._grad_cam_output_dir)
                            grad_cam_output = join_path(grad_cam_output_folder, '%s__%s__GRAD_CAM.nii.gz' % (data_source, subject_name))
                            # normalize
                            grad_cam_feature = (grad_cam_feature - np.min(grad_cam_feature)) / (np.max(grad_cam_feature) - np.min(grad_cam_feature))
                            # resample to original spatial channel size
                            input_channel_spatial_size = list(packed_image.shape[2:])
                            grad_cam_feature_shape = list(grad_cam_feature.shape)
                            assert len(input_channel_spatial_size) == len(grad_cam_feature_shape)
                            zoom_ratio = [ target/source for target, source in zip(input_channel_spatial_size, grad_cam_feature_shape) ]
                            grad_cam_feature = zoom(grad_cam_feature, zoom=zoom_ratio, order=1)
                            save_nifti_simple(grad_cam_feature, grad_cam_output)
                        else:
                            print('no feature grad cam')
                    # write record
                    self.stats.record(subject_name, data_source, class_probs_as_dict, real_class, grad_cam_output)

                losses.append(-L)
            minibar(msg, batch_idx+1, len(data_loader), timer.elapsed(), last='%.4f' % L)
        return np.mean(losses)

    def _on_epoch_end(self, epoch_num):
        def _make_confusion_matrix(all_class_names):
            d = {}
            for name1 in all_class_names:
                d[name1] = {}
                for name2 in all_class_names:
                    d[name1][name2] = 0
            return d
        def calculate_confusion_matrix(stat: PerformanceStatistics):
            num_records = stat.database.num_records()
            confusion_matrix = _make_confusion_matrix(self.all_class_names)
            for record_id in range(num_records):
                record = stat.database.get_record(record_id)
                pred_class_probs_as_dict: dict = eval(record['pred_class_probs'])
                real_class = record['real_class']
                pred_class = None
                pred_prob_max_ = -1.0
                for pred_class_, pred_prob_ in pred_class_probs_as_dict.items():
                    if pred_prob_max_ < pred_prob_:
                        pred_prob_max_ = pred_prob_
                        pred_class = pred_class_
                confusion_matrix[real_class][pred_class] += 1
            return confusion_matrix
        confusion_matrix = calculate_confusion_matrix(self.stats)
        print('* confusion_matrix:', confusion_matrix)
    
    def inference(self, test_loader, enable_tta = False, do_not_load_model=False):
        model_best = join_path(self.output_folder, 'model_best.model')
        if do_not_load_model == False:
            self.load_model(model_best)
        self.stats.reset()
        self._is_in_grad_cam_trace_mode = True

        xlsx_files = []

        # inference without tta
        if not file_exist(join_path(self.output_folder, 'final_statistics', 'without_tta','raw_probability.xlsx')):
            self._grad_cam_output_dir = join_path(self.output_folder, 'final_statistics', 'without_tta', 'grad_cam')
            self._on_epoch(0, 'Test Stage 1/2', test_loader, phase='test')
            print('')
            self.stats.save_to_xlsx(join_path(self.output_folder, 'final_statistics', 'without_tta','raw_probability.xlsx'))
        xlsx_files.append(join_path(self.output_folder, 'final_statistics', 'without_tta','raw_probability.xlsx'))

        if enable_tta:
            slice_offsets = [-4, -2, 0, +2, +4]
            for run_id, slice_offset in zip(range(1, len(slice_offsets) + 1), slice_offsets):
                if not file_exist(join_path(self.output_folder, 'final_statistics', 'with_tta', 'run_%d' % run_id, 'raw_probability.xlsx')):
                    self._grad_cam_output_dir = join_path(self.output_folder, 'final_statistics', 'with_tta', 'run_%d' % run_id, 'grad_cam')
                    test_loader.dataset.slice_offset = slice_offset
                    self._on_epoch(0,'Test Stage 2/2 %d/%d' % (run_id, len(slice_offsets)), test_loader, phase='test')
                    print('')
                    self.stats.save_to_xlsx(join_path(self.output_folder, 'final_statistics', 'with_tta', 'run_%d' % run_id, 'raw_probability.xlsx'))
            stats_ensembled = PerformanceStatistics(self.all_class_names)
            # load first run and add remaining runs
            stats_ensembled.load_from_xlsx(join_path(self.output_folder, 'final_statistics', 'with_tta', 'run_1', 'raw_probability.xlsx'))
            for run_id, slice_offset in zip(range(2, len(slice_offsets) + 1), slice_offsets[1:]): # skip first run
                stat = PerformanceStatistics(self.all_class_names)
                stat.load_from_xlsx(join_path(self.output_folder, 'final_statistics', 'with_tta', 'run_%d' % run_id, 'raw_probability.xlsx'))
                for record_id in range(stats_ensembled.database.num_records()):
                    record0 = stats_ensembled.database.get_record(record_id)
                    probs_dict0 = eval(record0['pred_class_probs'])
                    probs_dict1 = eval(stat.database.get_record(record_id)['pred_class_probs'])
                    for class_type in probs_dict0:
                        probs_dict0[class_type] += probs_dict1[class_type]
                    record0['pred_class_probs'] = str(probs_dict0)
                    stats_ensembled.database.set_record(record_id, record0)
            # average all runs
            for record_id in range(stats_ensembled.database.num_records()):
                record0 = stats_ensembled.database.get_record(record_id)
                probs_dict0 = eval(record0['pred_class_probs'])
                for class_type in probs_dict0:
                    probs_dict0[class_type] /= len(slice_offsets)
                record0['pred_class_probs'] = str(probs_dict0)
                stats_ensembled.database.set_record(record_id, record0)
            stats_ensembled.save_to_xlsx(join_path(self.output_folder, 'final_statistics', 'with_tta','raw_probability.xlsx'))
            xlsx_files.append(join_path(self.output_folder, 'final_statistics', 'with_tta','raw_probability.xlsx'))

        self._is_in_grad_cam_trace_mode = False

        # calculate lots of performance metrics here
        for xlsx_file in xlsx_files:
            perf_stats = PerformanceStatistics(self.all_class_names)
            perf_stats.load_from_xlsx(xlsx_file)
            perf_stats.measure_performance_binary(gd(xlsx_file))


class GenericRegressionTrainer(GenericClassifierTrainer):
    def __init__(self, 
        output_folder: str                                     = './out',
        gpu_index:     int                                     = 0, 
        model:         Union[torch.nn.Module, None]            = None,
        optim:         Union[torch.optim.Optimizer, str, None] = 'default',
        lr_scheduler:  Union[LRScheduler, str, None]           = 'default',
        train_loader:  Union[data.Dataset, None]               = None, 
        val_loader:    Union[data.Dataset, None]               = None, 
        test_loader:   Union[data.Dataset, None]               = None,
        #
        variable_name:                Union[str, None]         = None,
        pretrained_model:       Union[List[str], None]         = None,
    ):
        super(GenericClassifierTrainer, self).__init__(output_folder, gpu_index, model, optim, lr_scheduler, train_loader, val_loader, test_loader)
        self.variable_name = variable_name
        self.pretrained_model = pretrained_model
          
    def _on_epoch(self, epoch, msg, data_loader, phase):
        self.current_phase = phase
        self.model.train() if phase=='train' else self.model.eval()
        timer = Timer()
        losses = []
        for batch_idx, (sample) in enumerate(data_loader):
            record, packed_image, lesion_volume, sex, age, posvec, radiomics_vec, gt_variable = sample # unpack
            assert packed_image.shape[0] == 1, 'Only support batch_size=1.'
            packed_image, lesion_volume, sex, age, posvec, radiomics_vec = \
                packed_image.cuda(self.gpu), lesion_volume.float().cuda(self.gpu), \
                sex.float().cuda(self.gpu), age.float().cuda(self.gpu), \
                posvec.float().cuda(self.gpu), radiomics_vec.float().cuda(self.gpu)
            gt_variable = torch.Tensor([float(gt_variable[0])]).float().cuda(self.gpu)
            if phase=='train':
                Y_hat = self.model(packed_image, 
                              sex = sex,
                              age = age,
                    lesion_volume = lesion_volume,
                           posvec = posvec,
                    radiomics_vec = radiomics_vec)
                self.optim.zero_grad()
                loss = nn.MSELoss()
                L = loss(Y_hat, gt_variable)
                L.backward() 
                self.optim.step()
                losses.append(L.item())
            else:
                with torch.no_grad():
                    # b,c,x,y,z (b=1)
                    Y_hat = self.model(packed_image, 
                                  sex = sex,
                                  age = age,
                        lesion_volume = lesion_volume,
                               posvec = posvec,
                        radiomics_vec = radiomics_vec)
                    loss = nn.MSELoss()
                    L = loss(Y_hat, gt_variable)
                    losses.append(L.item())
            minibar(msg, batch_idx+1, len(data_loader), timer.elapsed(), last='%.4f' % L)
        return np.mean(losses)


    def _on_epoch_start(self, epoch_num):
        if epoch_num == 1 and self.pretrained_model is not None:
            print('* Loading pretrained model from path "%s"...' % self.pretrained_model)
            if hasattr(self.model, "on_load_model_weights") == False:
                raise RuntimeError('Expecting model class implementing a member function named "on_load_model_weights" '
                                   'in order to handle custom model loading event, but this function is not found. '
                                   'Implement this function to enable pretrained weights loading if you want to train '
                                   'the model based on a pretrained model.')
            self.model.on_load_model_weights(self.pretrained_model)
            

    def _on_epoch_end(self, epoch_num):
        pass

    def inference(self, test_loader, enable_tta=False):
        pass
