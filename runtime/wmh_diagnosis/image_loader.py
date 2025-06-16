import numpy as np
import scipy
import torch
from torch.utils import data
from copy import deepcopy
from scipy.ndimage import zoom as zoom_image
from digicare.utilities.data_io import load_nifti_simple, get_nifti_pixdim
from digicare.utilities.image_ops import barycentric_coordinate, center_crop, z_score
from digicare.utilities.database import Database

def load_volume_cube256_1mm(image):
    volume = load_nifti_simple(image)
    if volume.shape[0] != 256 or volume.shape[1] != 256 or volume.shape[2] != 256:
        raise RuntimeError('invalid volume shape %s, expected a 256x256x256 volume.' % str(volume.shape))
    pixdim = get_nifti_pixdim(image)
    if any([not(0.99 < pixdim[0] < 1.01), not(0.99 < pixdim[1] < 1.01), not(0.99 < pixdim[2] < 1.01)]):
        # set 1% tolerance, if pixdim still out of bound then raise error
        raise RuntimeError('invalid volume physical resolution, expected a 1mm^3 resolution but got %s.' % str(pixdim))
    return volume

class ImageLoader_Patch3D_Base(data.Dataset):
        
    def __init__(self, database: Database, config = None):
        super().__init__()
        if config is None:
            config = ImageLoader_Patch3D_Base.default_loading_config()
        assert self._check_database(database), 'Invalid database structure.'        
        assert self._check_config(config), 'Invalid trainer config.'
        self.database = deepcopy(database)
        self.config = deepcopy(config)
        if self.config['random_sampling']:
            print('Random sampling ON.')

    @staticmethod
    def default_loading_config():
        return {
            'patch_size': [128,128,128],
            'enable_data_augmentation': False,
            'random_sampling': True,
            'num_batches_per_epoch': 200,
            'data_augmentation_config':{
                'enable_noise': True,
                'noise_prob' : 0.2,
                'enable_mirror': True,
                'mirror_prob' : 0.3,
                'enable_rotation': True,
                'rotation_prob': 0.5,
            },
        }

    def _check_database(self, database: Database):
        expected_keys = [
            'FLAIR',
            'anomaly_map',
            'brain_mask',
            'diagnosis'
        ]
        for key in expected_keys:
            if key not in database.db_keys:
                print('key "%s" not found.' % key)
                return False
        return True

    def _check_config(self, config: dict):
        if 'patch_size' not in config or \
            config['patch_size'][0] != 128 or config['patch_size'][1] != 128 or config['patch_size'][2] != 128:
            return False
        return True

    def _generate_data_augmentation_param(self):
        def _should_apply(prob):
            return True if np.random.rand() < prob else False
        def _generate_mirror_axis():
            mirror_axis = ''
            for _ in range(0,3):
                mirror_axis += 'xyz-'[np.random.randint(0,4)]
            return mirror_axis
        param={
            'enable_noise':    True if self.config['data_augmentation_config']['enable_noise']    and _should_apply(self.config['data_augmentation_config']['noise_prob'])    else False,
            'enable_mirror':   True if self.config['data_augmentation_config']['enable_mirror']   and _should_apply(self.config['data_augmentation_config']['mirror_prob'])   else False,
            'enable_rotation': True if self.config['data_augmentation_config']['enable_rotation'] and _should_apply(self.config['data_augmentation_config']['rotation_prob']) else False,
            'mirror_axis': _generate_mirror_axis(),
            'rotation_deg_xyz': list(np.random.randint(0,30,size=[3])),
        }
        return param

    def _apply_data_augmentation_to_volume(self, volume, aug_params, apply_intensity_transform=True, apply_spatial_transform=True):
        if aug_params['enable_rotation'] and apply_spatial_transform:
            for axis_id, rot_axes in zip([0,1,2], [(1,2),(0,2),(0,1)]):
                volume = scipy.ndimage.interpolation.rotate(
                    volume, 
                    aug_params['rotation_deg_xyz'][axis_id], axes=rot_axes, 
                    reshape=False, order=1, mode='constant', 
                    cval=np.min(volume), prefilter=True)
        if aug_params['enable_mirror'] and apply_spatial_transform:
            for axis_id in range(3):
                mirror_axis = aug_params['mirror_axis'][axis_id]
                if mirror_axis == 'x': volume = np.flip(volume, 0)
                elif mirror_axis == 'y': volume = np.flip(volume, 1)
                elif mirror_axis == 'z': volume = np.flip(volume, 2)
        if aug_params['enable_noise'] and apply_intensity_transform:
            q5 = np.percentile(volume, 5)
            q95 = np.percentile(volume, 95)
            noise = np.random.normal(scale = 0.05 * (q95-q5),size=volume.shape)
            volume += noise
        # after augmentation it is possible that data is mirrored, and 
        # since scipy implements the mirroring operation just by simply 
        # negate the stride of the raw array, this can cause compatibility
        # problems for pytorch, so as a work around we need to use 
        # array.copy() to let numpy 'bake' the transform into the volume.
        volume = volume.copy() # this will bake all the transforms
        return volume
        
    def _generate_downsample_for_deep_supervision(self, x, levels):
        '''
        Downsampling image for deep supervision, returns a list with levels+1 items.
        Each level's down sampling ratio is 2x.
        '''
        data_for_DS = [x]
        zoom_ratio = 0.5
        for level in range(levels):
            data_for_DS.append(zoom_image(x, zoom_ratio, order=1))
            zoom_ratio /= 2.0
        return data_for_DS

    def __len__(self):
        if self.config['random_sampling']:
            return self.config['num_batches_per_epoch']
        else:
            return self.database.num_records()
    
    def __getitem__(self, index):
        raise RuntimeError('Unimplemented method __getitem__(self, index) called.')

    def map_class_id_to_name(self, id:int):
        for key_, id_ in self.config['class_mapping'].items():
            if id_ == id:
                return key_
            continue
        raise RuntimeError('Unknown class ID: "%d".' % id)

    def map_class_name_to_id(self, class_name:str):
        if class_name not in self.config['class_mapping']:
            raise RuntimeError('Unknown class name: "%s", class_mapping is: "%s".' % (class_name, str(self.config['class_mapping'])))
        return self.config['class_mapping'][class_name]


    def get_weight_of_each_class(self):
        def _sample_ratio_to_sample_weight(ratios):
            ratios = torch.Tensor(ratios).float()
            ratios = ratios / ratios.sum()
            weights = 1.0 / ratios
            weights = weights / weights.sum()
            return weights
        sample_num_of_each_class = {}
        for class_name in self.config['class_mapping']:
            sample_num_of_each_class[class_name] = 0
        for sample_id in range(self.database.num_records()):
            record = self.database.get_record(sample_id)
            diagnosis = record['diagnosis']
            if diagnosis not in self.config['class_mapping']:
                raise RuntimeError('Unknown diagnosis: "%s".' % diagnosis)
            sample_num_of_each_class[diagnosis] += 1
        all_classes = list(self.config['class_mapping'].keys())
        l = []
        force_uniform = False
        for class_id in range(len(all_classes)):
            class_name = self.map_class_id_to_name(class_id)
            if sample_num_of_each_class[class_name] == 0:
                print('dataset does not have any sample belong to class: "%s", using uniform weight.' % class_name)
                force_uniform = True
            l.append(sample_num_of_each_class[class_name])
        sample_weights = _sample_ratio_to_sample_weight( l ) if not force_uniform else _sample_ratio_to_sample_weight([1.0] * len(l)) 
        #print(sample_num_of_each_class, sample_weights)
        return sample_weights


class ImageLoader_Patch3D_Cube128_1p5mm(ImageLoader_Patch3D_Base):
    
    def __init__(self, database: Database, config = None):
        super().__init__(database, config)

    def __getitem__(self, index):
        sample_id = index
        if self.config['random_sampling']:
            sample_id = np.random.randint(0, self.database.num_records())
        record = self.database.get_record(sample_id)
        diagnosis = record['diagnosis']
        # acquire image paths and load images
        FLAIR_image = record['FLAIR']
        brain_mask = record['brain_mask']
        anomaly_map = record['anomaly_map']

        # images should be pre-processed before, assuming all images have been
        # resampled to 1mm^3 isotropic resolution
        x = load_volume_cube256_1mm(FLAIR_image)
        m = load_volume_cube256_1mm(brain_mask)
        a = load_volume_cube256_1mm(anomaly_map)

        # crop to 192x192x192 volume, and resize it to 128x128x128 afterwards to get 1.5mm physical resolution
        x = center_crop(x, [128,128,128], [192,192,192], default_fill=np.min(x))
        m = center_crop(m, [128,128,128], [192,192,192], default_fill=0.0)
        a = center_crop(a, [128,128,128], [192,192,192], default_fill=0.0)
        a = np.where(a > 10.0, 10.0, a)
        
        # binarize mask, in case encountering abnormal values
        m = np.where(m > 0.5, 1.0, 0.0)
        # z-score normalization, this must be done
        x = z_score(x, m)
        # zoom to 128x128x128
        x = zoom_image(x, 128/192, order=1)
        m = zoom_image(m, 128/192, order=1)
        a = zoom_image(a, 128/192, order=1)
        assert x.shape[0] == 128, 'we encountered numerical problem here!'
        # calculate center position and centering all images
        cpos = barycentric_coordinate(m)
        cpos = [int(cpos[0]), int(cpos[1]), int(cpos[2])]
        x = center_crop( x, cpos, self.config['patch_size'], np.min(x))
        m = center_crop( m, cpos, self.config['patch_size'], 0.0)
        a = center_crop( a, cpos, self.config['patch_size'], 0.0)

        if self.config['enable_data_augmentation']:
            data_augmentation_param = self._generate_data_augmentation_param()
            x = self._apply_data_augmentation_to_volume(x, data_augmentation_param)
            m = self._apply_data_augmentation_to_volume(m, data_augmentation_param, apply_intensity_transform=False)
            a = self._apply_data_augmentation_to_volume(a, data_augmentation_param, apply_intensity_transform=False)
        
        x_DS = self._generate_downsample_for_deep_supervision(x, 4)
        a_DS = self._generate_downsample_for_deep_supervision(a, 4)
        m_DS = self._generate_downsample_for_deep_supervision(m, 4)

        # wrap an extra channel dim for all inputs.
        for i in range(len(x_DS)):
            x_DS[i] = np.reshape(x_DS[i], [1, *x_DS[i].shape])
        for i in range(len(a_DS)):
            a_DS[i] = np.reshape(a_DS[i], [1, *a_DS[i].shape])
        for i in range(len(m_DS)):
            m_DS[i] = np.reshape(m_DS[i], [1, *m_DS[i].shape])
        
        return record, x_DS, m_DS, a_DS, diagnosis


class ImageLoader_Patch3D_Cube128_LVSHAN(ImageLoader_Patch3D_Base):
    
    def __init__(self, database: Database, config = None):
        super().__init__(database, config)

    def __getitem__(self, index):
        sample_id = index
        if self.config['random_sampling']:
            sample_id = np.random.randint(0, self.database.num_records())
        record = self.database.get_record(sample_id)
        diagnosis = record['diagnosis']
        # acquire image paths and load images
        FLAIR_image = record['FLAIR']
        brain_mask = record['brain_mask']

        # images should be pre-processed before, assuming all images have been
        # resampled to 1mm^3 isotropic resolution
        x = load_volume_cube256_1mm(FLAIR_image)
        m = load_volume_cube256_1mm(brain_mask)

        # crop to 192x192x192 volume, and resize it to 128x128x128 afterwards to get 1.5mm physical resolution
        x = center_crop(x, [128,128,128], [192,192,192], default_fill=np.min(x))
        m = center_crop(m, [128,128,128], [192,192,192], default_fill=0.0)
        
        # binarize mask, in case encountering abnormal values
        m = np.where(m > 0.5, 1.0, 0.0)
        # z-score normalization, this must be done
        x = z_score(x, m)
        # zoom to 128x128x128
        x = zoom_image(x, 128/192, order=1)
        m = zoom_image(m, 128/192, order=1)
        assert x.shape[0] == 128, 'we encountered numerical problem here!'
        # calculate center position and centering all images
        cpos = barycentric_coordinate(m)
        cpos = [int(cpos[0]), int(cpos[1]), int(cpos[2])]
        x = center_crop( x, cpos, self.config['patch_size'], np.min(x))
        m = center_crop( m, cpos, self.config['patch_size'], 0.0)

        if self.config['enable_data_augmentation']:
            data_augmentation_param = self._generate_data_augmentation_param()
            x = self._apply_data_augmentation_to_volume(x, data_augmentation_param)
            m = self._apply_data_augmentation_to_volume(m, data_augmentation_param, apply_intensity_transform=False)
        
        x_DS = self._generate_downsample_for_deep_supervision(x, 4)
        m_DS = self._generate_downsample_for_deep_supervision(m, 4)

        # wrap an extra channel dim for all inputs.
        for i in range(len(x_DS)):
            x_DS[i] = np.reshape(x_DS[i], [1, *x_DS[i].shape])
        for i in range(len(m_DS)):
            m_DS[i] = np.reshape(m_DS[i], [1, *m_DS[i].shape])
        
        return record, x_DS, m_DS, diagnosis
