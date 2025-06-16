import torch
import torch.nn as nn
from digicare.runtime.tumor_diagnosis.models._layers import _PredictionBranch
from digicare.runtime.tumor_diagnosis.models.squeezenet.squeezenet import squeezenet1_0 as squeezenet10_2d
from digicare.runtime.tumor_diagnosis.models.squeezenet.squeezenet import squeezenet1_1 as squeezenet11_2d
from digicare.runtime.tumor_diagnosis.models.squeezenet.squeezenet import squeezenet1_1_gc as squeezenet11_2d_gradcam
from digicare.runtime.tumor_diagnosis.models.squeezenet.squeezenet3d import SqueezeNet3D

import numpy as np

class SqueezeNetWrapper2D(nn.Module):
    def __init__(self, in_channels, out_classes, posvec_len, radiomics_vec_len, model_arch):
        super().__init__()
        self.in_channels = in_channels
        self.posvec_len = posvec_len
        self.radiomics_vec_len = radiomics_vec_len
        self.squeezenet_out_classes = 1024
        self.pred_layer = _PredictionBranch(self.squeezenet_out_classes, self.posvec_len, 3, radiomics_vec_len, out_classes)
        self.model_arch = model_arch
        if self.model_arch == 'squeezenet_v1.0':
            self.model = squeezenet10_2d(in_channels=self.in_channels, num_classes=self.squeezenet_out_classes)
        elif self.model_arch == 'squeezenet_v1.1':
            self.model = squeezenet11_2d(in_channels=self.in_channels, num_classes=self.squeezenet_out_classes)
        elif self.model_arch == 'squeezenet_v1.1_gradcam':
            self.model = squeezenet11_2d_gradcam(in_channels=self.in_channels, num_classes=self.squeezenet_out_classes)
        else:
            raise RuntimeError('Cannot build squeezenet 2D/2.5D model for model_arch="%s". No implementation given.' % \
                self.model_arch)

    def on_load_model_weights(self, model_path):
        
        state_dict = self.model.state_dict()
        
        loaded_dict = torch.load(model_path)
        param_dict = {}
        ignore_not_exist = []
        ignore_mismatch = []
        for name, param in loaded_dict.items():
            if isinstance(param, torch.Tensor):
                if name in state_dict: # name match
                    if param.shape == state_dict[name].shape: # shape also match
                        param_dict[name] = param
                    else: # name matched, but shape does not
                        ignore_mismatch.append(name)
                else: # name does not match
                    ignore_not_exist.append(name)
        state_dict.update(param_dict)        

        self.model.load_state_dict(state_dict, strict=True)
        print('* Loaded param(s): %s' % str(list(param_dict.keys())))
        print('* Ignored param(s) due to non-existence: %s' % str(ignore_not_exist))
        print('* Ignored param(s) due to shape mismatch: %s' % str(ignore_mismatch))

    def forward(self, 
        x,                     # input images [b, in_channels, 256, 256] 
        sex = None,            # sex: list of floats [b]
        age = None,            # age: list of floats [b]
        lesion_volume = None,  # lesion total volume (voxels)
        posvec = None,         # positional vector list of floats [posvec_len]
        radiomics_vec = None   # radiomics features
    ):
        assert len(x.shape) == 4, 'expected 4D input, got %s.' % str(x.shape)
        assert x.shape[2] == 256, 'expected input to have shape [~,~,256,256], got %s.' % str(x.shape)
        assert x.shape[3] == 256, 'expected input to have shape [~,~,256,256], got %s.' % str(x.shape)
        batch_size = x.shape[0]

        x0 = self.model(x) # x0 : [batch_size, channels]
        x0 = torch.reshape(x0, [batch_size, self.squeezenet_out_classes])

        sex = sex.reshape([batch_size, 1])
        age = age.reshape([batch_size, 1])
        lesion_volume = lesion_volume.reshape([batch_size, 1])
        
        sex_age_volume = torch.cat((sex, age, lesion_volume), dim=1)
        posvec = posvec.reshape([batch_size, self.posvec_len])
        radiomics_vec = radiomics_vec.reshape([batch_size, self.radiomics_vec_len])

        y_pred = self.pred_layer(x0, radiomics_vec, posvec, sex_age_volume)
        return y_pred

    def get_feature_for_grad_cam(self) -> np.ndarray:
        return self.model.get_feature_for_grad_cam()

class SqueezeNetWrapper3D(nn.Module):
    def __init__(self, in_channels, out_classes, posvec_len, radiomics_vec_len, model_arch):
        super().__init__()
        self.in_channels = in_channels
        self.posvec_len = posvec_len
        self.radiomics_vec_len = radiomics_vec_len
        self.squeezenet_out_classes = 1024
        self.pred_layer = _PredictionBranch(self.squeezenet_out_classes, self.posvec_len, 3, radiomics_vec_len, out_classes)
        self.model_arch = model_arch
        if self.model_arch == 'squeezenet_v1.1':
            self.model = SqueezeNet3D(112, 16, num_classes = self.squeezenet_out_classes//8, n_input_channels=in_channels)
        else:
            raise RuntimeError('Cannot build squeezenet 3D model for model_arch="%s". No implementation given.' % \
                self.model_arch)

    def on_load_model_weights(self, model_path):
        
        state_dict = self.model.state_dict()
        
        loaded_dict = torch.load(model_path)
        param_dict = {}
        ignore_not_exist = []
        ignore_mismatch = []
        for name, param in loaded_dict.items():
            if isinstance(param, torch.Tensor):
                if name in state_dict: # name match
                    if param.shape == state_dict[name].shape: # shape also match
                        param_dict[name] = param
                    else: # name matched, but shape does not
                        ignore_mismatch.append(name)
                else: # name does not match
                    ignore_not_exist.append(name)
        state_dict.update(param_dict)        

        self.model.load_state_dict(state_dict, strict=True)
        print('* Loaded param(s): %s' % str(list(param_dict.keys())))
        print('* Ignored param(s) due to non-existence: %s' % str(ignore_not_exist))
        print('* Ignored param(s) due to shape mismatch: %s' % str(ignore_mismatch))

    def forward(self, 
        x,                     # input images [b, in_channels, 256, 256] 
        sex = None,            # sex: list of floats [b]
        age = None,            # age: list of floats [b]
        lesion_volume = None,  # lesion total volume (voxels)
        posvec = None,         # positional vector list of floats [posvec_len]
        radiomics_vec = None   # radiomics features
    ):
        assert len(x.shape) == 5, 'expected 5D input, got %s.' % str(x.shape)
        assert x.shape[2] == 128 and x.shape[3] == 128 and x.shape[4] == 128, \
            'expected input to have shape [~,~,128,128,128], got %s.' % str(x.shape)
        batch_size = x.shape[0]

        x0 = self.model(x) # x0 : [batch_size, channels]
        x0 = torch.reshape(x0, [batch_size, self.squeezenet_out_classes])

        sex = sex.reshape([batch_size, 1])
        age = age.reshape([batch_size, 1])
        lesion_volume = lesion_volume.reshape([batch_size, 1])
        
        sex_age_volume = torch.cat((sex, age, lesion_volume), dim=1)
        posvec = posvec.reshape([batch_size, self.posvec_len])
        radiomics_vec = radiomics_vec.reshape([batch_size, self.radiomics_vec_len])

        y_pred = self.pred_layer(x0, radiomics_vec, posvec, sex_age_volume)
        return y_pred
