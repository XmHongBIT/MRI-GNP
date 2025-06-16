import torch
import torch.nn as nn
from digicare.runtime.tumor_diagnosis.models._layers import _PredictionBranch
from digicare.runtime.tumor_diagnosis.models.resnet.resnet2d import resnet18 as resnet18_2d
from digicare.runtime.tumor_diagnosis.models.resnet.resnet2d import resnet34 as resnet34_2d
from digicare.runtime.tumor_diagnosis.models.resnet.resnet2d import resnet50 as resnet50_2d
from digicare.runtime.tumor_diagnosis.models.resnet.resnet2d import resnet101 as resnet101_2d
from digicare.runtime.tumor_diagnosis.models.resnet.resnet2d import resnet152 as resnet152_2d
from digicare.runtime.tumor_diagnosis.models.resnet.resnet3d import \
    resnet10_3d, resnet18_3d, resnet34_3d, resnet50_3d, resnet101_3d, resnet152_3d, resnet200_3d

class ResNetWrapper2D(nn.Module):
    def __init__(self, in_channels, out_classes, posvec_len, radiomics_vec_len, resnet_model_arch):
        super().__init__()
        self.in_channels = in_channels
        self.posvec_len = posvec_len
        self.radiomics_vec_len = radiomics_vec_len
        self.resnet_out_classes = 1024
        self.pred_layer = _PredictionBranch(self.resnet_out_classes, self.posvec_len, 3, radiomics_vec_len, out_classes)
        self.resnet_model_arch = resnet_model_arch
        if self.resnet_model_arch == 'resnet18':
            self.resnet_model = resnet18_2d(in_channels=self.in_channels, num_classes=self.resnet_out_classes)
        elif self.resnet_model_arch == 'resnet34':
            self.resnet_model = resnet34_2d(in_channels=self.in_channels, num_classes=self.resnet_out_classes)
        elif self.resnet_model_arch == 'resnet50':
            self.resnet_model = resnet50_2d(in_channels=self.in_channels, num_classes=self.resnet_out_classes)
        elif self.resnet_model_arch == 'resnet101':
            self.resnet_model = resnet101_2d(in_channels=self.in_channels, num_classes=self.resnet_out_classes)
        elif self.resnet_model_arch == 'resnet152':
            self.resnet_model = resnet152_2d(in_channels=self.in_channels, num_classes=self.resnet_out_classes)
        else:
            raise RuntimeError('Cannot build resnet 2D/2.5D model for model_arch="%s". No implementation given.' % \
                self.resnet_model_arch)
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

        x0 = self.resnet_model(x) # x0 : [batch_size, channels]
        x0 = torch.reshape(x0, [batch_size, self.resnet_out_classes])

        sex = sex.reshape([batch_size, 1])
        age = age.reshape([batch_size, 1])
        lesion_volume = lesion_volume.reshape([batch_size, 1])
        
        sex_age_volume = torch.cat((sex, age, lesion_volume), dim=1)
        posvec = posvec.reshape([batch_size, self.posvec_len])
        radiomics_vec = radiomics_vec.reshape([batch_size, self.radiomics_vec_len])

        y_pred = self.pred_layer(x0, radiomics_vec, posvec, sex_age_volume)
        return y_pred

    def on_load_model_weights(self, model_path):
        
        state_dict = self.resnet_model.state_dict()
        
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

        self.resnet_model.load_state_dict(state_dict, strict=True)
        print('* Loaded param(s): %s' % str(list(param_dict.keys())))
        print('* Ignored param(s) due to non-existence: %s' % str(ignore_not_exist))
        print('* Ignored param(s) due to shape mismatch: %s' % str(ignore_mismatch))

class ResNetWrapper3D(nn.Module):
    def __init__(self, in_channels, out_classes, posvec_len, radiomics_vec_len, resnet_model_arch):
        super().__init__()
        self.in_channels = in_channels
        self.posvec_len = posvec_len
        self.radiomics_vec_len = radiomics_vec_len
        self.resnet_out_classes = 32
        self.pred_layer = _PredictionBranch(1024, self.posvec_len, 3, radiomics_vec_len, out_classes)
        self.resnet_model_arch = resnet_model_arch
        if self.resnet_model_arch == 'resnet10':
            self.resnet_model = resnet10_3d(in_channels=self.in_channels, num_classes=self.resnet_out_classes)
        elif self.resnet_model_arch == 'resnet18':
            self.resnet_model = resnet18_3d(in_channels=self.in_channels, num_classes=self.resnet_out_classes)
        elif self.resnet_model_arch == 'resnet34':
            self.resnet_model = resnet34_3d(in_channels=self.in_channels, num_classes=self.resnet_out_classes)
        elif self.resnet_model_arch == 'resnet50':
            self.resnet_model = resnet50_3d(in_channels=self.in_channels, num_classes=self.resnet_out_classes)
        elif self.resnet_model_arch == 'resnet101':
            self.resnet_model = resnet101_3d(in_channels=self.in_channels, num_classes=self.resnet_out_classes)
        elif self.resnet_model_arch == 'resnet152':
            self.resnet_model = resnet152_3d(in_channels=self.in_channels, num_classes=self.resnet_out_classes)
        elif self.resnet_model_arch == 'resnet200':
            self.resnet_model = resnet200_3d(in_channels=self.in_channels, num_classes=self.resnet_out_classes)
        else:
            raise RuntimeError('Cannot build resnet 3D model for model_arch="%s". No implementation given.' % \
                self.resnet_model_arch)
        self.resnet_downsamp = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=True), 
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=True), 
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=True), 
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 512, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=True), 
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 1024, kernel_size=(2, 2, 2), stride=(2, 2, 2), bias=True), 
        )
    def forward(self, 
        x,                     # input images [b, in_channels, 256, 256] 
        sex = None,            # sex: list of floats [b]
        age = None,            # age: list of floats [b]
        lesion_volume = None,  # lesion total volume (voxels)
        posvec = None,         # positional vector list of floats [posvec_len]
        radiomics_vec = None   # radiomics features
    ):
        # the forward pass of resnet3d is a bit different than 2d,
        # the default output spatial dimension is 32x32x32
        assert len(x.shape) == 5, 'expected 5D input, got %s.' % str(x.shape)
        assert x.shape[2] == 128, 'expected input to have shape [~,~,128,128,128], got %s.' % str(x.shape)
        assert x.shape[3] == 128, 'expected input to have shape [~,~,128,128,128], got %s.' % str(x.shape)
        assert x.shape[4] == 128, 'expected input to have shape [~,~,128,128,128], got %s.' % str(x.shape)
        batch_size = x.shape[0]


        x0 = self.resnet_model(x)
        x1 = self.resnet_downsamp(x0)
        x1 = torch.reshape(x1, [batch_size, 1024])

        sex = sex.reshape([batch_size, 1])
        age = age.reshape([batch_size, 1])
        lesion_volume = lesion_volume.reshape([batch_size, 1])
        
        sex_age_volume = torch.cat((sex, age, lesion_volume), dim=1)
        posvec = posvec.reshape([batch_size, self.posvec_len])
        radiomics_vec = radiomics_vec.reshape([batch_size, self.radiomics_vec_len])

        y_pred = self.pred_layer(x1, radiomics_vec, posvec, sex_age_volume)
        return y_pred

