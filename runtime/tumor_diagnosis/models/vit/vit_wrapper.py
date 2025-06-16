import torch
import torch.nn as nn
from digicare.runtime.tumor_diagnosis.models._layers import _PredictionBranch
from digicare.runtime.tumor_diagnosis.models.vit.vit import ViT
from digicare.runtime.tumor_diagnosis.models.vit.vit_v2 import build_vit_google_base_patch32_224
from digicare.runtime.tumor_diagnosis.models.vit.vit3d_pytorch import ViT3D

class VitWrapper2D(nn.Module):
    def __init__(self, in_channels, out_classes, posvec_len, radiomics_vec_len, vit_model_arch,load_pretrained_model = False):
        super().__init__()
        self.in_channels = in_channels
        self.posvec_len = posvec_len
        self.radiomics_vec_len = radiomics_vec_len
        self.vit_model_arch = vit_model_arch
        if self.vit_model_arch == 'vit':
            self.vit_out_classes = 1024
        elif self.vit_model_arch == 'vit_v2':
            self.vit_out_classes = 768
        else:
            raise RuntimeError('invalid vit output classes, shoule be one of "vit" or "vit_v2".')
        self.pred_layer = _PredictionBranch(self.vit_out_classes, self.posvec_len, 3, radiomics_vec_len, out_classes)
        self.load_pretrained_model = load_pretrained_model
        if self.vit_model_arch == 'vit':
            self.vit_model = ViT(image_size=256, channels=self.in_channels, num_classes=self.vit_out_classes)
        elif self.vit_model_arch == 'vit_v2':
            self.vit_model = build_vit_google_base_patch32_224(image_size=256, channels=self.in_channels)
        else:
            raise RuntimeError('Cannot build ViT 2D/2.5D model for model_arch="%s". No implementation given.' % \
                self.vit_model_arch)

    def on_load_model_weights(self, model_path):
        
        state_dict = self.vit_model.state_dict()
        
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

        self.vit_model.load_state_dict(state_dict, strict=True)
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

        x0 = self.vit_model(x) # x0 : [batch_size, channels]
        if self.vit_model_arch == 'vit_v2':
            x0 = x0.pooler_output
        x0 = torch.reshape(x0, [batch_size, self.vit_out_classes])

        sex = sex.reshape([batch_size, 1])
        age = age.reshape([batch_size, 1])
        lesion_volume = lesion_volume.reshape([batch_size, 1])
        
        sex_age_volume = torch.cat((sex, age, lesion_volume), dim=1)
        posvec = posvec.reshape([batch_size, self.posvec_len])
        radiomics_vec = radiomics_vec.reshape([batch_size, self.radiomics_vec_len])

        y_pred = self.pred_layer(x0, radiomics_vec, posvec, sex_age_volume)
        return y_pred


class VitWrapper3D(nn.Module):
    def __init__(self, in_channels, out_classes, posvec_len, radiomics_vec_len, vit_model_arch,load_pretrained_model = False):
        super().__init__()
        self.in_channels = in_channels
        self.posvec_len = posvec_len
        self.radiomics_vec_len = radiomics_vec_len
        self.vit_model_arch = vit_model_arch
        self.vit_out_classes = 1024
        self.pred_layer = _PredictionBranch(self.vit_out_classes, self.posvec_len, 3, radiomics_vec_len, out_classes)
        self.load_pretrained_model = load_pretrained_model
        if self.vit_model_arch == 'vit':
            self.vit_model = ViT3D(image_size=(128,128,128),patch_size=8,num_classes=1024, dim=1024,depth=4, heads=4, mlp_dim=1024, channels=in_channels)
        else:
            raise RuntimeError('Cannot build ViT 3D model for model_arch="%s". No implementation given.' % \
                self.vit_model_arch)

    def on_load_model_weights(self, model_path):
        
        state_dict = self.vit_model.state_dict()
        
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

        self.vit_model.load_state_dict(state_dict, strict=True)
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
        assert x.shape[2] == 128 and x.shape[3] == 128 and x.shape[4] == 128, 'expected input to have shape [~,~,128,128,128], got %s.' % str(x.shape)
        batch_size = x.shape[0]

        x0 = self.vit_model(x) # x0 : [batch_size, channels]
        x0 = torch.reshape(x0, [batch_size, self.vit_out_classes])

        sex = sex.reshape([batch_size, 1])
        age = age.reshape([batch_size, 1])
        lesion_volume = lesion_volume.reshape([batch_size, 1])
        
        sex_age_volume = torch.cat((sex, age, lesion_volume), dim=1)
        posvec = posvec.reshape([batch_size, self.posvec_len])
        radiomics_vec = radiomics_vec.reshape([batch_size, self.radiomics_vec_len])

        y_pred = self.pred_layer(x0, radiomics_vec, posvec, sex_age_volume)
        return y_pred
    
    # def forward(self, 
    #     x,                     # input images [b, in_channels, H, W] 
    #     sex = None,            # sex: list of floats [b]
    #     age = None,            # age: list of floats [b]
    #     lesion_volume = None,  # lesion total volume (voxels)
    #     posvec = None,         # positional vector list of floats [posvec_len]
    #     radiomics_vec = None   # radiomics features
    # ):
    #     assert len(x.shape) == 4, 'expected 4D input, got %s.' % str(x.shape)
    #     print('>>> Running patched VitWrapper2D forward(), input shape:', x.shape)
    #     # 自动 resize 到 256x256
    #     if x.shape[2] != 256 or x.shape[3] != 256:
    #         x = nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
    #     print('>>> Running patched VitWrapper2D forward(), input shape:', x.shape)

    #     batch_size = x.shape[0]

    #     x0 = self.vit_model(x)  # x0: [batch_size, features]
    #     if self.vit_model_arch == 'vit_v2':
    #         x0 = x0.pooler_output
    #     x0 = torch.reshape(x0, [batch_size, self.vit_out_classes])

    #     sex = sex.reshape([batch_size, 1])
    #     age = age.reshape([batch_size, 1])
    #     lesion_volume = lesion_volume.reshape([batch_size, 1])

    #     sex_age_volume = torch.cat((sex, age, lesion_volume), dim=1)
    #     posvec = posvec.reshape([batch_size, self.posvec_len])
    #     radiomics_vec = radiomics_vec.reshape([batch_size, self.radiomics_vec_len])

    #     y_pred = self.pred_layer(x0, radiomics_vec, posvec, sex_age_volume)
    #     return y_pred

