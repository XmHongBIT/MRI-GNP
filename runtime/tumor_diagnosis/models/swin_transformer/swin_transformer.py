import timm
import torch.nn as nn
import torch
from digicare.runtime.tumor_diagnosis.models._layers import _PredictionBranch
from digicare.runtime.tumor_diagnosis.models.swin_transformer.swin3d_layer import SwinTransformer

def create_swin_transformer_model(in_channels=2, out_classes=None):
    model = timm.create_model('swinv2_tiny_window16_256', num_classes=out_classes, pretrained_cfg_overlay=None)
    # modify model input channel if in_channels != 3
    if in_channels != 3:
        old_weight = model.patch_embed.proj.weight.data
        new_weight = torch.zeros([old_weight.size(0), in_channels, old_weight.size(2), old_weight.size(3)])
        new_weight[:,  :3, :, :] = old_weight
        new_weight[:, 3:5, :, :] = old_weight[:, 0:2, :, :]
        model.patch_embed.proj.weight = torch.nn.Parameter(new_weight)
        model.patch_embed.proj.in_channels = in_channels
    return model

def create_swin_transformer_3d_model(in_channels=2):
    model = SwinTransformer(in_chans=in_channels,
        embed_dim=96,
        window_size=(8,8,8),
        patch_size=(16,16,16),
        depths=[2,2,6,2],
        num_heads=[3,6,12,24],
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        spatial_dims=3,
        downsample='merging',
        use_v2=False)
    return model

class SwinTransformerWrapper2D(nn.Module):
    def __init__(self, in_channels, out_classes, posvec_len, radiomics_vec_len, model_arch):
        super().__init__()
        self.in_channels = in_channels
        self.posvec_len = posvec_len
        self.radiomics_vec_len = radiomics_vec_len
        self.swin_out_classes = 1024
        self.pred_layer = _PredictionBranch(self.swin_out_classes, self.posvec_len, 3, radiomics_vec_len, out_classes)
        self.model_arch = model_arch
        if self.model_arch == 'swin_transformer':
            self.model = create_swin_transformer_model(
                in_channels=in_channels,
                out_classes=self.swin_out_classes)
        else:
            raise RuntimeError('Cannot build swin transformer 2D/2.5D model for model_arch="%s". No implementation given.' % \
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
        x0 = torch.reshape(x0, [batch_size, self.swin_out_classes])

        sex = sex.reshape([batch_size, 1])
        age = age.reshape([batch_size, 1])
        lesion_volume = lesion_volume.reshape([batch_size, 1])
        
        sex_age_volume = torch.cat((sex, age, lesion_volume), dim=1)
        posvec = posvec.reshape([batch_size, self.posvec_len])
        radiomics_vec = radiomics_vec.reshape([batch_size, self.radiomics_vec_len])

        y_pred = self.pred_layer(x0, radiomics_vec, posvec, sex_age_volume)
        return y_pred


class SwinTransformerWrapper3D(nn.Module):
    def __init__(self, in_channels, out_classes, posvec_len, radiomics_vec_len, model_arch):
        super().__init__()
        self.in_channels = in_channels
        self.posvec_len = posvec_len
        self.radiomics_vec_len = radiomics_vec_len
        self.swin_out_classes = 1536
        self.pred_layer = _PredictionBranch(self.swin_out_classes, self.posvec_len, 3, radiomics_vec_len, out_classes)
        self.model_arch = model_arch
        if self.model_arch == 'swin_transformer':
            self.model = create_swin_transformer_3d_model(in_channels=in_channels)
        else:
            raise RuntimeError('Cannot build swin transformer 3D model for model_arch="%s". No implementation given.' % \
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

        x0 = self.model(x)[-1]
        x0 = torch.reshape(x0, [batch_size, self.swin_out_classes])

        sex = sex.reshape([batch_size, 1])
        age = age.reshape([batch_size, 1])
        lesion_volume = lesion_volume.reshape([batch_size, 1])
        
        sex_age_volume = torch.cat((sex, age, lesion_volume), dim=1)
        posvec = posvec.reshape([batch_size, self.posvec_len])
        radiomics_vec = radiomics_vec.reshape([batch_size, self.radiomics_vec_len])

        y_pred = self.pred_layer(x0, radiomics_vec, posvec, sex_age_volume)
        return y_pred
