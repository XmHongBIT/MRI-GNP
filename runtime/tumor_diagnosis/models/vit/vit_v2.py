# hacking the model
import torch
from transformers.models.vit.modeling_vit import ViTModel
from transformers.models.vit.configuration_vit import ViTConfig

def build_vit_google_base_patch32_224(image_size=256, channels=3):
    config = ViTConfig(attention_probs_dropout_prob=0.0, encoder_stride=16,
                    hidden_act='gelu',hidden_dropout_prob=0.0,
                    hidden_size=768, image_size=image_size, initializer_range=0.02, 
                    intermediate_size=3072,layer_norm_eps=1e-12,
                    num_attention_heads=12,num_channels=channels,
                    num_hidden_layers=12,patch_size=32,qkv_bias=True)
    model = ViTModel(config)
    return model

if __name__ == '__main__':
    model = build_vit_google_base_patch32_224(image_size=256, channels=3)
    x = torch.zeros(16,3,256,256)
    y = model(x)
    print(y.last_hidden_state.shape)
    print(y.pooler_output.shape)
