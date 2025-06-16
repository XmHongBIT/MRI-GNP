import argparse
import os
import torch.nn as nn
import argparse
import digicare
import torch
from glob import glob
from torch.utils import data
from torch.optim import Adam as AdamOptimizer
from digicare.utilities.file_ops import join_path, mkdir, gd
from digicare.utilities.data_io import save_pkl
from digicare.utilities.database import Database
from digicare.database.tumor_diagnosis.database_split_rules import *
from digicare.runtime.tumor_diagnosis.model_training import GenericClassifierTrainer, DataLoader
from digicare.runtime.tumor_diagnosis.models.custom.ds_res_encode import ResidualDownsampleEncoder2d, ResidualDownsampleEncoder3d
from digicare.runtime.tumor_diagnosis.models.resnet.resnet_wrapper import ResNetWrapper2D, ResNetWrapper3D
from digicare.runtime.tumor_diagnosis.models.vit.vit_wrapper import VitWrapper2D
from digicare.runtime.tumor_diagnosis.data_processing import DatabasePreprocessor, create_database_for_tumor_diagnosis, get_tumor_diagnosis_db_keys
from digicare.runtime.tumor_diagnosis.models.squeezenet.squeezenet_wrapper import SqueezeNetWrapper2D
from digicare.runtime.tumor_diagnosis.data_processing import add_segmentation_info_to_database
from radiomics.featureextractor import RadiomicsFeatureExtractor
import SimpleITK as sitk
def preprocess_input_data(args) -> Database:

    # prepare database
    input_database_xlsx = join_path(args['output_dir'], 'input.xlsx')
    input_database = create_database_for_tumor_diagnosis()
    record = input_database.make_empty_record()
    record['data_source'] = 'tiantan_hospital'
    record['subject_name'] = str(args['subject_name'])
    record['T1'] = str(args['t1_image'])
    record['T2'] = str(args['t2_image'])
    record['T1ce'] = str(args['t1ce_image'])
    record['T2FLAIR'] = str(args['flair_image'])
    record['sex'] = str(args['sex_info'])
    record['age'] = str(args['age_info'])
    record['preprocess_strategy'] = 'N4+affine'
    input_database.add_record(record)
    input_database.export_xlsx(input_database_xlsx, up_freezed_rows=1, left_freezed_cols=2)

    # preprocess data
    preprocessed_database_xlsx = join_path(args['output_dir'], 'preprocessed.xlsx')
    PACKAGE_PATH = gd(digicare.__file__)
    mni152_standard_template = join_path(PACKAGE_PATH, 'resources', 'T1_mni152_raw_manual_bm', 'mni152_standard.nii.gz')
    preprocessor = DatabasePreprocessor(raw_database=input_database, num_workers=1, mni152_template=mni152_standard_template)
    preprocessed_database, _ = preprocessor.preprocess_data(join_path(args['output_dir'], 'preprocessed_images'))
    # add record details
    subject_record = preprocessed_database.get_record(0)
    if args['debug_mode']:
        subject_record['radiomics_vec'] = '/hd/tumor_analysis/digicare/experiments/003_tiantan_glioma_10k/radiomics_feature/radiomics_feature_t1ce_features/anhui_DUKEMEI_F_036.pkl'
        subject_record['autoseg'] = '/hd/tumor_analysis/digicare/experiments/003_tiantan_glioma_10k/Preprocessed_data/anhui_DUKEMEI_F_036/origres/segmentation_native_t1ce.nii.gz'
    else:
        subject_record['autoseg'] = get_tumor_segmentation_from_preprocessed_images(
            subject_record['T1'], subject_record['T2'], subject_record['T1ce'], subject_record['T2FLAIR'],
        )
        subject_record['radiomics_vec'] = get_radiomics_features_from_preprocessed_images(
            subject_record['T1'], subject_record['T2'], subject_record['T1ce'], subject_record['T2FLAIR'],subject_record['autoseg']
        )
    preprocessed_database.set_record(0, subject_record)
    preprocessed_database = add_segmentation_info_to_database(preprocessed_database, raise_if_error=True)
    preprocessed_database.export_xlsx(preprocessed_database_xlsx)
    return preprocessed_database


def get_gene_inference_context(args, gene_name, database) -> dict:
    inference_context = {}

    def _calculate_input_image_channels(input_dim):
        assert input_dim in {'2D', '2.5D', '3D'}
        num_required_image_input_channels = 4
        if input_dim == '2.5D':
            # in 2.5D input mode we need to send adjacent slices into network
            # slice offset can be found in slice_step, and 7 slices for each 
            # input modality will be sent into network, so if input modality
            # is 3, 21 slices will be sent into network.
            num_required_image_input_channels *= 7
        # add extra channel for seg, no matter if seg is used or not
        num_required_image_input_channels += 1 
        return num_required_image_input_channels

    if gene_name == 'IDH':
        # initialize model
        best_model_path = '/hd/tumor_analysis/digicare/experiments/003_tiantan_glioma_10k/models_and_logs/ablation_study/lch_run/126_IDH_squeezenet_v1.1_2D_noradio/model_best.model'
        model = SqueezeNetWrapper2D(
                in_channels=_calculate_input_image_channels('2D'),
                out_classes=2, posvec_len=95, radiomics_vec_len=1316,
                model_arch='squeezenet_v1.1')
        model_state_dict = torch.load(best_model_path, map_location='cuda:%d' % args['run_on_which_gpu'])
        model_state_dict.pop('current_epoch')
        model_state_dict.pop('best_loss')
        model.load_state_dict(model_state_dict)
        model = model.cuda(args['run_on_which_gpu'])
        model.eval()

        # initialize dataloader
        data_loading_configs_for_inference = {
            'enable_data_augmentation': False,
            'augmentation_params': None,
            'with_image': True,
            'with_seg': True,
            'with_lesion_volume': True,
            'with_sex': True,
            'with_age': True,
            'with_posvec': True,
            'with_radiomics': False,
            'load_seg_from_which_key': 'autoseg',
            'use_which_channels_in_seg': [1, 2, 3], # channel id starts with 0
            'posvec_len': 95,      # must be set properly
            'radiomics_len': 1316,   # must be set properly
            'image_loading_mode': '2D',
            'class_key_name': 'IDH', # must be set properly
            '2.5D_slice_step': 5,
            'pretrained_model': None,
        }
        dataloader = data.DataLoader(
            DataLoader(database, num_iters=None, 
                loading_configs = data_loading_configs_for_inference), 
            batch_size=1, shuffle=False, num_workers=1
        )
        inference_context = {
            'model': model,
            'dataloader': dataloader,
            'output_classes': ['yes', 'no'],
            'run_on_which_gpu': args['run_on_which_gpu'],
            'use_radiomics_features_for_inference': False,
            'use_tumor_segmentation_for_inference': True,
            'enable_test_time_augmentation': False,
        }
    else:
        raise RuntimeError('Unknown gene name "%s".' % gene_name)
    return inference_context

def inference_gene(inference_context):
    def _initialize_class_name_id_mapping():
        class_name_to_id_map = {}
        id_to_class_name_map = {}
        i = 0
        for class_name in inference_context['output_classes']:
            class_name_to_id_map[class_name] = i
            id_to_class_name_map[i] = class_name
            i += 1
        return class_name_to_id_map, id_to_class_name_map
    
    class_name_to_id_map, id_to_class_name_map = _initialize_class_name_id_mapping()
    print('class_name_to_id:', class_name_to_id_map)
    print('class_id_to_name:', id_to_class_name_map)

    pred_class = ''
    class_probs_as_dict = {}
    for _, (sample) in enumerate(inference_context['dataloader']):
        _, packed_image, lesion_volume, sex, age, posvec, radiomics_vec, _ = sample # unpack
        assert packed_image.shape[0] == 1, 'Only support batch_size=1.'
        packed_image, lesion_volume, sex, age, posvec, radiomics_vec = \
            packed_image.cuda(inference_context['run_on_which_gpu']), lesion_volume.float().cuda(inference_context['run_on_which_gpu']), \
            sex.float().cuda(inference_context['run_on_which_gpu']), age.float().cuda(inference_context['run_on_which_gpu']), \
            posvec.float().cuda(inference_context['run_on_which_gpu']), radiomics_vec.float().cuda(inference_context['run_on_which_gpu'])
        with torch.no_grad():
            # b,c,x,y,z (b=1)
            Y_hat = inference_context['model'](packed_image, 
                            sex = sex,
                            age = age,
                lesion_volume = lesion_volume,
                        posvec = posvec,
                radiomics_vec = radiomics_vec)
            # Y_hat: tensor(batch_size * classes), where batch_size=1 for simplicity
            class_id_pred = int(torch.argmax(Y_hat, dim=1).long().cpu().numpy())
            pred_class = id_to_class_name_map[class_id_pred]
            probs = np.reshape(torch.softmax(Y_hat, dim=1).detach().cpu().numpy(), [len(inference_context['output_classes'])])
            class_probs = [(id_to_class_name_map[class_id], probs[class_id]) \
                for class_id in range(len(inference_context['output_classes']))]
            for clsname, clsprob in class_probs:
                class_probs_as_dict[clsname] = clsprob
    print(pred_class)
    print(class_probs_as_dict)

def run_img2gene_core(args):
    mkdir(args['output_dir'])
    database = preprocess_input_data(args)
    # build model and load pretrained weights
    print(database.get_record(0))
    print('Data preparation finished. Now running gene inference...')
    for gene in ['IDH']:
        print('Running %s inference...' % gene)
        inference_context = get_gene_inference_context(args, gene, database)
        inference_gene(inference_context)

def validate_and_convert_args(args: dict):
    assert args['sex_info'] in ['F', 'M'], 'Invalid sex info, must be "F" or "M", but got "%s".' % args['sex_info']
    assert args['radiomics_features'].endswith('.pkl'), 'Radiomics features should be a Python pickle file (*.pkl).'
    assert args['tumor_segmentation'].endswith('.nii.gz'), 'Tumor segmentation result should be a gzipped NIFTI file (*.nii.gz).'
def get_radiomics_features(t1ce_image,tumor_segmentation):
    
    yaml_dict={
    'featureClass':
    {
        'firstorder': None,
        'glcm': None,
        'gldm': None,
        'glrlm': None,
        'glszm': None,
        'ngtdm': None,
        'shape': None,
    },
   'imageType':
   {
        'Exponential': {},
        'Gradient': {},
        'LBP3D': {},
        'LoG': {},
        'Logarithm': {},
        'Original': {},
        'Square': {},
        'SquareRoot': {},
        'Wavelet': {},
   },
   'setting':
   {
        'binWidth': 50,
        'geometryTolerance': 0.0001,
        'interpolator': 'sitkBSpline',
        'label': 1,
        'normalize': True,
        'resampledPixelSpacing': None,
        'weightingNorm': None,
    }
    }
    extractor=RadiomicsFeatureExtractor()
    extractor._applyParams(paramsDict=yaml_dict)
    image = sitk.ReadImage(t1ce_image)
    roi_image = sitk.ReadImage(tumor_segmentation)
    #roi_data=sitk.GetArrayFromImage(roi_image)>0
    #roi_image=sitk.GetImageFromArray(roi_data.astype(np.uint32))
    roi_image.CopyInformation(image)
    result = extractor.execute(image, roi_image)
    return [v for k,v in result.items() if not 'diagnostics' in k]
def get_radiomics_features_from_preprocessed_images(t1_image, t2_image, t1ce_image, flair_image,tumor_segmentation):
    # TODO: provide implementation here
    radiomics_features_pkl_path=t1ce_image.replace('.nii.gz','_radiomics.pkl')
    if not os.path.exists(radiomics_features_pkl_path):
        result_list=get_radiomics_features(t1ce_image,tumor_segmentation)
        save_pkl(result_list,radiomics_features_pkl_path)

    return radiomics_features_pkl_path

def pre_tumor_ucl(path,name_list):
    command='docker run -it --gpus all -v '+path+':/wdir '+'registry.gitlab.com/picture/picture-ucl-glioma-segmentation/nnunet /wdir/'+' /wdir/'.join(name_list)           
    os.system(command)
def get_tumor_segmentation_from_preprocessed_images(t1_image, t2_image, t1ce_image, flair_image):
    # TODO: provide implementation here
    img_path,t1ce_name=os.path.split(t1ce_image)
    name_list=[os.path.split(t)[1] for t in [t1ce_image,t1_image,t2_image,flair_image]]
    tumor_segmentation_nii_gz_path=os.path.join(img_path,'segmentation_native_'+t1ce_name)
    if not os.path.exists(tumor_segmentation_nii_gz_path):
        pre_tumor_ucl(img_path,name_list)
    
    return tumor_segmentation_nii_gz_path

def main():
    print('** img2gene by Chenghao Liu **')
    parser = argparse.ArgumentParser(description='Tumor image to gene prediction.')
    parser.add_argument('--subject-name', '-subj', type=str, help='[Input] Subject name.', required=True)
    parser.add_argument('--sex-info', '-x', type=str, help='[Input] Sex info ("F"/"M").', required=True)
    parser.add_argument('--age-info', '-a', type=str, help='[Input] Age info.', required=True)
    parser.add_argument('--t1-image', '-t1', type=str, help='[Input] Raw T1w image path.', required=True)
    parser.add_argument('--t2-image', '-t2', type=str, help='[Input] Raw T2w image path.', required=True)
    parser.add_argument('--t1ce-image', '-t1ce', type=str, help='[Input] Raw T1ce image path.', required=True)
    parser.add_argument('--flair-image', '-flair', type=str, help='[Input] Raw T2-FLAIR image path.', required=True)
    # parser.add_argument('--radiomics-features', '-rfeats', type=str, help='[Input] Radiomics features.', required=True)
    # parser.add_argument('--tumor-segmentation', '-s', type=str, help='[Input] Tumor segmentation result.', required=True)
    parser.add_argument('--output-dir', '-o', type=str, help='[Output] Output folder.', required=True)
    parser.add_argument('--run-on-which-gpu', '-g', type=int, help='GPU ID (starts with 0).', required=False)
    parser.add_argument('--debug-mode', '-D', type=bool, default=False, action='store_true', help='Run img2gene in DEBUG mode.')
    args = vars(parser.parse_args())

    validate_and_convert_args(args)
    run_img2gene_core(args)

if __name__ == '__main__':
    # run unit test
    # also can be used as an example usage for the img2gene tool.
    args = {
        'debug_mode': False,
        'subject_name': 'TEST_SUBJECT',
        'sex_info': 'F',
        'age_info': '36',
        't1_image': '/hd/wjy/data_process/lijunjie/glioma_tumor_multi_gene/GBM_10000/anhui_DUKEMEI_F_036/AXT1.nii.gz',
        't2_image': '/hd/wjy/data_process/lijunjie/glioma_tumor_multi_gene/GBM_10000/anhui_DUKEMEI_F_036/AXT2WI.nii.gz',
        't1ce_image': '/hd/wjy/data_process/lijunjie/glioma_tumor_multi_gene/GBM_10000/anhui_DUKEMEI_F_036/3DT1C.nii.gz',
        'flair_image': '/hd/wjy/data_process/lijunjie/glioma_tumor_multi_gene/GBM_10000/anhui_DUKEMEI_F_036/AXFLAIR.nii.gz',
        'output_dir': '__img2gene_TEST_SUBJECT__',
        'run_on_which_gpu': 0
    }
    run_img2gene_core(args)
