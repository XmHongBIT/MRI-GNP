import argparse
import os
import torch.nn as nn
import argparse
import digicare
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

def preprocess_input_data(output_dir, subject_name, image_pack, sex_age):
    # prepare database
    input_database_xlsx = join_path(output_dir, 'input_info.xlsx')
    input_database = create_database_for_tumor_diagnosis()
    record = input_database.make_empty_record()
    record['data_source'] = 'tiantan_hospital'
    record['subject_name'] = subject_name
    record['T1'] = image_pack[0]
    record['T2'] = image_pack[1]
    record['T1ce'] = image_pack[2]
    record['T2FLAIR'] = image_pack[3]
    record['sex'] = sex_age[0]
    record['age'] = sex_age[1]
    record['preprocess_strategy'] = 'N4+affine'
    input_database.add_record(record)
    input_database.export_xlsx(input_database_xlsx, up_freezed_rows=1, left_freezed_cols=2)
    # preprocess data
    PACKAGE_PATH = gd(digicare.__file__)
    mni152_standard_template = join_path(PACKAGE_PATH, 'resources', 'T1_mni152_raw_manual_bm', 'mni152_standard.nii.gz')
    preprocessor = DatabasePreprocessor(raw_database=input_database, num_workers=1, mni152_template=mni152_standard_template)
    database_origres, database_lowres = preprocessor.preprocess_data(join_path(output_dir, 'preprocessed_images'))

def build_and_load_models():
    pretrained_weights = ''
    model = 

def run_tumor_gene_e2e(args):

    # get args
    mkdir(args.output_dir)
    subject_name = args.subject_name
    t1_image = args.t1_image
    t2_image = args.t2_image
    t1ce_image = args.t1ce_image
    flair_image = args.flair_image
    rad_feats = args.radiomics_features
    segmentation = args.tumor_segmentation
    output_dir = args.output_folder
    gpu = args.run_on_which_gpu
    sex = args.sex
    age = args.age
    
    preprocess_input_data(output_dir, subject_name, [t1_image, t2_image, t1ce_image, flair_image], [sex, age])

    # build model and load pretrained weights


    
def main():
    parser = argparse.ArgumentParser(description='Tumor image to gene prediction.')
    parser.add_argument('--subject-name', '-subj', type=str, help='[Input] Subject name.', required=True)
    parser.add_argument('--sex-info', '-x', type=str, help='[Input] Sex info ("F"/"M").', required=True)
    parser.add_argument('--age-info', '-a', type=str, help='[Input] Age info.', required=True)
    parser.add_argument('--t1-image', '-t1', type=str, help='[Input] T1w image path.', required=True)
    parser.add_argument('--t2-image', '-t2', type=str, help='[Input] T2w image path.', required=True)
    parser.add_argument('--t1ce-image', '-t1ce', type=str, help='[Input] T1ce image path.', required=True)
    parser.add_argument('--flair-image', '-flair', type=str, help='[Input] T2-FLAIR image path.', required=True)
    parser.add_argument('--radiomics-features', '-rfeats', type=str, help='[Input] Radiomics features.', required=True)
    parser.add_argument('--tumor-segmentation', '-s', type=str, help='[Input] Tumor segmentation result.', required=True)
    parser.add_argument('--output-dir', '-o', type=str, help='[Output] Output folder.', required=True)
    parser.add_argument('--run-on-which-gpu', '-g', type=int, help='GPU ID (starts with 0).', required=False)
    args = vars(parser.parse_args())
    
    if args.sex_info not in ['F', 'M']: 'Invalid sex info, must be "F" or "M", but got "%s".' % args.sex_info
    
    run_tumor_gene_e2e(args)

    