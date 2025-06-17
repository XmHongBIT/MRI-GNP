import os
import torch
import torch.nn as nn
import argparse
import digicare
import warnings
from glob import glob
from torch.utils import data
from torch.optim import Adam as AdamOptimizer
from digicare.utilities.file_ops import join_path, mkdir, gd, file_exist
from digicare.utilities.data_io import save_pkl
from digicare.utilities.database import Database
from digicare.database.tumor_diagnosis.database_split_rules import *
from digicare.runtime.tumor_diagnosis.model_training import GenericRegressionTrainer, DataLoader
from digicare.runtime.tumor_diagnosis.models.custom.ds_res_encode import ResidualDownsampleEncoder2d, ResidualDownsampleEncoder3d
from digicare.runtime.tumor_diagnosis.models.resnet.resnet_wrapper import ResNetWrapper2D, ResNetWrapper3D
from digicare.runtime.tumor_diagnosis.models.vit.vit_wrapper import VitWrapper2D
from digicare.runtime.tumor_diagnosis.models.densenet.densenet_wrapper import DenseNetWrapper2D
from digicare.runtime.tumor_diagnosis.models.googlenet.googlenet_wrapper import GoogLeNetWrapper2D
from digicare.runtime.tumor_diagnosis.models.squeezenet.squeezenet_wrapper import SqueezeNetWrapper2D
from digicare.runtime.tumor_diagnosis.models.vgg.vgg_wrapper import VGGWrapper2D
from digicare.runtime.tumor_diagnosis.models.swin_transformer.swin_transformer import SwinTransformerWrapper2D
from digicare.runtime.tumor_diagnosis.data_processing import create_database_for_2HG

def load_configuration_file(cfg_file_path):
    cfg = ""
    with open(cfg_file_path, 'r') as f:
        cfg = f.read()
    return eval(cfg)

def auto_infer_full_configuration_file_path(config:str):
    pattern = 'configs/regression/'+config+'*'
    possible_config_files = glob(pattern)
    if len(possible_config_files) == 0:
        raise RuntimeError('No matched configuration file found. Trying to glob "%s" but failed to find any matched configuration. ' \
                           'Config: "%s"' % (pattern, config))
    if len(possible_config_files) != 1:
        raise RuntimeError('Find multiple matched configs: %s.' % str(possible_config_files))
    return possible_config_files[0]

def parse_global_configuration(args):
    GLOBAL_CONFIG_PATH = auto_infer_full_configuration_file_path(args['global_config'])
    GLOBAL_CONFIG = load_configuration_file(GLOBAL_CONFIG_PATH)
    if args['run_on_which_gpu'] is not None:
        GLOBAL_CONFIG['run_on_which_gpu'] = args['run_on_which_gpu'] # override
    if 'verbose_level' in os.environ:
        print('warning: environment variable "verbose_level" already set.')
    os.environ['verbose_level'] = str(GLOBAL_CONFIG['verbose_level'])
    print('global config:', GLOBAL_CONFIG_PATH)
    mkdir(GLOBAL_CONFIG['output_root_dir'])
    return GLOBAL_CONFIG

def parse_experiment_configuration(args, GLOBAL_CONFIG):
    CUSTOM_EXPERIMENT_CONFIGS_PATH = auto_infer_full_configuration_file_path(args['experiment_config'])
    EXPERIMENT_CONFIG = load_configuration_file(CUSTOM_EXPERIMENT_CONFIGS_PATH)
    if 'pretrained_model' not in EXPERIMENT_CONFIG:
        EXPERIMENT_CONFIG['pretrained_model'] = None
    if 'model_arch' not in EXPERIMENT_CONFIG:
        print('NOTE: using custom model as "model_arch" is not defined in experiment config.')
        EXPERIMENT_CONFIG['model_arch'] = 'custom'
    print('experiment config:', CUSTOM_EXPERIMENT_CONFIGS_PATH)
    def _check_if_experiment_config_is_valid(EXPERIMENT_CONFIG: dict):
        if isinstance(EXPERIMENT_CONFIG, dict) == False:
            raise RuntimeError('experiment config should be a python dict object.')
        required_fields = ['database_xlsx',
                           'database_organize_function',
                           'input', 
                           'input_dim', 
                           'use_which_channels_in_seg', 
                           'enable_train_image_augmentation',
                           'output',
                           'num_epochs_for_training']
        for required_field in required_fields:
            if required_field not in EXPERIMENT_CONFIG:
                raise RuntimeError('Cannot find required field "%s" in experiment config. ' \
                                   'A valid experiment config should contain following fields: %s' \
                                    % (required_field, str(required_fields)))

        supported_model_archs = ['custom', 
                                 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200',
                                 'densenet201',
                                 'googlenet',
                                 'squeezenet_v1.0', 'squeezenet_v1.1',
                                 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'vit', 'vit_v2']
        if EXPERIMENT_CONFIG['model_arch'] not in supported_model_archs:
            raise RuntimeError('Found invalid model arch: "%s". Expected model arch should be one of: %s.' % \
                (EXPERIMENT_CONFIG['model_arch'], supported_model_archs))
    _check_if_experiment_config_is_valid(EXPERIMENT_CONFIG)
    def _auto_infer_model_output_dir():
        rpath = os.path.relpath(CUSTOM_EXPERIMENT_CONFIGS_PATH, join_path('configs', 'regression'))
        rpath = rpath.replace('.cfg', '')
        return join_path(GLOBAL_CONFIG['output_root_dir'], rpath)
    model_output_dir = mkdir(_auto_infer_model_output_dir())
    EXPERIMENT_CONFIG['model_output_dir'] = model_output_dir
    return EXPERIMENT_CONFIG

def process_database_using_custom_rules(GLOBAL_CONFIG, EXPERIMENT_CONFIG):

    print('removing samples with invalid output class...')
    MAIN_DATABASE = EXPERIMENT_CONFIG['database_xlsx']
    main_database_obj = create_database_for_2HG(MAIN_DATABASE)

    def _map_class_type_using_custom_rules(database_obj: Database, map_dict):
        if map_dict is None: return database_obj
        print('* mapping output class using custom rules: %s' % str(map_dict))
        for i in range(database_obj.num_records()):
            record = database_obj.get_record(i)
            class_type_before_mapping = record[EXPERIMENT_CONFIG['output']]
            for class_type_after_mapping, class_types_before_mapping in map_dict.items():
                if not isinstance(class_types_before_mapping, list):
                    class_types_before_mapping = [class_types_before_mapping]
                if class_type_before_mapping in class_types_before_mapping and class_type_before_mapping != class_type_after_mapping:
                    record[EXPERIMENT_CONFIG['output']] = class_type_after_mapping
                    print('* map type "%s" to "%s" for record %d.' % (class_type_before_mapping, class_type_after_mapping, i))
                    break
            database_obj.set_record(i, record)
        return database_obj

    if 'output_classes_mapping' in EXPERIMENT_CONFIG:
        main_database_obj = _map_class_type_using_custom_rules(main_database_obj, EXPERIMENT_CONFIG['output_classes_mapping'])
    train_database, val_database, test_database = EXPERIMENT_CONFIG['database_organize_function'](main_database_obj, EXPERIMENT_CONFIG)

    def _report_samples_that_do_not_satisfy_input_requirements(record):
        for input_key in EXPERIMENT_CONFIG['input']:
            if input_key in ['autoseg_vol']: continue # skip special keys
            if record[input_key] in ['', None]:
                raise RuntimeError(\
                    '* I found a sample that does not satisfy the input requirements.\n'
                    'Input requirements are: %s\n'
                    'Sample record is: %s\n'
                    'Database organize function is: %s\n'
                    '* Please check your "database_organize_function" and make sure every sample you have '
                    'chosen can satisfy the input requirements that you defined.' % \
                    (str(EXPERIMENT_CONFIG['input']), str(record), EXPERIMENT_CONFIG['database_organize_function'].__name__))
    if train_database is not None:
        for i in range(train_database.num_records()):
            _report_samples_that_do_not_satisfy_input_requirements(train_database.get_record(i))
    if val_database is not None:
        for i in range(val_database.num_records()):
            _report_samples_that_do_not_satisfy_input_requirements(val_database.get_record(i))
    if test_database is not None:
        for i in range(test_database.num_records()):
            _report_samples_that_do_not_satisfy_input_requirements(test_database.get_record(i))
    
    # print('')
    # print_database_summary(train_database, EXPERIMENT_CONFIG, prefix='* database summary (train): ', postfix='\n')
    # print_database_summary(val_database,   EXPERIMENT_CONFIG, prefix='* database summary (val): ',   postfix='\n')
    # print_database_summary(test_database,  EXPERIMENT_CONFIG, prefix='* database summary (test): ',  postfix='\n')

    print('model output folder:', EXPERIMENT_CONFIG['model_output_dir'])
    image_modalities_that_need_to_be_cleared = [modality for modality in GLOBAL_CONFIG['all_possible_input_image_modalities'] if modality not in EXPERIMENT_CONFIG['input']]
    print(image_modalities_that_need_to_be_cleared)
    train_database : Database
    val_database   : Database
    test_database  : Database
    train_database = train_database.clear_key(image_modalities_that_need_to_be_cleared) if train_database is not None else None
    val_database = val_database.clear_key(image_modalities_that_need_to_be_cleared) if val_database is not None else None
    test_database = test_database.clear_key(image_modalities_that_need_to_be_cleared) if test_database is not None else None

    # save databases
    train_database.export_xlsx(join_path(EXPERIMENT_CONFIG['model_output_dir'], 'train.xlsx'), left_freezed_cols=2) if train_database is not None else None
    val_database.export_xlsx(join_path(EXPERIMENT_CONFIG['model_output_dir'], 'val.xlsx'), left_freezed_cols=2) if val_database is not None else None
    test_database.export_xlsx(join_path(EXPERIMENT_CONFIG['model_output_dir'], 'test.xlsx'), left_freezed_cols=2) if test_database is not None else None

    # validate databases
    def validate_databases():
        train_subjects = train_database.data_dict['subject_name'] if train_database is not None else []
        val_subjects   = val_database.data_dict['subject_name']   if val_database   is not None else []
        test_subjects  = test_database.data_dict['subject_name']  if test_database  is not None else []
        def contain_duplicates(list_obj):
            return len(set(list_obj)) != len(list_obj)
        if contain_duplicates(train_subjects):
            raise RuntimeError('Train set contain one or more duplicates. Please check before proceed.')
        if contain_duplicates(val_subjects):
            raise RuntimeError('Validation set contain one or more duplicates. Please check before proceed.')
        if contain_duplicates(test_subjects):
            raise RuntimeError('Test set contain one or more duplicates. Please check before proceed.')
        for subject in train_subjects:
            if subject in test_subjects or subject in val_subjects:
                raise RuntimeError('Training subject "%s" is also in val/test set!' % subject)
        for subject in val_subjects:
            if subject in test_subjects:
                raise RuntimeError('Validation subject "%s" is also in test set!' % subject)
        print('* All dataset checks PASSED.')

    validate_databases()

    return (train_database, val_database, test_database)

def generate_data_loading_configurations(GLOBAL_CONFIG, EXPERIMENT_CONFIG):
    def _check_if_need_any_image_input():
        for input_key in EXPERIMENT_CONFIG['input']:
            if input_key in GLOBAL_CONFIG['all_possible_input_image_modalities']:
                return True
        return False

    data_loading_configs_for_training = {
        'enable_data_augmentation': EXPERIMENT_CONFIG['enable_train_image_augmentation'],
        'augmentation_params': GLOBAL_CONFIG['data_augmentation_parameters'],
        'with_image': _check_if_need_any_image_input(),
        'with_seg': 'autoseg' in EXPERIMENT_CONFIG['input'],
        'with_lesion_volume': 'autoseg_vol' in EXPERIMENT_CONFIG['input'],
        'with_sex': 'sex' in EXPERIMENT_CONFIG['input'],
        'with_age': 'age' in EXPERIMENT_CONFIG['input'],
        'with_posvec': 'autoseg_posvec' in EXPERIMENT_CONFIG['input'],
        'with_radiomics':'radiomics_vec' in EXPERIMENT_CONFIG['input'],
        'load_seg_from_which_key': EXPERIMENT_CONFIG['load_seg_from_which_key'],
        'use_which_channels_in_seg': EXPERIMENT_CONFIG['use_which_channels_in_seg'], # channel id starts with 0
        'posvec_len': GLOBAL_CONFIG['autoseg_posvec_length'],      # must be set properly
        'radiomics_len': GLOBAL_CONFIG['radiomics_vec_length'],   # must be set properly
        'image_loading_mode': EXPERIMENT_CONFIG['input_dim'],
        'class_key_name': EXPERIMENT_CONFIG['output'], # must be set properly
        '2.5D_slice_step': GLOBAL_CONFIG['2.5D_slice_step'],
        'pretrained_model': EXPERIMENT_CONFIG['pretrained_model'],
    }
    
    data_loading_configs_for_inference = {
        'enable_data_augmentation': False,
        'augmentation_params': None,
        'with_image': _check_if_need_any_image_input(),
        'with_seg': 'autoseg' in EXPERIMENT_CONFIG['input'],
        'with_lesion_volume': 'autoseg_vol' in EXPERIMENT_CONFIG['input'],
        'with_sex': 'sex' in EXPERIMENT_CONFIG['input'],
        'with_age': 'age' in EXPERIMENT_CONFIG['input'],
        'with_posvec': 'autoseg_posvec' in EXPERIMENT_CONFIG['input'],
        'with_radiomics':'radiomics_vec' in EXPERIMENT_CONFIG['input'],
        'load_seg_from_which_key': EXPERIMENT_CONFIG['load_seg_from_which_key'],
        'use_which_channels_in_seg': EXPERIMENT_CONFIG['use_which_channels_in_seg'], # channel id starts with 0
        'posvec_len': GLOBAL_CONFIG['autoseg_posvec_length'],      # must be set properly
        'radiomics_len': GLOBAL_CONFIG['radiomics_vec_length'],   # must be set properly
        'image_loading_mode': EXPERIMENT_CONFIG['input_dim'],
        'class_key_name': EXPERIMENT_CONFIG['output'], # must be set properly
        '2.5D_slice_step': GLOBAL_CONFIG['2.5D_slice_step'],
        'pretrained_model': None,
    }
    return (data_loading_configs_for_training, data_loading_configs_for_inference)

def generate_data_loaders(PROCESSED_DATABASES, DATA_LOADING_CONFIGS):
    train_database, val_database, test_database = PROCESSED_DATABASES
    data_loading_configs_for_training, data_loading_configs_for_inference = DATA_LOADING_CONFIGS
    # make data loaders
    train_loader = data.DataLoader(
        DataLoader(
            train_database,
            num_iters       = GLOBAL_CONFIG['num_training_iterations_per_epoch'],
            loading_configs = data_loading_configs_for_training
        ), 
        batch_size  = GLOBAL_CONFIG['network_batch_size_for_train_and_test'], 
        shuffle     = True, 
        num_workers = GLOBAL_CONFIG['data_loader_worker_num']
    ) if train_database is not None else None
    val_loader = data.DataLoader(
        DataLoader(
            val_database, 
            num_iters       = None, 
            loading_configs = data_loading_configs_for_inference), 
        batch_size  = GLOBAL_CONFIG['network_batch_size_for_train_and_test'], 
        shuffle     = False, 
        num_workers = GLOBAL_CONFIG['data_loader_worker_num']
    ) if val_database is not None else None
    test_loader = data.DataLoader(
        DataLoader(
            test_database, 
            num_iters       = None, 
            loading_configs = data_loading_configs_for_inference), 
        batch_size  = GLOBAL_CONFIG['network_batch_size_for_train_and_test'], 
        shuffle     = False, 
        num_workers = GLOBAL_CONFIG['data_loader_worker_num']
    ) if test_database is not None else None

    return (train_loader, val_loader, test_loader)

def select_and_initialize_model(GLOBAL_CONFIG, EXPERIMENT_CONFIG):
    model = None
    input_dim = EXPERIMENT_CONFIG['input_dim']
    model_arch = EXPERIMENT_CONFIG['model_arch']

    # the objective of this function is pretty simple:
    # initialize model based on input_dim and model_arch
    # and return the built model object

    def _calculate_input_image_channels():
        num_required_image_input_channels = 0
        for input_key in EXPERIMENT_CONFIG['input']:
            if input_key in GLOBAL_CONFIG['all_possible_input_image_modalities']:
                num_required_image_input_channels += 1
        if input_dim == '2.5D':
            # in 2.5D input mode we need to send adjacent slices into network
            # slice offset can be found in slice_step, and 7 slices for each 
            # input modality will be sent into network, so if input modality
            # is 3, 21 slices will be sent into network.
            num_required_image_input_channels *= 7
        # add extra channel for seg, no matter if seg is used or not
        num_required_image_input_channels += 1 
        return num_required_image_input_channels

    if model_arch == 'custom':
        ModelClass_t = None
        if input_dim in ['2D', '2.5D']:
            ModelClass_t = ResidualDownsampleEncoder2d
        elif input_dim == '3D':
            ModelClass_t = ResidualDownsampleEncoder3d
        if ModelClass_t in [ResidualDownsampleEncoder2d, ResidualDownsampleEncoder3d]:
            model = ModelClass_t(
                in_channels=_calculate_input_image_channels(),
                out_classes=1,
                fm=GLOBAL_CONFIG['model_fm'],
                conv_blks_per_res=GLOBAL_CONFIG['num_convolution_stages_per_resblock'],
                posvec_len=GLOBAL_CONFIG['autoseg_posvec_length'],
                radiomics_vec_len=GLOBAL_CONFIG['radiomics_vec_length'])
    elif model_arch in ['swin_transformer']:
        if input_dim in ['2D', '2.5D']:
            model = SwinTransformerWrapper2D(
                in_channels=_calculate_input_image_channels(),
                out_classes=1,
                posvec_len=GLOBAL_CONFIG['autoseg_posvec_length'],
                radiomics_vec_len=GLOBAL_CONFIG['radiomics_vec_length'],
                model_arch=model_arch
            )
    elif model_arch in ['resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200']:
        if input_dim in ['2D', '2.5D']:
            model = ResNetWrapper2D(
                in_channels=_calculate_input_image_channels(),
                out_classes=1,
                posvec_len=GLOBAL_CONFIG['autoseg_posvec_length'],
                radiomics_vec_len=GLOBAL_CONFIG['radiomics_vec_length'],
                resnet_model_arch=model_arch)
        elif input_dim == '3D':
            model = ResNetWrapper3D(
                in_channels=_calculate_input_image_channels(),
                out_classes=1,
                posvec_len=GLOBAL_CONFIG['autoseg_posvec_length'],
                radiomics_vec_len=GLOBAL_CONFIG['radiomics_vec_length'],
                resnet_model_arch=model_arch)
    elif model_arch in ['vit', 'vit_v2']:
        if input_dim in ['2D', '2.5D']:
            model = VitWrapper2D(
                in_channels=_calculate_input_image_channels(),
                out_classes=1,
                posvec_len=GLOBAL_CONFIG['autoseg_posvec_length'],
                radiomics_vec_len=GLOBAL_CONFIG['radiomics_vec_length'],
                vit_model_arch=model_arch)
    elif model_arch in ['densenet201']:
        if input_dim in ['2D', '2.5D']:
            model = DenseNetWrapper2D(
                in_channels=_calculate_input_image_channels(),
                out_classes=1,
                posvec_len=GLOBAL_CONFIG['autoseg_posvec_length'],
                radiomics_vec_len=GLOBAL_CONFIG['radiomics_vec_length'],
                densenet_model_arch=model_arch)
    elif model_arch in ['googlenet']:
        if input_dim in ['2D', '2.5D']:
            model = GoogLeNetWrapper2D(
                in_channels=_calculate_input_image_channels(),
                out_classes=1,
                posvec_len=GLOBAL_CONFIG['autoseg_posvec_length'],
                radiomics_vec_len=GLOBAL_CONFIG['radiomics_vec_length'],
                model_arch=model_arch)
    elif model_arch in ['squeezenet_v1.0', 'squeezenet_v1.1']:
        if input_dim in ['2D', '2.5D']:
            model = SqueezeNetWrapper2D(
                in_channels=_calculate_input_image_channels(),
                out_classes=1,
                posvec_len=GLOBAL_CONFIG['autoseg_posvec_length'],
                radiomics_vec_len=GLOBAL_CONFIG['radiomics_vec_length'],
                model_arch=model_arch)
    elif model_arch in ['vgg11', 'vgg13', 'vgg16', 'vgg19']:
        if input_dim in ['2D', '2.5D']:
            model = VGGWrapper2D(
                in_channels=_calculate_input_image_channels(),
                out_classes=1,
                posvec_len=GLOBAL_CONFIG['autoseg_posvec_length'],
                radiomics_vec_len=GLOBAL_CONFIG['radiomics_vec_length'],
                model_arch=model_arch)
    
    if model is None:
        raise RuntimeError('Cannot build model for input_dim="%s", model_arch="%s". No implementation given.' % \
            (input_dim, model_arch))

    # convert relative path to absolute path if pretrained model path is given
    if EXPERIMENT_CONFIG['pretrained_model'] is not None:
        PACKAGE_PATH  = gd(digicare.__file__)
        all_pretrained_model_dir = join_path(PACKAGE_PATH, 'resources', 'pretrained_models')
        EXPERIMENT_CONFIG['pretrained_model'] = join_path(all_pretrained_model_dir, EXPERIMENT_CONFIG['pretrained_model'])

    return model

def generate_model_optimizer_trainer(GLOBAL_CONFIG, EXPERIMENT_CONFIG, DATA_LOADERS):
    train_loader, val_loader, test_loader = DATA_LOADERS
    model: nn.Module = select_and_initialize_model(GLOBAL_CONFIG, EXPERIMENT_CONFIG)
    model = model.cuda(device=GLOBAL_CONFIG['run_on_which_gpu'])
    optim = AdamOptimizer(model.parameters(), lr=GLOBAL_CONFIG['learning_rate'], betas=(0.9, 0.999))
    trainer = GenericRegressionTrainer(
        output_folder=EXPERIMENT_CONFIG['model_output_dir'], gpu_index=GLOBAL_CONFIG['run_on_which_gpu'], 
        model=model, optim=optim, lr_scheduler=None, 
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, 
        variable_name=EXPERIMENT_CONFIG['output'],
        pretrained_model=EXPERIMENT_CONFIG['pretrained_model'])
    return (model, optim, trainer)

if __name__ == '__main__':
    PACKAGE_PATH  = gd(digicare.__file__)
    parser = argparse.ArgumentParser(
        description='Run ablation study for tumor diagnosis.')
    parser.add_argument(
        '--experiment-config', '-c', type=str, 
        help='Specify experiment configuration file name (located at "configs/regression/"). '
            'For example, "--experiment-config 001_test" will load experiment configuration from '
            '"configs/regression/001_test.cfg".', 
        required=True)
    parser.add_argument(
        '--run-on-which-gpu', '-g', type=int, 
        help='GPU ID (starts with 0).', 
        required=False)
    parser.add_argument(
        '--global-config', '-G', type=str, 
        help='Specify global configuration file name (located at "configs/regression/"). '
            'For example, "--experiment-config global" will load global configuration from '
            '"configs/regression/global.cfg".', 
        required=False, 
        default="GLOBAL_CONFIGS")
    args = vars(parser.parse_args())

    # parse and generate a lot of things
    GLOBAL_CONFIG        = parse_global_configuration(args)
    EXPERIMENT_CONFIG    = parse_experiment_configuration(args, GLOBAL_CONFIG)
    PROCESSED_DATABASES  = process_database_using_custom_rules(GLOBAL_CONFIG, EXPERIMENT_CONFIG)
    DATA_LOADING_CONFIGS = generate_data_loading_configurations(GLOBAL_CONFIG, EXPERIMENT_CONFIG)
    DATA_LOADERS         = generate_data_loaders(PROCESSED_DATABASES, DATA_LOADING_CONFIGS)

    # print info
    data_loading_configs_for_training, data_loading_configs_for_inference = DATA_LOADING_CONFIGS
    print('* global and experiment configs (please check carefully before proceed):')
    print('global configs:',     GLOBAL_CONFIG)
    print('experiment configs:', EXPERIMENT_CONFIG)
    print('* inferred data loading configs:')
    print('training:', data_loading_configs_for_training)
    print('inference:',  data_loading_configs_for_inference)
    print('train/val/test databases have been saved.')
    save_pkl(GLOBAL_CONFIG,     join_path(EXPERIMENT_CONFIG['model_output_dir'], 'GLOBAL_CONFIG.pkl'))
    save_pkl(EXPERIMENT_CONFIG, join_path(EXPERIMENT_CONFIG['model_output_dir'], 'EXPERIMENT_CONFIG.pkl'))
    save_pkl(data_loading_configs_for_training,  join_path(EXPERIMENT_CONFIG['model_output_dir'], 'data_loading_configs_for_training.pkl'))
    save_pkl(data_loading_configs_for_inference, join_path(EXPERIMENT_CONFIG['model_output_dir'], 'data_loading_configs_for_inference.pkl'))

    # train model and run inference
    model, optim, trainer = generate_model_optimizer_trainer(GLOBAL_CONFIG, EXPERIMENT_CONFIG, DATA_LOADERS)
    train_loader, val_loader, test_loader = DATA_LOADERS
    finished_flag = join_path(EXPERIMENT_CONFIG['model_output_dir'], 'EXPERIMENT_FINISHED')
    if not file_exist(finished_flag):
        trainer.train(num_epochs=EXPERIMENT_CONFIG['num_epochs_for_training'])
        trainer.inference(test_loader=val_loader, enable_tta=(EXPERIMENT_CONFIG['input_dim'] in ['2D', '2.5D']))
        with open(finished_flag, 'w') as f: pass
    trainer.inference(test_loader=train_loader)
    print('Finished.')
