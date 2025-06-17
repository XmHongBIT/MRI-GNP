from digicare.utilities.database import Database
import digicare
from digicare.runtime.tumor_diagnosis.data_processing import DatabaseChecker, DatabasePreprocessor, create_database_for_tumor_diagnosis, get_tumor_diagnosis_db_keys
from digicare.utilities.image_ops import barycentric_coordinate, center_crop
from digicare.utilities.data_io import load_csv_simple, load_nifti, save_nifti_simple, save_pkl
from digicare.utilities.file_ops import file_exist, gn, join_path, mkdir, gd
from digicare.database.tumor_diagnosis.ljj_tiantan_database_v2 import parse_ljj_final_organize
from digicare.database.tumor_diagnosis.ljj_tiantan_new_OS import parse_new_OS
import numpy as np
from cmath import nan

def add_radiomics_info_to_database(database : Database, radiomics_csv : str):
    # iterate each record in database and add relevant info
    num_records = database.num_records()
    radiomics_pkl_output_dir = mkdir(join_path(gd(radiomics_csv), gn(radiomics_csv, no_extension=True)))
    print('Loading radiomics features "%s"...' % radiomics_csv)
    radiomics_data = load_csv_simple(radiomics_csv)
    print('number of radiomics features : %d' % (len(radiomics_data.keys()) - 1) )
    all_feature_names = list(radiomics_data.keys())
    all_feature_names.remove('CaseName')
    num_records = len(radiomics_data['CaseName'])
    print('calculating mean and std for each feature and apply z-score...')
    feature_dict = {}
    for feature_name in all_feature_names:
        feature_dict[feature_name] = []
        for record_id in range(num_records):
            cell_content = radiomics_data[feature_name][record_id]
            if cell_content == 'nan':
                print('found nan, replaced with 0.0')
                cell_content = '0.0'
            feature_value = float(cell_content)
            feature_dict[feature_name].append(feature_value)
        feature_dict[feature_name + '_mean'] = np.mean(feature_dict[feature_name])
        feature_dict[feature_name + '_std'] = np.std(feature_dict[feature_name])
        if str(feature_dict[feature_name + '_mean']).find('nan') != -1 or \
            str(feature_dict[feature_name + '_std']).find('nan') != -1:
            for record_id in range(num_records):
                feature_dict[feature_name][record_id] = 0.0
            print(feature_name)
        else:
            mean = feature_dict[feature_name + '_mean']
            std = feature_dict[feature_name + '_std']
            if std < 1e-10:
                for record_id in range(num_records):
                    feature_dict[feature_name][record_id] = 0.0
            else:
                feature_dict[feature_name] = [(value - mean)/std for value in feature_dict[feature_name]]
            print(feature_name, mean, std)
        
        if nan in feature_dict[feature_name]:
            raise RuntimeError('nan in feature', feature_name)
        
    for record_id in range(num_records):
        subject_name = radiomics_data['CaseName'][record_id]
        subject_radiomics_pkl_output = join_path(radiomics_pkl_output_dir, '%s.pkl' % subject_name)
        subject_radiomics_features = []
        for feature_name in all_feature_names:
            if str(feature_dict[feature_name][record_id]).find('nan') != -1:
                raise RuntimeError('nan in feature', feature_name)
            subject_radiomics_features.append(feature_dict[feature_name][record_id])
        save_pkl(subject_radiomics_features, subject_radiomics_pkl_output)
        # write into database
        record_id_in_database, record_in_database = database.get_record_from_key_val_pair('subject_name', subject_name)
        if record_in_database is not None:
            record_in_database['radiomics_vec'] = subject_radiomics_pkl_output
            database.set_record(record_id_in_database, record_in_database)
            print('subject_radiomics_features (len):', len(subject_radiomics_features))
            #print(subject_radiomics_features)
    
    return database

if __name__ == '__main__':

    PACKAGE_PATH             = gd(digicare.__file__)
    preprocess_worker_num    = 8
    mni152_standard_template = join_path(PACKAGE_PATH, 'resources', 'T1_mni152_raw_manual_bm', 'mni152_standard.nii.gz')
    preprocess_ss_backend    = 'robex' # skull strip program. can be 'bet' (default) or 'robex'
    preprocessed_data_folder = mkdir('./Preprocessed_data/')
    xlsx_folder              = mkdir('./xlsx/')
    database_raw_output      = join_path(xlsx_folder, './processed/dataset_raw.xlsx')
    database_origres_output  = join_path(xlsx_folder, './processed/dataset_origres.xlsx')
    database_lowres_output   = join_path(xlsx_folder, './processed/dataset_lowres.xlsx')
    radiomics_csv            = 'radiomics_feature/radiomics_feature_t1ce_features.csv'
    database_origres_final   = join_path(xlsx_folder, './processed/dataset_origres_seg+radiomics.xlsx')
    database_origres_final2  = join_path(xlsx_folder, './processed/dataset_origres_seg+radiomics_fixed.xlsx')

    # step 1: organize dataset from raw xlsx tables
    database_raw = Database()
    database_raw += parse_ljj_final_organize()
    database_raw += parse_new_OS()
    database_raw.export_xlsx(database_raw_output)
    database_origres, database_lowres = DatabasePreprocessor_v2(
        database_raw, num_workers=preprocess_worker_num, mni152_template=mni152_standard_template, 
        ss_backend=preprocess_ss_backend).preprocess_data(preprocessed_data_folder)
    database_origres : Database
    database_lowres  : Database
    database_origres.export_xlsx(database_origres_output)
    database_lowres.export_xlsx(database_lowres_output)

    # # step 2: fix broken file links. Some files are found corrupted during preprocessing, simply remove those files
    # database_origres = Database(database_origres_output)
    # database_checker = DatabaseChecker(database_origres)
    # database_origres = database_checker.remove_broken_file_records(
    #     database_checker.run_existence_check(
    #         check_keys=['T1','T1ce','T2','T2FLAIR','ADC','brain_mask','autoseg','radiomics_vec']
    #     )
    # ).get_database_obj()
    # database_origres.export_xlsx(database_origres_output)

    # # step 3: add segmentation and radiomics information
    # database_origres = add_segmentation_info_to_database(database_origres)
    # database_origres = add_radiomics_info_to_database(database_origres, radiomics_csv=radiomics_csv)
    # database_origres.export_xlsx(database_origres_final)

    # # step 4: remove broken files
    database_origres = create_database_for_tumor_diagnosis(get_tumor_diagnosis_db_keys(), database_origres_final)
    database_checker = DatabaseChecker(database_origres)
    database_origres = database_checker.remove_broken_file_records(
        database_checker.run_existence_check(
            check_keys=['T1','T1ce','T2','T2FLAIR','ADC','brain_mask','autoseg','radiomics_vec']
        )
    ).get_database_obj()
    database_origres.export_xlsx(database_origres_final2,left_freezed_cols=2)
