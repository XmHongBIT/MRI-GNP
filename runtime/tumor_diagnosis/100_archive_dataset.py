from digicare.utilities.database import Database
from digicare.utilities.file_ops import join_path, mkdir, file_exist, cp
from digicare.utilities.data_io import try_load_nifti
from digicare.runtime.tumor_diagnosis.data_processing import create_database_for_tumor_diagnosis
from typing import Dict


def archive_database(xlsx_file, sheet_name, archive_folder: str = './archive'):
    mkdir(archive_folder)
    database = create_database_for_tumor_diagnosis()
    database.load_xlsx(xlsx_file, sheet_name)
    def _database_archive_function(record: Dict[str, str]) -> str:
        image_archive_folder = mkdir(join_path(archive_folder, record['data_source'] + '_' + record['subject_name']))
        def _archive_image(image_key) -> str:
            src_image_path = record[image_key]
            dst_image_path = join_path(image_archive_folder, '%s.nii.gz' % image_key)
            if src_image_path != '':
                if file_exist(dst_image_path) and try_load_nifti(dst_image_path):
                    return 'Skipped'
                if not file_exist(src_image_path):
                    print('\n"%s" not found.\n' % src_image_path,end='')
                    return 'Failed'
                if not src_image_path.endswith('.nii.gz'): 
                    print('\n"%s" is not a gzipped nifti file.\n' % src_image_path,end='')
                    return 'Failed'
                cp(src_image_path, dst_image_path)
                return 'Success'
            else:
                return 'Skipped'
        all_status = []
        all_status.append(_archive_image('T1'))
        all_status.append(_archive_image('T1ce'))
        all_status.append(_archive_image('T2'))
        all_status.append(_archive_image('T2FLAIR'))
        all_status.append(_archive_image('ADC'))
        if any([status == 'Failed' for status in all_status]):
            return 'Failed'
        elif all([status == 'Skipped' for status in all_status]):
            return 'Skipped'
        else:
            return 'Success'
    database.archive_by_rule(_database_archive_function)

archive_database('/hd/tumor_analysis/digicare/experiments/003_tiantan_glioma_10k/xlsx/processed/dataset_raw.xlsx', 'Sheet1',
                 archive_folder='/hd/tumor_analysis/digicare/experiments/003_tiantan_glioma_10k/archive_tumor_10k_ljj_v1.0')

