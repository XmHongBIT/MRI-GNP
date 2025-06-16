from digicare.utilities.database import Database
from digicare.runtime.tumor_diagnosis.data_processing import create_database_for_tumor_diagnosis

upenn_age_sex_os_xlsx = '/hd/tumor_analysis/digicare/experiments/003_tiantan_glioma_10k/xlsx/ljj_final/UPENN-GBM_clinical_info_v2.1.xlsx'
def add_upenn_age_info(tumor_xlsx_file, out_xlsx_file):
    dbse = create_database_for_tumor_diagnosis(xlsx_file=tumor_xlsx_file)
    upenn_dbse = Database(db_keys=['ID', 'Gender', 'Age_at_scan_years', 'Survival_from_surgery_days_UPDATED'], 
                          xlsx_file=upenn_age_sex_os_xlsx,
                          sheet_name='UPENN-GBM_clinical_info_v2.1')
    for i in range(dbse.num_records()):
        record = dbse.get_record(i)
        if record['data_source'] == 'upenn':
            subject_name = record['subject_name']
            ind, rec = upenn_dbse.get_record_from_key_val_pair('ID', subject_name)
            if rec is not None:
                record['sex'] = str(rec['Gender'])
                record['age'] = str(int(float(str(rec['Age_at_scan_years']))))
                record['OS'] = str(rec['Survival_from_surgery_days_UPDATED'])
                record['follow_up'] = 'dead'
                print('update', record)
        dbse.set_record(i, record)
    dbse.export_xlsx(out_xlsx_file)

add_upenn_age_info('/hd/tumor_analysis/digicare/experiments/003_tiantan_glioma_10k/xlsx/processed/tiantan_ljj_v1.0.xlsx', 
                   '/hd/tumor_analysis/digicare/experiments/003_tiantan_glioma_10k/xlsx/processed/tiantan_ljj_v1.1.xlsx')
