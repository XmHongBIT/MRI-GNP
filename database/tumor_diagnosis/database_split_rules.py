from digicare.utilities.database import Database
from digicare.runtime.tumor_diagnosis.data_processing import create_database_for_tumor_diagnosis, create_database_for_2HG
import numpy as np
from typing import Union

def print_database_summary(database: Union[Database, None], EXPERIMENT_CONFIG: dict, prefix: str='', postfix: str=''):
    if database is None:
        print(prefix,end='')
        print('N/A')
        print(postfix,end='')
        return
    print(prefix,end='')
    all_data_sources = list(set(database.data_dict['data_source']))
    output_key = EXPERIMENT_CONFIG['output']
    output_classes = EXPERIMENT_CONFIG['output_classes']
    class_info_summary = ''
    for i, output_class in zip(range(len(output_classes)), output_classes):
        def _filter_class(record: dict):
            return True if record[output_key] == output_class else False
        database_for_this_class = database.keep_by_rule(_filter_class)
        class_info_summary += '%s=%d' % (output_class, database_for_this_class.num_records())
        if i < len(output_classes) - 1:
            class_info_summary += ', '
    print('%d in total,' % database.num_records(), class_info_summary)
    for data_source in all_data_sources:
        summary_for_this_DC = ''
        def _filter_data_source(record: dict):
            return True if record['data_source'] == data_source else False
        database_for_this_DC = database.keep_by_rule(_filter_data_source)
        summary_for_this_DC += '%32s: %d in total,' % (data_source, database_for_this_DC.num_records())
        for i, output_class in zip(range(len(output_classes)), output_classes):
            def _filter_output_class(record: dict):
                return True if record[output_key] == output_class else False
            database_for_this_DC_class = database_for_this_DC.keep_by_rule(_filter_output_class)
            summary_for_this_DC += ' %s=%d' % (output_class, database_for_this_DC_class.num_records())
            if i < len(output_classes) - 1: summary_for_this_DC += ','
        print(summary_for_this_DC)
    print(postfix,end='')

def ljj_split(main_database_obj: Database, EXPERIMENT_CONFIG: dict):
    # step 1: filter dataset
    print('step 1: database initial filtering...')
    print('# records before filtering:', main_database_obj.num_records())
    def check_and_remove_record_without_key(database: Database, key: str):
        def _remove_record(record:dict):
            return True if record[key] == '' else False
        return database.remove_by_rule(_remove_record)
    REQUIRED_KEYS = ['T1', 'T1ce', 'T2', 'brain_mask', 'autoseg', 'autoseg_posvec', 'radiomics_vec']
    for key in REQUIRED_KEYS:
        main_database_obj = check_and_remove_record_without_key(main_database_obj, key)
    print('# records after filtering:', main_database_obj.num_records())
    print_database_summary(main_database_obj, EXPERIMENT_CONFIG)
    print('')
    # split train set:
    db_train, db_val = create_database_for_tumor_diagnosis(), create_database_for_tumor_diagnosis()
    db_tiantan_3000 = create_database_for_tumor_diagnosis()
    for i in range(main_database_obj.num_records()):
        record = main_database_obj.get_record(i)
        data_source = record['data_source']
        if data_source in ['tiantan_cgga', 'tiantan_cgga_new', 'tiantan_duofenzi', 'tiantan_new_OS', 'tiantan_2007-2107']:
            db_train.add_record(record)
        elif data_source in ['tiantan_3000']:
            db_tiantan_3000.add_record(record)
        else:
            db_val.add_record(record)
    # pick 900 positives and 500 negatives from tiantan_3000 and put them in train set.
    db_tiantan_3000 = db_tiantan_3000.shuffle()
    def _split_pos_neg(record: dict):
        if   record[EXPERIMENT_CONFIG['output']] == 'yes': return 1
        elif record[EXPERIMENT_CONFIG['output']] == 'no':  return 2
        else: return 3
    db_tiantan_3000_pos, db_tiantan_3000_neg, _ = db_tiantan_3000.binary_split_by_rule(_split_pos_neg)
    npos = db_tiantan_3000_pos.num_records()
    nneg = db_tiantan_3000_neg.num_records()
    keep_pos, keep_neg = 700, 400
    db_train += db_tiantan_3000_pos.keep_first_n(keep_pos)
    db_val += db_tiantan_3000_pos.keep_last_n(npos-keep_pos)
    db_train += db_tiantan_3000_neg.keep_first_n(keep_neg)
    db_val += db_tiantan_3000_neg.keep_last_n(nneg-keep_neg)

    return db_train, db_val, None

def wjy_2hg_split(main_database_obj: Database, EXPERIMENT_CONFIG: dict):

    # step 1: filter dataset
    print('step 1: database initial filtering...')
    print('# records before filtering:', main_database_obj.num_records())
    def check_and_remove_record_without_key(database: Database, key: str):
        def _remove_record(record:dict):
            return True if record[key] == '' else False
        return database.remove_by_rule(_remove_record)
    REQUIRED_KEYS = ['T1', 'T1ce', 'T2', 'T2FLAIR', 'brain_mask', 'autoseg', 'autoseg_posvec', 'radiomics_vec', '2-HG']
    for key in REQUIRED_KEYS:
        main_database_obj = check_and_remove_record_without_key(main_database_obj, key)
    print('# records after filtering:', main_database_obj.num_records())
    print('')
    # split train set:
    db_train, db_val = create_database_for_2HG(), create_database_for_2HG()
    db_tiantan_3000 = create_database_for_2HG()
    for i in range(main_database_obj.num_records()):
        record = main_database_obj.get_record(i)
        data_source = record['data_source']
        if data_source in ['tiantan_cgga', 'tiantan_cgga_new', 'tiantan_duofenzi', 'tiantan_new_OS', 'tiantan_2007-2107']:
            db_train.add_record(record)
        elif data_source in ['tiantan_3000']:
            db_tiantan_3000.add_record(record)
        else:
            db_val.add_record(record)
    # pick 900 positives and 500 negatives from tiantan_3000 and put them in train set.
    db_tiantan_3000 = db_tiantan_3000.shuffle()
    def _split_pos_neg(record: dict):
        if   record[EXPERIMENT_CONFIG['output']] == 'yes': return 1
        elif record[EXPERIMENT_CONFIG['output']] == 'no':  return 2
        else: return 3
    db_tiantan_3000_pos, db_tiantan_3000_neg, _ = db_tiantan_3000.binary_split_by_rule(_split_pos_neg)
    npos = db_tiantan_3000_pos.num_records()
    nneg = db_tiantan_3000_neg.num_records()
    keep_pos, keep_neg = 700, 400
    db_train += db_tiantan_3000_pos.keep_first_n(keep_pos)
    db_val += db_tiantan_3000_pos.keep_last_n(npos-keep_pos)
    db_train += db_tiantan_3000_neg.keep_first_n(keep_neg)
    db_val += db_tiantan_3000_neg.keep_last_n(nneg-keep_neg)

    return db_train, db_val, None


def lch_ki67_split(main_database_obj: Database, EXPERIMENT_CONFIG: dict):
    # step 1: filter dataset
    print('step 1: database initial filtering...')
    print('# records before filtering:', main_database_obj.num_records())
    def check_and_remove_record_without_key(database: Database, key: str):
        def _remove_record(record:dict):
            return True if record[key] == '' else False
        return database.remove_by_rule(_remove_record)
    REQUIRED_KEYS = ['T1', 'T1ce', 'T2', 'T2FLAIR', 'brain_mask', 'autoseg', 'autoseg_posvec', 'radiomics_vec']
    for key in REQUIRED_KEYS:
        main_database_obj = check_and_remove_record_without_key(main_database_obj, key)
    print('# records after filtering:', main_database_obj.num_records())
    print_database_summary(main_database_obj, EXPERIMENT_CONFIG)
    print('')
    # split train set:
    db_train, db_val = create_database_for_tumor_diagnosis(), create_database_for_tumor_diagnosis()
    db_tiantan_3000 = create_database_for_tumor_diagnosis()
    for i in range(main_database_obj.num_records()):
        record = main_database_obj.get_record(i)
        data_source = record['data_source']
        if data_source in ['tiantan_cgga', 'tiantan_cgga_new', 'tiantan_duofenzi', 'tiantan_new_OS', 'tiantan_2007-2107']:
            db_train.add_record(record)
        elif data_source in ['tiantan_3000']:
            db_tiantan_3000.add_record(record)
        else:
            db_val.add_record(record)
    # pick 900 positives and 500 negatives from tiantan_3000 and put them in train set.
    db_tiantan_3000 = db_tiantan_3000.shuffle()
    def _split_pos_neg(record: dict):
        if   record[EXPERIMENT_CONFIG['output']] == 'yes': return 1
        elif record[EXPERIMENT_CONFIG['output']] == 'no':  return 2
        else: return 3
    db_tiantan_3000_pos, db_tiantan_3000_neg, _ = db_tiantan_3000.binary_split_by_rule(_split_pos_neg)
    npos = db_tiantan_3000_pos.num_records()
    nneg = db_tiantan_3000_neg.num_records()
    keep_pos, keep_neg = 700, 400
    db_train += db_tiantan_3000_pos.keep_first_n(keep_pos)
    db_val += db_tiantan_3000_pos.keep_last_n(npos-keep_pos)
    db_train += db_tiantan_3000_neg.keep_first_n(keep_neg)
    db_val += db_tiantan_3000_neg.keep_last_n(nneg-keep_neg)

    # 在原有基础上将训练数据调成1比1
    db_train = db_train.shuffle(9887)
    def _split_func(record):
        if record['Ki67'] == 'yes':
            return 1
        elif record['Ki67'] == 'no':
            return 2
        else:
            return 3
    db_train_pos, db_train_neg, _ = db_train.binary_split_by_rule(_split_func)
    keep_per_class = min( db_train_pos.num_records(), db_train_neg.num_records() )
    db_train_pos = db_train_pos.keep_first_n(keep_per_class)
    db_train_neg = db_train_neg.keep_first_n(keep_per_class)

    # 训练集的样本数量太多，验证集过少，将部分训练数据集的样本挪到验证集
    db_val_add_pos, db_train_new_pos = db_train_pos.split([0.3,0.7])
    db_val_add_neg, db_train_new_neg = db_train_neg.split([0.3,0.7])

    db_train = db_train_new_pos + db_train_new_neg
    db_val += db_val_add_pos + db_val_add_neg

    return db_train, db_val, None

def ljj_old_split(main_database_obj: Database, EXPERIMENT_CONFIG: dict):
    # step 1: filter dataset
    print('step 1: database initial filtering...')
    print('# records before filtering:', main_database_obj.num_records())
    def check_and_remove_record_without_key(database: Database, key: str):
        def _remove_record(record:dict):
            return True if record[key] == '' else False
        return database.remove_by_rule(_remove_record)
    REQUIRED_KEYS = ['T1', 'T1ce', 'T2', 'T2FLAIR', 'sex', 'age', 'brain_mask', 'autoseg', 'autoseg_posvec', 'radiomics_vec']
    for key in REQUIRED_KEYS:
        main_database_obj = check_and_remove_record_without_key(main_database_obj, key)
    print('# records after filtering:', main_database_obj.num_records())

    # step 2: split training and test sets using custom rules
    print('step 2: split training and test sets.')
    def split_train_and_test_set(database: Database, split_ratio = [0.5, 0.5]):
        total_samples = database.num_records()
        num_train_samples = int(total_samples * (split_ratio[0] / (split_ratio[0] + split_ratio[1])))
        def _records_that_must_be_put_in_train_set(record:dict):
            return True if record['data_source'] in ['tiantan_3000', 'tiantan_2007-2107', 'tiantan_cgga', 'tiantan_cgga_new'] else False
        def _select_tiantan_pro_database(record: dict):
            return True if record['data_source'] == 'tiantan_pro' else False
        tiantan_pro_database = database.keep_by_rule(_select_tiantan_pro_database)
        train_database = database.keep_by_rule(_records_that_must_be_put_in_train_set)
        test_database = create_database_for_tumor_diagnosis()
        if train_database.num_records() > num_train_samples:
            print('warning: training set is larger than required size.')
        else:
            num_train_samples_that_still_requires = num_train_samples - train_database.num_records()
            if num_train_samples_that_still_requires > tiantan_pro_database.num_records():
                print('warning: training set is smaller than requested size.')
                train_database += tiantan_pro_database
                tiantan_pro_database = create_database_for_tumor_diagnosis() # empty database
            else:
                tiantan_pro_database = tiantan_pro_database.shuffle()
                train_database += tiantan_pro_database.keep_first_n(num_train_samples_that_still_requires)
                tiantan_pro_database.remove_first_n(num_train_samples_that_still_requires)
        # then put all remaining samples in test set
        def _filter_records_that_are_not_in_train_set(full_database: Database, train_database: Database):
            num_samples = full_database.num_records()
            new_database = create_database_for_tumor_diagnosis()
            for i in range(num_samples):
                record = full_database.get_record(i)
                if record['subject_name'] in train_database.data_dict['subject_name']:
                    continue
                else:
                    new_database.add_record(record)
            return new_database
        test_database += _filter_records_that_are_not_in_train_set(database, train_database)
        return train_database, test_database
    train_database, test_database = split_train_and_test_set(main_database_obj)
    print('# of samples for training/test:', '%d/%d' % (train_database.num_records(), test_database.num_records()))
    val_database = test_database
    test_database = None
    return train_database, val_database, test_database
