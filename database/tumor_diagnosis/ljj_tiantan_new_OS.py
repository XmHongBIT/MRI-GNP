from digicare.utilities.database import Database
from re import L
from digicare.runtime.tumor_diagnosis.data_processing import create_database_for_tumor_diagnosis
from digicare.utilities.data_io import SimpleExcelReader, gz_compress
from digicare.utilities.file_ops import dir_exist, file_exist, join_path, ls, gn, file_size, files_exist
import datetime

class Database_NewOS(Database):
    def __init__(self, xlsx_file=None) -> None:
        super().__init__(xlsx_file)
    def _define_database_structure(self):
        self.db_keys = [
            '文件夹',
            '病历号',
            '手术时间',
            '性别',
            '年龄',
            'T1',
            'T1ce',
            'T2',
            'T2FLAIR',
            'ADC',
            'WHO级别',
            '肿瘤类别',
            'IDH',
            'ATRX',
            'TP53',
            'Ki67',
            'OS',
            '死亡', # yes/no
            '扫描前是否手术', # yes/no
        ]
    def to_super_class(self):
        database = create_database_for_tumor_diagnosis()
        num_records = self.num_records()
        for i in range(num_records):
            record = self.get_record(i)
            target_record = database.make_empty_record()
            target_record['data_source'] = 'tiantan_new_OS'
            target_record['subject_name'] = record['文件夹']
            target_record['T1'] = record['T1']
            target_record['T1ce'] = record['T1ce']
            target_record['T2'] = record['T2']
            target_record['T2FLAIR'] = record['T2FLAIR']
            target_record['ADC'] = record['ADC']
            target_record['OS'] = record['OS']
            target_record['sex'] = 'M' if record['性别'] == '男' else ('F' if record['性别'] == '女' else '')
            target_record['age'] = record['年龄']
            target_record['follow_up'] = 'dead' if record['死亡'] == 'yes' else ('alive' if record['死亡'] == 'no' else '')
            target_record['preprocess_strategy'] = 'N4+affine'
            target_record['IDH'] = record['IDH']
            target_record['Ki67'] = record['Ki67']

            database.add_record(target_record)
        return database

def parse_new_OS():
    database = Database_NewOS()

    list0ori = SimpleExcelReader('xlsx/new_os/LIST0ori.xlsx')
    kgroup = SimpleExcelReader('xlsx/new_os/clinicalinfo_K_group - 副本.xlsx')
    fufawai = SimpleExcelReader('xlsx/new_os/复发队列之外患者信息 - 副本.xlsx')

    def _read(obj:SimpleExcelReader, row:int, col:int, accepted_values=None):
        if obj is list0ori: sheet = 'finaljj'
        elif obj is kgroup: sheet = 'Sheet3'
        elif obj is fufawai: sheet = 'Sheet2'
        else: 
            raise RuntimeError('unknown excel reader object.')
        cell = obj.read((row,col),sheet)
        cell = '' if cell in [None, ''] else str(cell)
        if accepted_values is not None and cell not in accepted_values:
            return ''
        else:
            return cell

    os_info = {}
    for line in range(1, kgroup.max_row('Sheet3')): # skip table head
        # blh, ssrq -> OS, 存活, WHO, 肿瘤类别, IDH, Ki67
        key = (_read(kgroup, line, 0) + _read(kgroup, line, 2)).replace('-','/')
        key = key.split(' ')[0]
        os_info[key] = [
            _read(kgroup, line, 15),
            _read(kgroup, line, 16),
            _read(kgroup, line, 17),
            _read(kgroup, line, 8),
            '',
            _read(kgroup, line, 19),
        ]
    for line in range(1, fufawai.max_row('Sheet2')): # skip table head
        # blh, ssrq -> OS, 存活, WHO, 肿瘤类别, IDH, Ki67
        key = (_read(fufawai, line, 0) + _read(fufawai, line, 2)).replace('-', '/')
        key = key.split(' ')[0]
        os_info[key] = [
            _read(fufawai, line, 6),
            _read(fufawai, line, 7),
            '',
            _read(fufawai, line, 8),
            _read(fufawai, line, 3),
            ''
        ]
    
    image_dir = '/hd/wjy/data_process/lijunjie/glioma_tumor_multi_gene/daixiexiaokang_nii2/'

    def find_file_with_largest_size(files: list):
        ms, f = 0, ''
        for t in files:
            if file_size(t) > ms:
                f = t
                ms = file_size(t)
        return f
    def find_T1_from_folder(folder):
        files = ls(folder, full_path=True)
        filtered = []
        for file in files:
            filename = gn(file, no_extension=True).lower()
            if 't1' in filename:
                if 'regist' in filename:
                    continue
                if 't1c' not in filename and 't1ce' not in filename:
                    filtered.append(file)
        return find_file_with_largest_size(filtered)
    def find_T1ce_from_folder(folder):
        files = ls(folder, full_path=True)
        filtered = []
        for file in files:
            filename = gn(file, no_extension=True).lower()
            if 't1c' in filename or 't1ce' in filename:
                if 'regist' in filename:
                    continue
                else:
                    filtered.append(file)
        return find_file_with_largest_size(filtered)
    def find_T2_from_folder(folder):
        files = ls(folder, full_path=True)
        filtered = []
        for file in files:
            filename = gn(file, no_extension=True).lower()
            if 't2' in filename:
                if 'regist' in filename:
                    continue
                if 'flair' not in filename:
                    filtered.append(file)
        return find_file_with_largest_size(filtered)
    def find_T2FLAIR_from_folder(folder):
        files = ls(folder, full_path=True)
        filtered = []
        for file in files:
            filename = gn(file, no_extension=True).lower()
            if 'flair' in filename:
                if 'regist' in filename:
                    continue
                else:
                    filtered.append(file)
        return find_file_with_largest_size(filtered)
    def find_ADC_from_folder(folder):
        files = ls(folder, full_path=True)
        filtered = []
        for file in files:
            filename = gn(file, no_extension=True).lower()
            if 'adc' in filename:
                if 'regist' in filename:
                    continue
                else:
                    filtered.append(file)
        return find_file_with_largest_size(filtered)

    for line in range(1, list0ori.max_row('finaljj')): # skip table head
        record = database.make_empty_record()
        record['文件夹'] = _read(list0ori, line, 6)
        if len(record['文件夹']) < 4: # skip empty string
            continue
        record['年龄'] = _read(list0ori, line, 5).replace('岁','')
        record['性别'] = _read(list0ori, line, 7, ['男','女'])
        record['性别'] = _read(list0ori, line, 7, ['男','女'])
        record['病历号'] = _read(list0ori, line, 0)
        record['手术时间'] = _read(list0ori, line, 1).replace('-','/').replace('_','/').split(' ')[0]
        record['扫描前是否手术'] = _read(list0ori, line, 31, ['0', '1']).replace('0', 'no').replace('1', 'yes')
        key = (record['病历号'] + record['手术时间']).split(' ')[0]
        if key in os_info:
            print(key)
            try:
                record['OS'] = str(30 * int(float(os_info[key][0])))
            except ValueError:
                record['OS'] = ''
            record['死亡'] = 'yes' if os_info[key][1] == '1' else ('no' if os_info[key][1] == '0' else '')
            try:
                record['WHO级别'] = str(int(os_info[key][2]))
            except ValueError:
                record['WHO级别'] = ''
            record['肿瘤类别'] = os_info[key][3]
            record['IDH'] = 'yes' if os_info[key][4] == '1' else ('no' if os_info[key][4] == '0' else '')
            try:
                ki67 = float(os_info[key][5])
                record['Ki67'] = 'yes' if ki67 > 0.1 else 'no'
            except:
                record['Ki67'] = ''
        if len(record['文件夹']) > 0:
            record['T1']   = find_T1_from_folder(join_path(image_dir, record['文件夹']))
            record['T1ce'] = find_T1ce_from_folder(join_path(image_dir, record['文件夹']))
            record['T2']   = find_T2_from_folder(join_path(image_dir, record['文件夹']))
            record['T2FLAIR'] = find_T2FLAIR_from_folder(join_path(image_dir, record['文件夹']))
            record['ADC'] = find_ADC_from_folder(join_path(image_dir, record['文件夹']))

        database.add_record(record)

    database.export_xlsx('xlsx/new_os/new_os.xlsx', left_freezed_cols=0)

    return database.to_super_class()


if __name__ == '__main__':
    parse_new_OS()




















