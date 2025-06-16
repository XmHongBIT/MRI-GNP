from digicare.utilities.database import Database
from digicare.runtime.tumor_diagnosis.data_processing import create_database_for_tumor_diagnosis
from digicare.utilities.data_io import SimpleExcelReader, gz_compress
from digicare.utilities.file_ops import dir_exist, file_exist, join_path, ls, gn, file_size, files_exist, mkdir
from digicare.utilities.misc import printx
import datetime
import numpy as np

def _parse_base_info() -> Database:
    database   = create_database_for_tumor_diagnosis()
    data_path  = '/hd/wjy/data_process/lijunjie/glioma_tumor_multi_gene/GBM_10000/'
    xlsx_path  = './xlsx/ljj_final/胶质瘤多肿瘤汇总.xlsx'
    sheet_name = 'Sheet1'
    col_info = {
        'data_source':  1,
        'subject_name': 2,
        'sex': 5,
        'age': 6,
        'WHO': 7,
        'tumor_type': 8,
        'IDH':9,
        '1p/19q':10,
        'MGMT':11,
        'TERT':12,
        'EGFR':13,
        '+7/-10':14,
        'CDKN':15,
        'TP53':17,
        'ATRX':18,
        'OS': 69,
        'follow_up':70,
        'radio_status':71,
        'chemo_status':72,
    }

    print('preparing to parse data...')

    xlsx_file  = SimpleExcelReader(xlsx_path)
    max_row = xlsx_file.max_row(worksheet_name=sheet_name)

    def _read(row, col):
        return str(xlsx_file.read((row,col),worksheet_name=sheet_name))

    for i in range(1, max_row): # skip table head
        #print('parsing [%d/%d]' % (i+1, max_row))
        record = database.make_empty_record()
        # data source
        record['data_source'] = _read(i, col_info['data_source'])
        if record['data_source'] == 'dognzong': record['data_source'] = 'Tiantan_dongzong'
        else:
            if record['data_source'] in ['2007-2107', '3000', 'anhui', 'CGGA', 'CGGA_NEW', 'duofenzi',
                'fujianxiehe', 'guangzhoudiyi', 'huanhu', 'HUASHAN', 'landaeryuan', 'nanchang', 'PRO',
                'sanbonaoke', 'shandongyingxiang', 'yuhuangding']:
                record['data_source'] = 'Tiantan_' + record['data_source']
        record['data_source'] = record['data_source'].lower()
        # subject name
        record['subject_name'] = _read(i, col_info['subject_name'])
        subjdir = join_path(data_path, record['subject_name'])
        if not dir_exist(subjdir):
            print('ignore subject "%s" because no subjdir was found.' % subjdir)
            continue

        # sex
        sex = _read(i, col_info['sex'])
        record['sex'] = 'F' if sex == '0' else ('M' if sex == '1' else '')
        # age
        age = _read(i, col_info['age'])
        try:
            int(age)
        except ValueError:
            age = ''
        except:
            raise
        record['age'] = age
        # WHO
        record['WHO'] = _read(i, col_info['WHO'])
        if record['WHO'] not in ['1', '2', '3', '4']:
            record['WHO'] = ''
        # tumor_type
        record['tumor_type'] = _read(i, col_info['tumor_type'])
        if record['tumor_type'] not in ['AA', 'AO', 'GBM', 'GBM(MOL)', 'NEC', 'NOS']:
            record['tumor_type'] = ''
        if record['tumor_type'] == 'GBM(MOL)':
            record['tumor_type'] = 'GBM'
        # IDH
        gs = _read(i, col_info['IDH'])
        record['IDH'] = 'yes' if gs=='1' else ('no' if gs=='0' else '')
        # 1p/19q
        gs = _read(i, col_info['1p/19q'])
        record['1p/19q'] = 'yes' if gs=='1' else ('no' if gs=='0' else '')
        # MGMT
        gs = _read(i, col_info['MGMT'])
        record['MGMT'] = 'yes' if gs=='1' else ('no' if gs=='0' else '')
        # TERT
        gs = _read(i, col_info['TERT'])
        record['TERT'] = 'yes' if gs=='1' else ('no' if gs=='0' else '')
        # EGFR
        gs = _read(i, col_info['EGFR'])
        record['EGFR'] = 'yes' if gs=='1' else ('no' if gs=='0' else '')
        # +7/-10
        gs = _read(i, col_info['+7/-10'])
        record['+7/-10'] = 'yes' if gs=='1' else ('no' if gs=='0' else '')
        # CDKN
        gs = _read(i, col_info['CDKN'])
        record['CDKN'] = 'yes' if gs=='1' else ('no' if gs=='0' else '')
        # ATRX
        gs = _read(i, col_info['ATRX'])
        record['ATRX'] = 'yes' if gs=='1' else ('no' if gs=='0' else '')
        # ATRX
        gs = _read(i, col_info['TP53'])
        record['TP53'] = 'yes' if gs=='1' else ('no' if gs=='0' else '')
        # OS
        OS = _read(i, col_info['OS'])
        try:
            int(OS)
        except ValueError:
            OS = ''
        except:
            raise
        record['OS'] = OS
        # follow_up
        follow_up = _read(i, col_info['follow_up'])
        record['follow_up'] = 'dead' if follow_up == '1' else ('alive' if follow_up == '0' else '')
        # radio_status
        radio = _read(i, col_info['radio_status'])
        record['radio_status'] = 'yes' if radio in ['1', 'Yes', '有'] else ( 'no' if radio in ['0', 'No', '无'] else '' )
        # radio_status
        chemo = _read(i, col_info['chemo_status'])
        record['chemo_status'] = 'yes' if chemo in ['1', 'Yes', '有'] else ( 'no' if chemo in ['0', 'No', '无'] else '' )

        def _pick_file_with_max_size(files):
            maxsize = 0
            maxi = 0
            for i, file in zip(range(len(files)),files):
                if file_size(file) > maxsize:
                    maxsize = file_size(file)
                    maxi = i
            return files[maxi] if len(files) > 0 else ''
        def _make_gzipped_nifti(file):
            if file == '':
                return file 
            if file[-7:] == '.nii.gz':
                return file
            elif file[-4:] == '.nii':
                gzfile = file + '.gz'
                if not file_exist(gzfile):
                    gz_compress(file, gzfile)
                    print('compressing "%s" to "%s"...' % (file, gzfile))
                return gzfile
            else:
                raise RuntimeError('cannot gzip non nifti file "%s".' % file)
        def _filter_t1(folder):
            l = []
            for item in ls(folder, full_path=True):
                if not file_exist(item): continue
                fn = gn(item).lower()
                if 't1' in fn and 't1c' not in fn:
                    if item[-4:]=='.nii':
                        l.append(_make_gzipped_nifti(item))
                    elif item[-7:] == '.nii.gz':                    
                        l.append(item)
                    else:
                        raise RuntimeError('found a file with unknown file extension: "%s".' % item)            
            return _pick_file_with_max_size(l)
        def _filter_t1ce(folder):
            l = []
            for item in ls(folder, full_path=True):
                if not file_exist(item): continue
                fn = gn(item).lower()
                if 't1c' in fn or 't1gd' in fn:
                    if item[-4:]=='.nii':
                        l.append(_make_gzipped_nifti(item))
                    elif item[-7:] == '.nii.gz':
                        l.append(item)
                    else:
                        raise RuntimeError('found a file with unknown file extension: "%s".' % item)            
            return _pick_file_with_max_size(l)
        def _filter_t2(folder):
            l = []
            for item in ls(folder, full_path=True):
                if not file_exist(item): continue
                fn = gn(item).lower()
                if 't2' in fn and 'flair' not in fn:
                    if item[-4:]=='.nii':
                        l.append(_make_gzipped_nifti(item))
                    elif item[-7:] == '.nii.gz':                    
                        l.append(item)
                    else:
                        raise RuntimeError('found a file with unknown file extension: "%s".' % item)            
            return _pick_file_with_max_size(l)
        def _filter_t2flair(folder):
            l = []
            for item in ls(folder, full_path=True):
                if not file_exist(item): continue
                fn = gn(item).lower()
                if 'flair' in fn:
                    if item[-4:]=='.nii':
                        l.append(_make_gzipped_nifti(item))
                    elif item[-7:] == '.nii.gz':                    
                        l.append(item)
                    else:
                        raise RuntimeError('found a file with unknown file extension: "%s".' % item)            
            return _pick_file_with_max_size(l)
        def _filter_adc(folder):
            l = []
            for item in ls(folder, full_path=True):
                if not file_exist(item): continue
                fn = gn(item).lower()
                if 'adc' in fn:
                    if item[-4:]=='.nii':
                        l.append(_make_gzipped_nifti(item))
                    elif item[-7:] == '.nii.gz':                    
                        l.append(item)
                    else:
                        raise RuntimeError('found a file with unknown file extension: "%s".' % item)            
            return _pick_file_with_max_size(l)

        def _assert_file_exist(file):
            if file == '': return file
            assert file_exist(file), 'file "%s" not exist.' % file
            return file
        # T1
        record['T1'] = _assert_file_exist(_filter_t1(subjdir))
        # T1ce
        record['T1ce'] = _assert_file_exist(_filter_t1ce(subjdir))
        # T1ce
        record['T2'] = _assert_file_exist(_filter_t2(subjdir))
        # T1ce
        record['T2FLAIR'] = _assert_file_exist(_filter_t2flair(subjdir))
        # ADC
        record['ADC'] = _assert_file_exist(_filter_adc(subjdir))
        # 去除TCGA的术后图像
        yuanfa_fufa = _read(i, 4)
        if 'tcga' in record['data_source'] and yuanfa_fufa != '0':
            # 仅保留TCGA中原发复发字段为0（原发）的病人
            print('warning: skipped subject "%s"' % record['subject_name'])
            continue

        # preprocess_strategy
        if 'tiantan' in record['data_source']:
            record['preprocess_strategy'] = 'N4+affine'
        elif 'brats' in record['data_source']:
            record['preprocess_strategy'] = 'raw'
        elif 'tcga' in record['data_source']:
            record['preprocess_strategy'] = 'N4+affine'
        elif 'ucsf' in record['data_source']:
            record['preprocess_strategy'] = 'raw'
        elif 'upenn' in record['data_source']:
            record['preprocess_strategy'] = 'N4+affine'
        elif 'ivy' in record['data_source']:
            record['preprocess_strategy'] = 'N4+affine'
        elif 'lgg' in record['data_source']:
            record['preprocess_strategy'] = 'N4+affine'
        elif 'rembrandt' in record['data_source']:
            record['preprocess_strategy'] = 'N4+affine'

        database.add_record(record)
    
    return database

def _add_OS_info(database: Database) -> Database:
    additional_OS_xlsx = './xlsx/ljj_final/os_final1.xlsx'
    sheet_name = 'Sheet1'
    xlsx_file  = SimpleExcelReader(additional_OS_xlsx)

    def _read(row, col):
        return str(xlsx_file.read((row,col),worksheet_name=sheet_name))
    
    def _parse_OS(content):
        try:
            float(content)
        except ValueError:
            return ''
        else:
            return str(int(float(content) * 30))

    def _read_OS_xlsx():
        d = {}
        num_rows = xlsx_file.max_row(worksheet_name=sheet_name)
        for i in range(1, num_rows):
            subjname = _read(i, 0)
            OS_days = _parse_OS(_read(i, 32))
            is_alive = 'dead' if _read(i, 31) == '0' else ('alive' if _read(i, 31)=='1' else '')
            d[subjname] = {'OS_days': OS_days, 'is_alive': is_alive}
        return d

    OS_info = _read_OS_xlsx()
    num_records = database.num_records()
    for i in range(num_records):
        record = database.get_record(i)
        subjname = record['subject_name']
        if subjname in OS_info:
            record['OS'] = str(OS_info[subjname]['OS_days'])
            record['follow_up'] = str(OS_info[subjname]['is_alive'])
        # manipulate record
        database.set_record(i, record)
    return database

def _add_additional_gene_info(database : Database) -> Database:
    gene_info_files = [
        './xlsx/ljj_final/ID0-finished-30001.xlsx',
        './xlsx/ljj_final/ID0-finished-30002.xlsx',
        './xlsx/ljj_final/ID0-finished-20072107.xlsx',
        './xlsx/ljj_final/ID0-finished-duofenzi.xlsx',
        './xlsx/ljj_final/ID0-finished-pro.xlsx'
    ]
    sheetname = 'Sheet1'

    def _read(xlsx, row, col):
        return str(xlsx.read((row,col),worksheet_name=sheetname))
    
    def _is_valid_date_string(s):
        try:
            date = datetime.datetime.strptime(s, '%Y-%m-%d')
        except ValueError:
            return False
        else:
            return True

    def _datediff(datestr1, datestr2, takeabs = False):
        d1 = datetime.datetime.strptime(datestr1, '%Y-%m-%d')
        d2 = datetime.datetime.strptime(datestr2, '%Y-%m-%d')
        daydiff = (d2-d1).days
        if takeabs and daydiff < 0: daydiff = -daydiff
        return daydiff
    
    def _parse_gene_info_files() -> dict:
        ginfo = {}
        log = ''
        def _parse_gene_info_file(file:str, log:str) -> dict:
            print('parsing "%s"...' % file)
            # return: info[住院号] = {
            #   [
            #       (大致检测日期1, 免疫组化描述1),
            #       (大致检测日期2, 免疫组化描述2),
            #       (大致检测日期3, 免疫组化描述3),
            #       ...
            #       (大致检测日期N, 免疫组化描述N),
            #   ]
            # }
            xlsx = SimpleExcelReader(file)
            num_rows = xlsx.max_row(worksheet_name=sheetname)
            for line in range(1, num_rows):
                zhuyuanhao = _read(xlsx, line, 1) # 住院号
                jianceriqi = _read(xlsx, line, 3) # 检测日期 yyyy-mm-dd
                mianyizuhua = _read(xlsx, line, 8) # 免疫组化
                # 有时候检测日期是错的，遇到这种情况就先提醒一下跳过
                if _is_valid_date_string(jianceriqi) == False:
                    log += '无法从住院号"%s"中解析出有效的检测日期，给出的检测日期是"%s".\n' % (zhuyuanhao, jianceriqi)
                    continue
                print(zhuyuanhao, jianceriqi, mianyizuhua)
                if zhuyuanhao not in ginfo:
                    # 这个住院号没有，则新建
                    ginfo[zhuyuanhao] = [[jianceriqi, mianyizuhua]]
                else:
                    oldentry = -1
                    for entryid, (prev_jianceriqi, prev_mianyizuhua) in zip(range(len(ginfo[zhuyuanhao])), ginfo[zhuyuanhao]):
                        if _datediff(prev_jianceriqi, jianceriqi, takeabs=True) < 100:
                            oldentry = entryid
                            break
                    if oldentry < 0:
                        # 检测日期差的有点远，视为不同的检测，可能是复发
                        ginfo[zhuyuanhao].append([jianceriqi, mianyizuhua])
                    else:
                        # 如果检测日期间隔很小，视为同一批检测
                        ginfo[zhuyuanhao][entryid][1] += mianyizuhua
            return log
        for gene_file in gene_info_files:
            log = _parse_gene_info_file(gene_file, log)
        return ginfo, log

    ginfo, log = _parse_gene_info_files()
    print(ginfo)
    print('\n\n%s\n\n' % log)

    def _delete_duplicates():
        for zhuyuanhao in ginfo:
            num_entries = len(ginfo[zhuyuanhao])
            if num_entries > 1:
                print(num_entries, ginfo[zhuyuanhao])
                print('before',ginfo[zhuyuanhao])
            selected_entry = 0
            jianceriqi0 = ginfo[zhuyuanhao][0][0]
            for entryid, (jianceriqi1, _) in zip(range(len(ginfo[zhuyuanhao])), ginfo[zhuyuanhao]):
                if _datediff(jianceriqi0, jianceriqi1, takeabs=False) < 0:
                    # 这个检测日期是更早的，换成这个
                    jianceriqi0 = jianceriqi1
                    selected_entry = entryid
            jianceriqi, mianyizuhua = ginfo[zhuyuanhao][selected_entry]
            ginfo[zhuyuanhao] = (jianceriqi, mianyizuhua)
            if num_entries > 1:
                print('after',ginfo[zhuyuanhao])

    _delete_duplicates()

    def _parse_gene_status_from_info():
        def _split_by_separators(s:str, seps:list):
            def __split(words:list, sep:str):
                subl = []
                for word in words:
                    l = word.split(sep)
                    subl += l
                return subl
            words = [s]
            for sep in seps:
                words = __split(words, sep)
            return words
        def _remove_meaningless_words(words:list):
            l = [word.strip() for word in words if len(word.strip()) > 0]
            l = [word for word in l if word not in ['：', '免疫组化：']]
            return l

        def _parse_description(desc:str):
            # 将免疫组化描述信息整理为结构化的信息
            parts = _remove_meaningless_words(_split_by_separators(desc, [' ', ',', '，', ';', '；', '.', '。']))
            return parts
            
        for zhuyuanhao in ginfo:
            jianceriqi, mianyizuhua = ginfo[zhuyuanhao]
            parts = _parse_description(mianyizuhua)
            ginfo[zhuyuanhao] = parts

    _parse_gene_status_from_info()
    
    # ginfo存储的是住院号到免疫组化结果的映射
    # 此时开始正式的添加信息到现有数据库中
    def _get_zhuyuanhao_mapping():
        # 尝试找到这个记录的住院号
        base_info_xlsx = './xlsx/ljj_final/胶质瘤多肿瘤汇总.xlsx'
        base_xlsx = SimpleExcelReader(base_info_xlsx)
        d = {}
        num_rows = base_xlsx.max_row(worksheet_name='Sheet1')
        for i in range(1, num_rows):
            subjname = str(base_xlsx.read((i, 2),worksheet_name='Sheet1'))
            yuanfa_fufa = str(base_xlsx.read((i, 4),worksheet_name='Sheet1'))
            if yuanfa_fufa == '1':
                print('skip', subjname)
                #复发，跳过
                continue
            zhuyuanhao = str(base_xlsx.read((i, 74),worksheet_name='Sheet1'))
            d[subjname] = zhuyuanhao
        return d
    zhuyuanhao_mapping = _get_zhuyuanhao_mapping()
    def _try_add_gene_status_to_record(record):
        subjname = record['subject_name']
        ki67_log = ''
        if subjname not in zhuyuanhao_mapping:
            return ''
        if zhuyuanhao_mapping[subjname] in ginfo:
            zhuyuanhao = zhuyuanhao_mapping[subjname]
            mianyizuhua = ginfo[zhuyuanhao]
            def _remove_unrelated_info(s):
                words = [
                    'Ki67', 'Ki-67', 'KI67', 'KI-67',
                    'P53', 'TP53', 'p53', 'tp53',
                    '1号片','2号片','3号片','4号片','5号片','6号片','7号片','8号片','9号片',
                    '1号','2号','3号','4号','5号','6号','7号','8号','9号',
                    '蜡块1号','蜡块2号','蜡块3号','蜡块4号','蜡块5号','蜡块6号','蜡块7号','蜡块8号','蜡块9号',
                    '冰1','冰2','冰3','冰4','冰5','冰6','冰7','冰8','冰9',
                ]
                for word in words:
                    s = s.replace(word, '*' * len(word))
                return s
            def _keep_numeric_value(s):
                s0 = ''
                for ch in s:
                    if ord('0') <= ord(ch) <= ord('9'):
                        s0 += ch
                    else:
                        s0 += '*'
                return s0

            def _parse_Ki67():
                w = ''
                for word in mianyizuhua:
                    if 'Ki-67' in word or 'Ki67' in word:
                        w += word
                vals = _keep_numeric_value(_remove_unrelated_info(w))
                values = [float(item) for item in vals.split('*') if len(item)>0]
                ki67 = str(int(np.max(values))) + '%' if len(values) > 0 else ''
                return w, vals, ki67

            def _parse_TP53():
                w = ''
                for word in mianyizuhua:
                    if 'p53' in word.lower():
                        w += word
                vals = _keep_numeric_value(_remove_unrelated_info(w))
                z = [float(item) for item in vals.split('*') if len(item)>0]
                max_value = int(np.max(z)) if len(z) > 0 else None
                tp53 = ''
                if max_value == None:
                    if '++' in w or '+++' in w:
                        tp53 = 'yes'
                    elif '+' in w or '±' in w or '-' in w:
                        tp53 = 'no'
                elif max_value <= 10:
                    tp53 = 'no'
                elif max_value > 10:
                    tp53 = 'yes'
                return w, tp53

            def _parse_ATRX():
                w = ''
                for word in mianyizuhua:
                    if 'atrx' in word.lower():
                        w += word
                vals = _keep_numeric_value(_remove_unrelated_info(w))
                z = [float(item) for item in vals.split('*') if len(item)>0]
                max_value = int(np.max(z)) if len(z) > 0 else None
                atrx = ''
                if max_value == None:
                    if '++' in w or '+++' in w:
                        atrx = 'yes'
                    elif '+' in w or '±' in w or '-' in w:
                        atrx = 'no'
                elif max_value <= 10:
                    atrx = 'no'
                elif max_value > 10:
                    atrx = 'yes'
                return w, atrx

            # ki67
            w, vals, ki67 = _parse_Ki67()
            if ki67 != '':
                ki67_value = int(ki67.replace('%',''))
                if ki67_value <= 10:
                    record['Ki67'] = 'no'
                else:
                    record['Ki67'] = 'yes'
            else:
                record['Ki67'] = ''
            ki67_log += '%s, %s, "%s", "%s", %s, "%s"' % (subjname, zhuyuanhao, w, vals, ki67, record['Ki67'])
            print(ki67_log)
            # p53
            w, tp53 = _parse_TP53()
            if record['TP53'] == '':
                record['TP53'] = tp53
            print(w, tp53)
            # ATRX
            w, atrx = _parse_ATRX()
            if record['ATRX'] == '':
                record['ATRX'] = atrx
            print(w, atrx)
        return ki67_log
        
    num_records = database.num_records()
    for i in range(num_records):
        record = database.get_record(i)
        ki67_log = _try_add_gene_status_to_record(record)
        database.set_record(i, record)
    return database

def parse_ljj_final_organize():
    database = create_database_for_tumor_diagnosis()
    database = _parse_base_info()
    database = _add_OS_info(database)
    database = _add_additional_gene_info(database)
    return database
