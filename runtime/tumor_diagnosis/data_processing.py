import os
import random
import numpy as np
import scipy
import digicare
from digicare.utilities.file_ops import mkdir, join_path, file_exist
from digicare.utilities.data_io import try_load_nifti, load_nifti, save_nifti, sync_nifti_header, save_nifti_simple, load_nifti_simple
from digicare.utilities.process_parallelism import run_parallel
from digicare.utilities.misc import Checkpoints, printx
from digicare.external_tools.ANTs_group_registration import antsRegistration, antsApplyTransforms
from digicare.utilities.external_call import run_shell
from digicare.utilities.file_ops import gd, gn, rm, mv, cp
from scipy.ndimage import zoom
from digicare.utilities.image_ops import barycentric_coordinate, center_crop, z_score, make_onehot_from_label
from matplotlib.pyplot import imsave
from digicare.utilities.database import Database

def get_tumor_diagnosis_db_keys():
    return [
        # input information
        'data_source',  # where the original data was from 
        'subject_name', # subject full name (possible contains sex and age infomation) 
        
        'T1',           # T1 image path for this subject (empty if not exist)
        'T1ce',         # aka T1c, T1gd (empty if not exist)
        'T2',           # T2 image path for this subject (empty if not exist)
        'T2FLAIR',      # T2 FLAIR image path for this subject (empty if not exist)
        #'DWI',          # DWI image path for this subject (empty if not exist)
        'ADC',          # ADC image path
        
        'IDH',          # IDH mutation status (yes/no)
        '1p/19q',       # 1p/19q co-deletion status (yes/no)
        'MGMT',         # MGMT promoter methylation status (yes/no)
        'TERT',         # TERT C228T or TERT C250T mutation status (yes/no)

        '+7/-10',       # chromosome 7/10 anomaly (yes/no), 7号染色体扩增, 10号染色体缺失
        'ATRX',         # ATRX mutation (yes/no)
        'TP53',         # TP53 mutation (yes/no)
        'CDKN',         # CDKN2A/CDKN2B co-deletion status (yes/no)
        'EGFR',         # EGFR扩增
        'Ki67',

        'tumor_type',   # can be one of the following:
                        # GBM (glioblastoma) 胶质母细胞瘤
                        # AO/AO2/AO3 (anaplastic oligodendroglioma grade 2/3) 少突胶质细胞瘤
                        # AA/AA2/AA3/AA4 (astrocytoma grade 2/3/4) 星形胶质细胞瘤
        
        'WHO',          # WHO grade (2/3/4)
        'radio_status', # 是否放疗（部分通过电话联系确认，部分外接数据)
        'chemo_status', # 是否化疗
        #'radio_chemo_status', # 是否放化疗（部分通过电话联系确认， 部分通过放疗、化疗结果粘贴）

        'OS',           # patient overall survival (in days), 
                        # measuring how long someone lives after starting on a treatment.
        'follow_up',    # dead/alive

        # other information

        'sex',
        'age',
        #'annotation',   # manual tumor segmentation 
        'brain_mask',   # brain mask
        'preprocess_strategy', # data preprocessing strategy (only suitable for raw data)
                                # can be one of the following: "raw", "N4+affine"
                                # NOTE: for data that is already processed to 1mm^3 isotropic 
                                # resolution, you can use "raw" to skip data preprocessing.

        # generated/predicted information
        'autoseg',        # automatic segmentation (whole lesion)
        'autoseg_posvec', # lesion position vector obtained from automatic segmentation, using Hammer Atlas in MNI152 space
        'radiomics_vec',  # radiomics feature vector
        # 
        'ki67_deform',    # estimated ki67 deform data
    ]

def create_database_for_tumor_diagnosis(xlsx_file=None):
    return Database(db_keys=get_tumor_diagnosis_db_keys(), xlsx_file=xlsx_file)

def create_database_for_2HG(xlsx_file=None):
    return Database(db_keys=get_tumor_diagnosis_db_keys() + ['2-HG'], xlsx_file=xlsx_file)

def get_lesion_position_vector_from_segmentation(atlas, segmentation):
    assert atlas.shape == segmentation.shape, 'atlas shape != seg.shape.'
    num_classes = int(np.max(atlas))
    classes = sorted( list(np.unique( (segmentation > 0.5).astype('int32') * atlas.astype('int32') )) )
    if len(classes) == 1 and classes[0] == 0:
        print('warning: no lesion found in image.')
    # remove 0
    pos_vector = np.zeros([num_classes if num_classes > 1 else 1]).astype('int')
    for region_id in range(1, num_classes+1):
        pos_vector[region_id-1] = 1 if region_id in classes else 0
    pos_vector = ','.join([ str(item) for item in list(pos_vector)])
    return pos_vector

class SegmentationFileNotFoundError(Exception): pass

def add_segmentation_info_to_database(database : Database, raise_if_error=False):
    # prepare atlas
    PACKAGE_PATH = gd(digicare.__file__)
    atlas_nifti_path = join_path(PACKAGE_PATH, 'resources', 'T1_mni152_hammer_atlas', 'MNI152_Hammer_atlas.nii.gz')
    atlas_data, _    = load_nifti(atlas_nifti_path)
    # you need to ensure the atlas is in MNI152 standard space (1mm^3)
    atlas_data_bin = (atlas_data > 0.5).astype('float32')
    x, y, z = barycentric_coordinate(atlas_data_bin)
    x, y, z = int(x), int(y), int(z)
    atlas = center_crop( atlas_data, [x,y,z], [256,256,256] )
    # iterate each record in database and add relevant info
    num_records = database.num_records()
    for record_id in range(num_records):
        record = database.get_record(record_id)
        subject_name = record['subject_name']
        print('[%d/%d] %s' % (record_id+1, num_records, subject_name))
        autoseg_file = record['autoseg'] #join_path('Preprocessed_data', subject_name, 'origres', 'segmentation_native_t1ce.nii.gz')
        if file_exist(autoseg_file):
            print('found segmentation: "%s".' % autoseg_file)
            segmentation, _ = load_nifti(autoseg_file)
            record['autoseg_posvec'] = get_lesion_position_vector_from_segmentation(atlas, segmentation)
            print('lesion position vector: %s' % record['autoseg_posvec'])
        else:
            if raise_if_error:
                raise SegmentationFileNotFoundError('Error: segmentation for subject "%s" not found!' % subject_name)
            else:
                print('WARNING: segmentation for subject "%s" not found!' % subject_name)
        database.set_record(record_id, record)
    return database

def preprocess_subject(subject_name, origres_shape, lowres_shape, record, mni152, output_dir, ss_backend):

    def _bias_correction(i, o):
        run_shell('N4BiasFieldCorrection -d 3 -i %s -o %s -c [50x50x50,0.0] -s 2' % (i, o), print_command=False, print_output=False)

    def _affine_registration(i,t,o, interp, deform):
        run_shell(antsRegistration(i, t, o, interpolation_method=interp,deform_type=deform))

    def _apply_affine_transform(i, t, xfm ,o, interp):
        run_shell(antsApplyTransforms(i, t, xfm, o, interpolation_method=interp))

    def _extract_bm(i, o, backend):
        if i == '' or i is None:
            return

        if backend == 'bet':
            b = join_path(gd(i),gn(i,no_extension=True) + '_brain.nii.gz')
            m = join_path(gd(i),gn(i,no_extension=True) + '_brain_mask.nii.gz')
            run_shell('bet %s %s -m -n' % (i, b))
            rm(b)
            mv(m,o)
        elif backend == 'robex':
            ROBEX_sh = join_path(os.environ['ROBEX_DIR'], 'runROBEX.sh')
            b = join_path(gd(i),gn(i,no_extension=True) + '_brain.nii.gz')
            m = join_path(gd(i),gn(i,no_extension=True) + '_brain_mask.nii.gz')
            run_shell('%s %s %s %s' % (ROBEX_sh, i, b, m), print_output=False)
            rm(b)
            mv(m,o)

    def _compress_image_if_needed(i, o):
        data, header = load_nifti(i)
        if np.max(data) - np.min(data) < 255:
            # image dynamic range is too low, don't convert it as integer
            return
        else:
            data = data.astype('int32')
        save_nifti(data, header, o)         

    t1, t2, t1ce, t2flair, adc = record['T1'], record['T2'], record['T1ce'], record['T2FLAIR'], record['ADC']
    anno = '' if 'annotation' not in record else record['annotation']

    origres_folder = mkdir(join_path(output_dir, 'origres'))
    lowres_folder = mkdir(join_path(output_dir, 'lowres'))

    t1_out      = join_path(origres_folder, 't1.nii.gz')
    t1ce_out    = join_path(origres_folder, 't1ce.nii.gz')
    t2_out      = join_path(origres_folder, 't2.nii.gz')
    t2flair_out = join_path(origres_folder, 't2flair.nii.gz')
    adc_out     = join_path(origres_folder, 'adc.nii.gz')
    anno_out    = join_path(origres_folder, 'anno.nii.gz')
    bm_out      = join_path(origres_folder, 'brain_mask.nii.gz')

    do_bias_correction = False
    preprocess_strategy = record['preprocess_strategy']

    if preprocess_strategy == 'raw':
        print('Note: using "raw" database processing strategy. In this strategy, you need to '
              'ensure 1) the image has 1mm^3 isotropic resolution, 2) images are skull-stripped, '
              '3) images are co-registered to the same space, 4) images are bias corrected (N4 '
              'correction).')
        # directly use raw data
        # this applies to data that are already preprocessed (for example, some public dataset only provides preprocessed data)
        if t1 != '':      cp(t1, t1_out) 
        if t2 != '':      cp(t2, t2_out) 
        if t2flair != '': cp(t2flair, t2flair_out) 
        if t1ce != '':    cp(t1ce, t1ce_out) 
        if adc != '':     cp(adc, adc_out) 
        if anno != '':    cp(anno, anno_out)
        bm = record['brain_mask']
        if bm != '':      cp(bm, bm_out)
        else:
            # this dataset do not provide brain mask, we then provide a default brain mask for each image
            # make sure these images are already skull-stripped !!!
            print('making default brain mask for subject "%s"...' % subject_name)
            pivot = t1
            if pivot == '': pivot = t1ce
            if pivot == '': pivot = t2
            if pivot == '': pivot = t2flair
            if pivot == '':
                raise RuntimeError('cannot find image for subject "%s".' % subject_name)
            dat, hdr = load_nifti(pivot)
            save_nifti((dat > 0).astype('float32'), hdr, bm_out)

    elif preprocess_strategy == 'N4+affine':
        # bias correction
        if t1 != '':      _bias_correction(t1, t1_out)           if do_bias_correction else cp(t1, t1_out)
        if t2 != '':      _bias_correction(t2, t2_out)           if do_bias_correction else cp(t2, t2_out)
        if t2flair != '': _bias_correction(t2flair, t2flair_out) if do_bias_correction else cp(t2flair, t2flair_out)
        if t1ce != '':    _bias_correction(t1ce, t1ce_out)       if do_bias_correction else cp(t1ce, t1ce_out)
        if adc != '':     _bias_correction(adc, adc_out)         if do_bias_correction else cp(adc, adc_out)

        # registration
        imgs = []
        if t1 != '':      imgs.append(t1_out)
        if t2flair != '': imgs.append(t2flair_out)
        if t2 != '':      imgs.append(t2_out)
        if t1ce != '':    imgs.append(t1ce_out)
        if adc != '':     imgs.append(adc_out)
        if len(imgs) == 0:
            raise RuntimeError('cannot find a pivot image during registration for subject "%s".' % subject_name)
        # synchronize image header between annotation and image
        if anno != '':    sync_nifti_header( anno, imgs[0],  anno_out)
        # register pivot to template
        _affine_registration(imgs[0], mni152, imgs[0], 'Linear', 'Linear')
        xfm_matrix = join_path(output_dir, 'warp_0GenericAffine.mat')
        if anno != '': _apply_affine_transform(anno_out, mni152, xfm_matrix, anno_out, interp='NearestNeighbor')
        for img in imgs[1:]: _affine_registration(img, mni152, img, 'Linear', 'Linear')
        # convert float to integer to save disk space
        # if dynamic range is too low, don't convert
        for img in imgs: _compress_image_if_needed(img, img)
        # extract brain mask
        _extract_bm(imgs[0], bm_out, ss_backend)

    else:
        raise RuntimeError('unknown preprocessing strategy "%s".' % preprocess_strategy)

    if file_exist(bm_out) == False:
        raise RuntimeError('cannot find brain mask for subject "%s".' % subject_name)

    t1_out_lowres      = join_path(lowres_folder, 't1.nii.gz')
    t1ce_out_lowres    = join_path(lowres_folder, 't1ce.nii.gz')
    t2_out_lowres      = join_path(lowres_folder, 't2.nii.gz')
    t2flair_out_lowres = join_path(lowres_folder, 't2flair.nii.gz')
    adc_out_lowres     = join_path(lowres_folder, 'adc.nii.gz')
    anno_out_lowres    = join_path(lowres_folder, 'anno.nii.gz')
    bm_out_lowres      = join_path(lowres_folder, 'brain_mask.nii.gz')


    def centering(img_in, img_out, bm, shape):
        if img_in is None or img_in == '': return
        # relocate brain to image center using brain mask
        bm_data = load_nifti_simple(bm)
        img_data = load_nifti_simple(img_in)
        x,y,z = barycentric_coordinate(bm_data)
        x,y,z = int(x), int(y), int(z)
        print('img_in: %s, center coord: (%d, %d, %d)' % (img_in, x,y,z), 'crop to:', shape)
        img_data = center_crop(img_data,[x,y,z], shape)
        save_nifti_simple(img_data, img_out)

    print('lowres centering...')
    centering(t1_out,      t1_out_lowres,      bm_out, origres_shape) if t1      != '' else None
    centering(t1ce_out,    t1ce_out_lowres,    bm_out, origres_shape) if t1ce    != '' else None
    centering(t2_out,      t2_out_lowres,      bm_out, origres_shape) if t2      != '' else None
    centering(t2flair_out, t2flair_out_lowres, bm_out, origres_shape) if t2flair != '' else None
    centering(adc_out,     adc_out_lowres,     bm_out, origres_shape) if adc     != '' else None
    centering(bm_out,      bm_out_lowres,      bm_out, origres_shape)
    if anno!='': centering(anno_out, anno_out_lowres, bm_out, origres_shape)

    # scale image to 128x128x128 so that the whole brain can fit into a network
    print('resampling...')

    def resample(img_in, img_out, scale, order = 2):
        # resample image by zoom ratio and save it as the same file
        # as the data has already been registered to a 1mm template we
        # can only load its raw data and ignore its header
        if img_in is None or img_in == '': return
        data, _ = load_nifti(img_in)
        print('zoom scale: %s' % str(scale), 'order: %d' % order)
        data = zoom(data, scale, order = order)
        save_nifti_simple(data, img_out)
        _compress_image_if_needed(img_out, img_out)

    scale = [lowres_shape[0] / origres_shape[0],
             lowres_shape[1] / origres_shape[1],
             lowres_shape[2] / origres_shape[2]]
    resample(t1_out_lowres,      t1_out_lowres,      scale, order = 2) if t1      != '' else None
    resample(t1ce_out_lowres,    t1ce_out_lowres,    scale, order = 2) if t1ce    != '' else None
    resample(t2_out_lowres,      t2_out_lowres,      scale, order = 2) if t2      != '' else None
    resample(t2flair_out_lowres, t2flair_out_lowres, scale, order = 2) if t2flair != '' else None
    resample(adc_out_lowres,     adc_out_lowres,     scale, order = 2) if adc     != '' else None
    resample(bm_out_lowres,      bm_out_lowres,      scale, order = 0)
    if anno!='': resample(anno_out_lowres, anno_out_lowres, scale, order = 0)

    print('origres centering...')
    centering(t1_out,      t1_out,      bm_out, origres_shape) if t1      != '' else None
    centering(t1ce_out,    t1ce_out,    bm_out, origres_shape) if t1ce    != '' else None
    centering(t2_out,      t2_out,      bm_out, origres_shape) if t2      != '' else None
    centering(t2flair_out, t2flair_out, bm_out, origres_shape) if t2flair != '' else None
    centering(adc_out,     adc_out,     bm_out, origres_shape) if adc     != '' else None
    centering(bm_out,      bm_out,      bm_out, origres_shape)
    if anno!='': centering(anno_out, anno_out, bm_out, origres_shape)

    print('generating preview...')
    images_lowres = [t1_out_lowres, t1ce_out_lowres, t2_out_lowres, 
                     t2flair_out_lowres, adc_out_lowres, bm_out_lowres]
    image_plane = np.zeros([lowres_shape[0], len(images_lowres)*lowres_shape[0]]).astype('float32')
    image_id = 0
    for image_lowres in images_lowres:
        if file_exist(image_lowres):
            image_slice = load_nifti_simple(image_lowres)[:,:,lowres_shape[0]//2]
            image_slice = (image_slice - np.min(image_slice)) / (np.max(image_slice) - np.min(image_slice) + 0.001)
            image_slice = np.transpose(image_slice, axes=[1,0])
            image_plane[:, image_id*lowres_shape[0] : (image_id+1)*lowres_shape[0]] = image_slice
            image_id += 1
    preview_image = join_path(output_dir, '%s_preview.png' % subject_name)
    image = np.zeros([image_plane.shape[0], image_plane.shape[1], 3]).astype('float32')
    image[0:image_plane.shape[0], 0:image_plane.shape[1],0] = image_plane
    image[0:image_plane.shape[0], 0:image_plane.shape[1],1] = image_plane
    image[0:image_plane.shape[0], 0:image_plane.shape[1],2] = image_plane
    imsave(preview_image, image)

    print('finished.')

class DatabasePreprocessor():
    @staticmethod
    def check_and_get_ROBEX_sh():
        # check ROBEX
        if "ROBEX_DIR" not in os.environ:
            raise RuntimeError('Environment variable "ROBEX_DIR" is not set. Please download and install ROBEX and set '
                            '"ROBEX_DIR" in your ~/.bashrc')
        ROBEX_folder = os.environ['ROBEX_DIR']
        ROBEX_sh = join_path(ROBEX_folder, 'runROBEX.sh')
        ROBEX_bin = join_path(ROBEX_folder, 'ROBEX')
        if file_exist(ROBEX_sh) == False or file_exist(ROBEX_bin) == False:
            raise RuntimeError("Cannot find 'runROBEX.sh' and 'ROBEX' binary file in "
                "folder '%s', be sure to download and install ROBEX in your local "
                "machine and check the path given is correct." % ROBEX_folder)
        return ROBEX_sh
    @staticmethod
    def _data_preprocessing_worker_func(args):
        '''
        Internal worker function. DO NOT call this directly.
        '''
        subject_name, output_dir, record, template, origres_shape, lowres_shape, ss_backend = args
        subject_preprocess_dir = mkdir( join_path( output_dir, subject_name ) )
        ckpts = Checkpoints(subject_preprocess_dir)
        if not ckpts.is_finished('PREPROCESS_FINISHED'):
            print('processing subject "%s"' % subject_name)
            try:
                preprocess_subject(subject_name, origres_shape, lowres_shape, record, template, subject_preprocess_dir, ss_backend)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print('*** Error(s) occurred during data processing for subject "%s".' % subject_name)
                print(e)
                return False
            else:
                ckpts.set_finish('PREPROCESS_FINISHED')
        return True
    
    def __init__(self, raw_database: Database = None, num_workers = 2, mni152_template = '', 
        origres_shape = [256,256,256], lowres_shape = [128,128,128], ss_backend = 'bet'):
        '''
        Description
        --------------
        Do image preprocessing for each record in database.

        Parameters
        --------------
        origres_shape: target image shape in original resolution (1mm^3)
        lowres_shape: low resolution image shape (directly resized from origres_shape)
        ss_backend: program used to do skull stripping for input images, can be "bet" or "robex"

        '''
        assert raw_database is not None, 'must provide raw dataset.'
        if file_exist(mni152_template) == False:
            raise RuntimeError('MNI152 template "%s" not exist.' % mni152_template)
        self.num_workers = num_workers
        self.mni152_template = mni152_template
        self.dataset = raw_database
        self.keys = raw_database.get_db_keys()
        self.origres_shape = origres_shape
        self.lowres_shape = lowres_shape
        self.ss_backend = ss_backend

    def preprocess_data(self, output_dir):
        print('Starting database preprocessing...')

        self.check_and_get_ROBEX_sh()

        mkdir(output_dir)
        num_subjects = self.dataset.num_records()
        print('total subjects: %d.' % num_subjects)
        tasks = []
        for i in range(num_subjects):
            record = self.dataset.get_record(i)
            subject_name = record['subject_name']
            tasks.append((subject_name, output_dir, record, self.mni152_template, self.origres_shape, self.lowres_shape, self.ss_backend))
        results = run_parallel(self._data_preprocessing_worker_func, tasks, self.num_workers, 'Preprocessing', print_output=True)
        total, error = 0,0
        for result in results:
            total += 1
            if result == False: error += 1
        print('Preprocessing finished. %d in total, %d error(s).' % (total, error))

        # make new dataset
        origres_dataset, lowres_dataset = create_database_for_tumor_diagnosis(), create_database_for_tumor_diagnosis()
        for i in range(num_subjects):
            record = self.dataset.get_record(i)
            # origres
            subject_name = record['subject_name']
            if 'annotation' in record:
                if record['annotation'] != '': record['annotation'] = join_path(output_dir, subject_name, 'origres', 'anno.nii.gz')
            if record['T1'] != '':         record['T1']         = join_path(output_dir, subject_name, 'origres', 't1.nii.gz')
            if record['T1ce'] != '':       record['T1ce']       = join_path(output_dir, subject_name, 'origres', 't1ce.nii.gz')
            if record['T2'] != '':         record['T2']         = join_path(output_dir, subject_name, 'origres', 't2.nii.gz')
            if record['T2FLAIR'] != '':    record['T2FLAIR']    = join_path(output_dir, subject_name, 'origres', 't2flair.nii.gz')
            if record['ADC'] != '':        record['ADC']        = join_path(output_dir, subject_name, 'origres', 'adc.nii.gz')
            record['brain_mask'] = join_path(output_dir, subject_name, 'origres', 'brain_mask.nii.gz')
            origres_dataset.add_record(record)
            # lowres
            record = self.dataset.get_record(i)
            if 'annotation' in record:
                if record['annotation'] != '': record['annotation'] = join_path(output_dir, subject_name, 'lowres', 'anno.nii.gz')
            if record['T1'] != '':         record['T1']         = join_path(output_dir, subject_name, 'lowres', 't1.nii.gz')
            if record['T1ce'] != '':       record['T1ce']       = join_path(output_dir, subject_name, 'lowres', 't1ce.nii.gz')
            if record['T2'] != '':         record['T2']         = join_path(output_dir, subject_name, 'lowres', 't2.nii.gz')
            if record['T2FLAIR'] != '':    record['T2FLAIR']    = join_path(output_dir, subject_name, 'lowres', 't2flair.nii.gz')
            if record['ADC'] != '':        record['ADC']        = join_path(output_dir, subject_name, 'lowres', 'adc.nii.gz')
            record['brain_mask'] = join_path(output_dir, subject_name, 'lowres', 'brain_mask.nii.gz')
            lowres_dataset.add_record(record)

        return origres_dataset, lowres_dataset

    def get_origres_shape(self):
        return self.origres_shape
    
    def get_lowres_shape(self):
        return self.lowres_shape

class DatabaseChecker():
    def __init__(self, database: Database):
        self.database = database

    def get_database_obj(self):
        return self.database        

    def remove_broken_file_records(self, missing_files):
        print('removing broken file links...')
        if len(missing_files) == 0:
            return self
        num_records = self.database.num_records()
        for i in range(num_records):
            record = self.database.get_record(i)
            for key in record:
                if record[key] in missing_files:
                    print('removed broken file link: "%s".' % record[key])
                    record[key] == ''
            self.database.set_record(i, record)
        return self    

    def run_existence_check(self, check_keys = []):
        print('running file existence check...')
        missing_files = []
        num_records = self.database.num_records()
        for i in range(num_records):
            record = self.database.get_record(i)
            for check_key in check_keys:
                if check_key not in record:
                    raise RuntimeError('cannot find required key "%s" in record: %s.' % (check_key, str(record)))
                file_path = record[check_key]
                if len(file_path) == 0:
                    continue
                file_disp = file_path if len(file_path) < 48 else ('...' + file_path[-45:])
                printx('[%d/%d] checking %s' % (i+1, num_records, file_disp))
                missing_or_error = False
                if not file_exist(file_path):
                    missing_or_error = True
                elif file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
                    if try_load_nifti(file_path) == False:
                        missing_or_error = True
                if missing_or_error:
                    missing_files.append(file_path)
                    printx('')
                    print('found a broken file path: "%s".' % file_path)

        printx('')
        if len(missing_files) > 0:
            print('%d file(s) missing:' % len(missing_files))
            for missing_file in missing_files:
                print(missing_file)
        else:
            print('OK, no missing file.')
        return missing_files
        
    def run_shape_check(self):
        print('Checking data...')
        err_subjects = set()
        for i in range(self.database.num_records()):
            record = self.database.get_record(i)
            subject_name = record['subject_name']
            print('[%d/%d] %s' % (i+1, self.database.num_records(), subject_name))
            data_shapes = []
            for file in [ record['T1'], record['T2'], record['T1ce'], record['T2FLAIR'], record['annotation'], record['brain_mask']]:
                if file != '':
                    try:
                        data, _ = load_nifti(file)
                    except KeyboardInterrupt as e:
                        raise e
                    except:
                        print('file "%s" not exist or is not a valid NIFTI file.' % file)
                        err_subjects.add(subject_name)
                    else:
                        data_shapes.append(data.shape)
            if len(data_shapes) == 0:
                print('no NIFTI file found.')
                err_subjects.add(subject_name)
            elif len(data_shapes) == 1:
                continue
            else:
                for shapeid in range(1, len(data_shapes)):
                    if data_shapes[shapeid] != data_shapes[0]:
                        print('shape not compatible %s vs %s.' % (str(data_shapes[0]),str(data_shapes[shapeid]))  )
                        err_subjects.add(subject_name)            
        if len(list(err_subjects) > 0):
            print('The following subject(s) failed the checking:')
            print(list(err_subjects))
        else:
            print('All subject(s) passed the checking.')

def load_images_from_record(
        record:            dict = {},    # database record
        data_augmentation: bool = False, # enable data augmentation when loading image
        aug_params:        dict = {
            'enable_flip'        : True, # enable image flip
            'enable_rotate'      : True, # enable image rotation
            'rotate_angle_range' : 45,   # image rotation angle (degrees)
            'rotate_plane'       : [(0,1), (0,2), (1,2)], # image rotation planes
            'enable_noise'       : True, # enable gaussian noise
            'noise_std_percent'  : 0.1   # gaussian noise amplitude
        },                               # data augmentation parameters
        load_seg_key:       str = 'seg', # load segmentation from 
        load_ki67_estimated_deform_key: str = 'ki67_deform',
        load_mode:          str = '3D',  # image loading mode (2D, 2.5D, 3D)
        slice_step:         int = 5,     # only for 2.5D data loading mode
        slice_offset:       int = 0,     # offset applied when loading image slices
    ):

    '''
    Description
    --------------
    image processing steps used for generating training/test samples. 
    '''

    assert load_mode in ['2D', '2.5D', '3D'], 'Invalid load_mode. Must be one of "2D", "2.5D", or "3D".'

    FLIP_PROB   = 0.7
    ROTATE_PROB = 0.5
    NOISE_PROB  = 0.5

    def _load_nifti(path):
        if path in ['', None]: return None, None
        else: 
            return load_nifti(path)
    def _z_score(image, mask):
        if image is None or mask is None:
            return None
        else:
            return z_score(image, mask)
    def _flip(image, axes):
        if image is None:
            return None
        else:
            return np.flip(image, axes)
    def _rotate(image, plane, angle, order=1):
        if image is None:
            return None
        else:
            return scipy.ndimage.rotate(image, angle=angle,axes=plane, reshape=False,order=order)
    def _noise(image, noise_percentage):
        if image is None:
            return None
        q5, q95 = np.quantile(image, 0.05), np.quantile(image, 0.95)
        std = noise_percentage * (q95 - q5)
        noise = std * np.random.randn(*image.shape)
        return image + noise

    t1,         _ = _load_nifti(record['T1'])
    t1ce,       _ = _load_nifti(record['T1ce'])
    t2,         _ = _load_nifti(record['T2'])
    t2flair,    _ = _load_nifti(record['T2FLAIR'])
    adc,        _ = _load_nifti(record['ADC'])
    anno,       _ = _load_nifti(record[load_seg_key])
    ki67_deform,_ = _load_nifti(record[load_ki67_estimated_deform_key]) if load_ki67_estimated_deform_key in record else None
    brain_mask, _ = _load_nifti(record['brain_mask'])

    subject_name = record['subject_name']

    if brain_mask is not None:
        brain_mask = brain_mask.astype('int32')
        if np.sum(brain_mask) == 0:
            raise RuntimeError('"%s": invalid brain mask given (valid brain region voxels = 0).' % subject_name)
    else:
        raise RuntimeError(
            'cannot load subject "%s" brain mask. record=%s\n'\
            'possible reasons are:\n'\
            '1) record for the brain mask is empty;\n'\
            '2) file is corrupted;\n'\
            '3) file does not exist or have no access.' % (subject_name, str(record)))

    image_shape = None
    if   t1      is not None: image_shape = t1.shape
    elif t1ce    is not None: image_shape = t1ce.shape
    elif t2      is not None: image_shape = t2.shape
    elif t2flair is not None: image_shape = t2flair.shape
    elif adc     is not None: image_shape = adc.shape
    elif anno    is not None: image_shape = anno.shape
    if image_shape == None:
        raise RuntimeError('no input image was found.')
    if len(image_shape) != 3:
        raise RuntimeError('only support 3D image, got input image with shape %s.' % image_shape)
    
    in_modalities = np.sum([(1 if image is not None else 0) for image in [t1, t1ce, t2, t2flair, adc, ki67_deform]])
    if in_modalities == 0:
        # no input image is given
        in_modalities = 1

    # apply data augmentation
    if data_augmentation:
        # random flip
        if np.random.rand() < FLIP_PROB and aug_params['enable_flip']:
            flip_axes = [[], [0], [1], [2], [0,1], [0,2], [1,2], [0,1,2]]
            axes        = random.choice(flip_axes)
            t1          = _flip(t1,         axes)
            t1ce        = _flip(t1ce,       axes)
            t2          = _flip(t2,         axes)
            t2flair     = _flip(t2flair,    axes)
            adc         = _flip(adc,        axes)
            anno        = _flip(anno,       axes)
            brain_mask  = _flip(brain_mask, axes)
            ki67_deform = _flip(ki67_deform, axes)

        # random rotation
        if np.random.rand() < ROTATE_PROB and aug_params['enable_rotate']:
            angle = np.random.randint(-aug_params['rotate_angle_range'],+aug_params['rotate_angle_range'])
            rotate_plane = random.choice(aug_params['rotate_plane'])
            t1         = _rotate(t1,         rotate_plane, angle, order = 1)
            t1ce       = _rotate(t1ce,       rotate_plane, angle, order = 1)
            t2         = _rotate(t2,         rotate_plane, angle, order = 1)
            t2flair    = _rotate(t2flair,    rotate_plane, angle, order = 1)
            adc        = _rotate(adc,        rotate_plane, angle, order = 1)
            anno       = _rotate(anno,       rotate_plane, angle, order = 0)
            brain_mask = _rotate(brain_mask, rotate_plane, angle, order = 0)
            ki67_deform = _rotate(ki67_deform, rotate_plane, angle, order = 1)

        # noise
        if np.random.rand() < NOISE_PROB and aug_params['enable_noise']:
            noise_percentage = aug_params['noise_std_percent']
            t1         = _noise(t1,      noise_percentage)
            t1ce       = _noise(t1ce,    noise_percentage)
            t2         = _noise(t2,      noise_percentage)
            t2flair    = _noise(t2flair, noise_percentage)
            adc        = _noise(adc,     noise_percentage)
            # ki67_deform does not need noise

    # z-score normalization
    t1      = _z_score(t1,      brain_mask)
    t1ce    = _z_score(t1ce,    brain_mask)
    t2      = _z_score(t2,      brain_mask)
    t2flair = _z_score(t2flair, brain_mask)
    adc     = _z_score(adc,     brain_mask)
    # ki67_deform does not need to z-score, we need the raw input

    # pack images
    X = np.zeros([in_modalities, *image_shape])
    i = 0
    for img in [t1, t1ce, t2, t2flair, adc, ki67_deform]:
        if img is not None:
            X[i] = img
            i+=1

    Y = make_onehot_from_label(anno) if anno is not None else None
    C = anno if anno is not None else None
    V = np.sum(np.where(C > 0.5, 1, 0))

    if load_mode == '3D':
        if image_shape[0] == 256 and image_shape[1] == 256 and image_shape[2] == 256:
            # resize to 128x128x128
            def _zoom_4D(data, scale, order = 0):
                channel, x, y, z = data.shape
                data_zoom = None
                for c in range(channel):
                    data_channel = zoom(data[c], scale, order = order)
                    if data_zoom is None:
                        data_zoom = np.zeros([channel,*data_channel.shape])
                    data_zoom[c] = data_channel
                return data_zoom.astype('float32')
            X_zoom = _zoom_4D(X, 0.5, order = 1)
            Y_zoom = _zoom_4D(Y, 0.5, order = 1)
            C_zoom = zoom(C, 0.5, order = 0)
            return X_zoom, Y_zoom, C_zoom, V
        elif image_shape[0] == 128 and image_shape[1] == 128 and image_shape[2] == 128:
            return X, Y, C, V
        else:
            raise RuntimeError('unknown input shape: %s' % image_shape)
    elif load_mode == '2D':
        if anno is None:
            raise RuntimeError('Segmentation mask was not provided!')
        C_mask = np.where(C > 0.5, 1, 0)
        C_mask_sum = np.sum(C_mask, axis=(0,1))
        center_slice = np.argmax(C_mask_sum)
        X_slice = np.zeros([X.shape[0], X.shape[1], X.shape[2]]) # channels x width x height
        Y_slice = np.zeros([Y.shape[0], Y.shape[1], Y.shape[2]])
        C_slice = np.zeros([C.shape[1], C.shape[2]])
        src_slice_index = center_slice + slice_offset
        if 0 <= src_slice_index < X.shape[3]:
            for i in range(X.shape[0]):
                X_slice[i] = X[i,:,:,src_slice_index]
        if 0 <= src_slice_index < Y.shape[3]:
            for i in range(Y.shape[0]):
                Y_slice[i] = Y[i,:,:,src_slice_index]
        C_slice = C[:,:,center_slice]
        return X_slice, Y_slice, C_slice, V
    elif load_mode == '2.5D':
        if anno is None:
            raise RuntimeError('Segmentation mask was not provided!')
        C_mask = np.where(C > 0.5, 1, 0)
        C_mask_sum = np.sum(C_mask, axis=(0,1))
        center_slice = np.argmax(C_mask_sum)
        local_slice_offsets = [(i-3) * slice_step for i in range(7)]
        X_slice = np.zeros([X.shape[0] * len(local_slice_offsets), X.shape[1], X.shape[2]]) # channels x width x height
        Y_slice = np.zeros([Y.shape[0], Y.shape[1], Y.shape[2]])
        C_slice = np.zeros([C.shape[1], C.shape[2]])
        cur_dst_slice_index = 0
        for modality_id in range(X.shape[0]):
            for local_slice_offset in local_slice_offsets:
                src_slice_index = center_slice + local_slice_offset + slice_offset
                if 0 <= src_slice_index < X.shape[3]:
                    X_slice[cur_dst_slice_index] = X[modality_id,:,:,src_slice_index]
                cur_dst_slice_index += 1
        if 0 <= center_slice + slice_offset < Y.shape[3]:
            for i in range(Y.shape[0]):
                Y_slice[i] = Y[i,:,:,center_slice + slice_offset]
        C_slice = C[:,:,center_slice]
        return X_slice, Y_slice, C_slice, V
