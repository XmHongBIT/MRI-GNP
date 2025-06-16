import os
from digicare.utilities.data_io import load_nifti, load_nifti_simple, save_nifti, get_nifti_pixdim, save_nifti_simple
from digicare.utilities.external_call import run_shell
from digicare.utilities.process_parallelism import run_parallel
from digicare.utilities.file_ops import file_exist, gd, gn, mkdir, make_unique_dir, join_path, rm, mv
from digicare.utilities.image_ops import barycentric_coordinate, center_crop, z_score
import numpy as np
from scipy.ndimage import zoom as zoom_image


def _parallel_preprocessor(task_param):
    subject_name, FLAIR_path, additional_images, \
        subject_remarks, output_folder, skull_strip_impl = task_param
    additional_images: list
    mkdir(output_folder)

    print('additional images:', additional_images)

    def _run_N4(in_FLAIR, out_FLAIR):
        wdir = make_unique_dir(basedir=output_folder)
        try:
            run_shell('N4BiasFieldCorrection -d 3 -i %s -o %s -c [50x50x50,0.0] -s 2' % (in_FLAIR, out_FLAIR))
        finally:
            rm(wdir)

    def _run_ROBEX(in_FLAIR, out_mask):
        wdir = make_unique_dir(basedir=output_folder)
        try:
            # check ROBEX installation
            if 'ROBEX_DIR' not in os.environ:
                raise RuntimeError('Environment variable "ROBEX_DIR" not set. '
                    'Please set "ROBEX_DIR" and make sure runROBEX.sh is in '
                    'the given directory.')
            # run ROBEX
            ROBEX_shell = join_path(os.environ['ROBEX_DIR'], 'runROBEX.sh')
            brain_out = join_path(wdir, subject_name + '_brain.nii.gz')
            FLAIR_mask = join_path(wdir, subject_name + '_mask.nii.gz')
            run_shell('%s %s %s %s' % (ROBEX_shell, in_FLAIR, brain_out, FLAIR_mask), print_output=False)
            mv(FLAIR_mask, out_mask)
        finally:
            rm(wdir)

    def _run_z_score(in_FLAIR, in_mask, out_FLAIR):
        wdir = make_unique_dir(basedir=output_folder)
        try:
            data, hdr = load_nifti(in_FLAIR)
            mask, _ = load_nifti(in_mask)
            data_z = z_score(data, mask)
            save_nifti(data_z, hdr, out_FLAIR)
        finally:
            rm(wdir)
    
    def _run_resample(in_FLAIR, additional_imgs, out_FLAIR):
        # resample will ignore header info and replace it with a default header.
        # just treat them as a raw 3D data array.
        pixdim = get_nifti_pixdim(in_FLAIR) # mm
        FLAIR_data = load_nifti_simple(in_FLAIR)
        FLAIR_shape = FLAIR_data.shape
        FLAIR_data = zoom_image(FLAIR_data, pixdim, order=1)
        save_nifti_simple(FLAIR_data, out_FLAIR)

        iouts = []
        for img in additional_imgs:
            iout = join_path(output_folder, gn(img,no_extension=True)+'_resampled.nii.gz')
            img_data = load_nifti_simple(img)
            if img_data.shape != FLAIR_shape:
                raise RuntimeError('image shape mismatch: expected shape is %s, but got %s.' % (str(FLAIR_shape), str(img_data.shape)))
            img_data = zoom_image(img_data, pixdim, order=1)
            save_nifti_simple(img_data, iout)
            iouts.append(iout)

        return iouts

    def _run_centering(in_FLAIR, in_mask, additional_imgs, out_FLAIR):
        cx, cy, cz = barycentric_coordinate(load_nifti_simple(in_mask))
        dat, hdr = load_nifti(in_FLAIR)
        save_nifti( center_crop(dat, [int(cx),int(cy),int(cz)], [256,256,256], default_fill=np.min(dat)), hdr, out_FLAIR)
        iouts = []
        for image in additional_imgs:
            dat, _ = load_nifti(image)
            iout = join_path(output_folder, gn(image,no_extension=True)+'_centered.nii.gz')
            save_nifti( center_crop(dat, [int(cx),int(cy),int(cz)], [256,256,256], default_fill=0.0), hdr, iout)
            iouts.append(iout)
        return iouts

    n4_corrected = join_path(output_folder, 'n4_corrected.nii.gz')
    brain_mask = join_path(output_folder, 'brain_mask.nii.gz')
    z_scored = join_path(output_folder, 'z_score.nii.gz')
    resampled = join_path(output_folder, 'resampled.nii.gz')
    centered = join_path(output_folder, 'centered.nii.gz')

    finish_flag = join_path(output_folder, 'PREPROCESS_FINISHED')
    if file_exist(finish_flag):
        return
    
    print('* N4 correction (subj="%s")...' % subject_name)
    _run_N4(FLAIR_path, n4_corrected)
    print('* skull stripping (subj="%s",impl="ROBEX")...' % subject_name)
    _run_ROBEX(n4_corrected, brain_mask)
    additional_images.append(brain_mask)
    print('* z-score normalization (subj="%s")...' % subject_name)
    _run_z_score(n4_corrected, brain_mask, z_scored)
    print('* resample to 1mm (subj="%s")...' % subject_name)
    iouts = _run_resample(z_scored, additional_images, resampled)
    print('* centering image...')
    _run_centering(resampled, iouts[-1], iouts, centered)

    with open(finish_flag, 'w') as f:
        pass

class ImagePreprocessor:
    def __init__(self, tasks, num_workers, skull_strip_impl = 'ROBEX'):
        '''
        Description
        -----------
        For each input image
        -> Calculate coarse brain mask
        -> N4 bias field correction
        -> z-score intensity normalization
        -> Resample to 1mm^3 voxel size
        -> Centering to 256*256*256 volume
        -> Save outputs to disk

        Parameters
        -----------
        @param tasks
            A list of tuples. Each tuple describes a complete image preprocessing task.
            For example: 
            >>> tasks = [
            >>>     (
            >>>         "subject_name1",                # subject name
            >>>         "/path/to/FLAIR/image1.nii.gz", # path to FLAIR image
            >>>         ["/path/to/additional/img1.nii.gz",     # path to additional images
            >>>          "/path/to/additional/img2.nii.gz",...]
            >>>         "remarks for this subject",     # additional info for this subject
            >>>         "/path/to/output/dir1/",        # output folder
            >>>     ),
            >>>     (
            >>>         "subject_name2", 
            >>>         "/path/to/FLAIR/image2.nii.gz", 
            >>>         ["/path/to/additional/img3.nii.gz",
            >>>          "/path/to/additional/img4.nii.gz",...]
            >>>         "remarks for this subject"
            >>>         "/path/to/output/dir2/",
            >>>     ),
            >>>     ...
            >>> ]
        @param num_workers
            Number of concurrent processes 
        @param skull_strip_impl
            Select a software implementation of skull stripping algorithm. Implementation
            can be one of the following: 
                "ROBEX" -- Robust brain extraction
        '''
        avaliable_skull_strip_impl = ['ROBEX']
        assert skull_strip_impl in avaliable_skull_strip_impl, \
            'Invalid setting for "skull_strip_impl". Should be one of %s.' \
            % str(avaliable_skull_strip_impl)
        self.tasks = tasks
        self.num_workers = num_workers
        self.skull_strip_impl = skull_strip_impl
    
    def launch(self):
        '''
        Launch image preprocessing jobs.
        '''
        processing_tasks = []
        for task in self.tasks:
            task_param = list(task) + [self.skull_strip_impl]
            processing_tasks.append(task_param)
        run_parallel(_parallel_preprocessor, processing_tasks, self.num_workers, 'Preprocessing',
            print_output=True)



