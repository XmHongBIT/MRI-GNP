import os
import random
import numpy as np
import scipy
import digicare
import pandas as pd 
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

# ====== 提取病灶位置向量 ======
def get_lesion_position_vector_from_segmentation(atlas, segmentation):
    assert atlas.shape == segmentation.shape, 'atlas shape != seg.shape.'
    num_classes = int(np.max(atlas))
    classes = sorted(list(np.unique((segmentation > 0.5).astype('int32') * atlas.astype('int32'))))
    if len(classes) == 1 and classes[0] == 0:
        print('warning: no lesion found in image.')
    pos_vector = np.zeros([num_classes if num_classes > 1 else 1]).astype('int')
    for region_id in range(1, num_classes + 1):
        pos_vector[region_id - 1] = 1 if region_id in classes else 0
    pos_vector = ','.join([str(item) for item in list(pos_vector)])
    return pos_vector

# ====== 主处理函数 ======
def process_excel_autoseg_vector(xlsx_path, output_path=None):
    # 读取Excel
    df = pd.read_excel(xlsx_path)

    # 加载atlas（只加载一次）
    PACKAGE_PATH = gd(digicare.__file__)
    atlas_nifti_path = join_path(PACKAGE_PATH, 'resources', 'T1_mni152_hammer_atlas', 'MNI152_Hammer_atlas.nii.gz')
    atlas_data, _ = load_nifti(atlas_nifti_path)
    atlas_data_bin = (atlas_data > 0.5).astype('float32')
    x, y, z = barycentric_coordinate(atlas_data_bin)
    x, y, z = int(x), int(y), int(z)
    atlas_crop = center_crop(atlas_data, [x, y, z], [256, 256, 256])

    # 遍历每一行
    for idx, row in df.iterrows():
        seg_path = row.get('autoseg')
        if isinstance(seg_path, str) and file_exist(seg_path):
            try:
                segmentation, _ = load_nifti(seg_path)
                posvec = get_lesion_position_vector_from_segmentation(atlas_crop, segmentation)
                df.at[idx, 'autoseg_posvec'] = posvec
                print(f"[{idx+1}] Processed: {seg_path}")
            except Exception as e:
                print(f"[{idx+1}] ERROR processing {seg_path}: {e}")
                df.at[idx, 'autoseg_posvec'] = 'ERROR'
        else:
            print(f"[{idx+1}] Segmentation file missing or invalid: {seg_path}")
            df.at[idx, 'autoseg_posvec'] = 'NOT_FOUND'

    # 保存为新文件
    if output_path is None:
        output_path = xlsx_path.replace('.xlsx', '_with_posvec.xlsx')
    df.to_excel(output_path, index=False)
    print(f"\n✔ Saved updated Excel to: {output_path}")

if __name__ == '__main__':
    input_excel_path = "/home/wengjy/tumor_analysis/digicare/experiments/003_tiantan_glioma_10k/xlsx/hxm/tiantan_buchong_v1.0.xlsx"  
    output_excel_path = None  
    process_excel_autoseg_vector(input_excel_path, output_excel_path)
