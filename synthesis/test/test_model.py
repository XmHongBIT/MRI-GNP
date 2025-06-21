import tqdm
from monai.inferers import SlidingWindowInferer
import os
import SimpleITK as sitk
from torch.utils.data import DataLoader
import torchio as tio
import torch
import monai
import numpy as np
import pandas as pd


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_case():

    xlsx_path = r''

    data = pd.read_excel(xlsx_path, sheet_name='Sheet1', dtype=str)
    filtered_df_val = data[(data['use'] == "test") & (data['T1'].notna()) & (data['T1ce'].notna()) & (data['T2'].notna()) & (data['T2FLAIR'].notna())]

    subjects_val = []
    for index, row in filtered_df_val.iterrows():
        subject = tio.Subject(t1n=tio.ScalarImage(row['T1']), t2w=tio.ScalarImage(row['T2']),
                              t2f=tio.ScalarImage(row['T2FLAIR']), t1c=tio.ScalarImage(row['T1ce']))
        subjects_val.append(subject)

    transform = tio.Compose([tio.transforms.ZNormalization()])
    val_subjects = tio.SubjectsDataset(subjects_val, transform)
    val_dataloader = DataLoader(val_subjects, batch_size=1, pin_memory=True)

    experiment_path = r''
    model_name = ''
    model = monai.networks.nets.SwinUNETR(img_size=(96, 96, 96), in_channels=3, out_channels=1, depths=(2, 4, 2, 2)).cuda()
    model.load_state_dict(torch.load(os.path.join(experiment_path, model_name + '.pth')))

    inferer = SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=1, overlap=0.5, mode='gaussian')
    with torch.no_grad():
        model.eval()
        for images in tqdm.tqdm(val_dataloader):
            m0, m1, m2 = images['t1n']['data'].to(DEVICE), images['t2w']['data'].to(DEVICE), images['t2f']['data'].to(DEVICE)
            image_input = torch.cat([m0, m1, m2], dim=1)
            pred = inferer(inputs=image_input, network=model)
            pred_np = pred.detach().cpu().squeeze().permute(2, 1, 0).numpy()
            output_img = sitk.GetImageFromArray(pred_np)
		
            out_path = os.path.join(images['t1n']['path'][0].replace('t1.nii.gz', 't1ce_syn.nii.gz'))
            output_img.CopyInformation(sitk.ReadImage(images['t1n']['path'][0]))
            print(out_path)
            sitk.WriteImage(output_img, out_path)


if __name__ == '__main__':
    test_case()
