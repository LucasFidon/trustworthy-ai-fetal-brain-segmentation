import os
import numpy as np
import nibabel as nib
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--pred_folder', type=str)

# THRESHOLD_ET = 50  # value for the BraTS 2020 challenge
THRESHOLD_ET = 0  # deactivate the port-processing


def convert_seg(nii_file_path, save_folder):
    # 0,1,2,3 -> 0,1,2,4
    nii = nib.load(nii_file_path)
    seg = nii.get_data()
    converted_seg = np.copy(seg)
    converted_seg[seg == 2] = 1  # non-enhancing tunor
    converted_seg[seg == 1] = 2  # edema
    converted_seg[seg == 3] = 4  # enhancing tumor
    num_ET_voxels = np.sum(converted_seg == 4)
    if num_ET_voxels < THRESHOLD_ET and num_ET_voxels > 0:
        print('')
        print('Only %d voxels were predicted as ET for' % num_ET_voxels)
        print(nii_file_path)
        print('Changed all ET predictions to NET')
        converted_seg[converted_seg == 4] = 1
    new_nii = nib.Nifti1Image(converted_seg, nii.affine, nii.header)
    # Save the converted seg
    seg_name = os.path.split(nii_file_path)[1]
    seg_save_path = os.path.join(save_folder, seg_name)
    nib.save(new_nii, seg_save_path)


def main(args):
    base_folder = os.path.split(args.pred_folder)[0]
    seg_folder_name = os.path.split(args.pred_folder)[1]
    save_folder = os.path.join(base_folder, '%s_converted' % seg_folder_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    for seg_nii_name in os.listdir(args.pred_folder):
        if not seg_nii_name.endswith('.nii.gz'):
            continue
        seg_file_path = os.path.join(args.pred_folder, seg_nii_name)
        convert_seg(seg_file_path, save_folder)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
