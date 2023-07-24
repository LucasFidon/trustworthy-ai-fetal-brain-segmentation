"""
Run nnUNet inference for the pre-trained nnUNet models for 3D fetal brain segmentation.
"""

import os
import numpy as np
import nibabel as nib
from argparse import ArgumentParser
from loguru import logger
from scipy.ndimage import binary_dilation
import pickle


parser = ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--mask', type=str)
parser.add_argument('--output_folder', type=str)
parser.add_argument('--task', default='Task225_FetalBrain3dTrust', type=str)  # Fetal 3D
parser.add_argument('--fold', default='all', type=str, help='all, 0, 1, 2, 3, or 4.')
parser.add_argument('--model', default='3d_fullres', type=str)
parser.add_argument('--trainer', default='nnUNetTrainerV2', type=str)
parser.add_argument('--plan', default='nnUNetPlansv2.1', type=str)
parser.add_argument('--save_npz', action='store_true')
parser.add_argument('--seg_prior', type=str)


def load_softmax(softmax_path, pkl_path):
    """
    Load a softmax volume file saved as .npz
    :param softmax_path: str; path to the softmax .npz file
    :param pkl_path: str; path to the pickle config file
    :return: numpy array; softmax prediction in the space of the original image
    """
    softmax_cropped = np.load(softmax_path)['softmax'][None][0,...]
    # This is a cropped volume
    # we need to put it back into the space of the full image
    with open(pkl_path, 'rb') as f:
        prop = pickle.load(f)
    ori_img_shape = prop['original_size_of_raw_data']
    shape = (softmax_cropped.shape[0], ori_img_shape[0], ori_img_shape[1], ori_img_shape[2])
    softmax_full = np.zeros(shape)
    softmax_full[0, ...] = 1  # initialize to background
    crop_coord = np.array(prop['crop_bbox'])
    softmax_full[:, crop_coord[0,0]:crop_coord[0,1], crop_coord[1,0]:crop_coord[1,1], crop_coord[2,0]:crop_coord[2,1]] = softmax_cropped
    softmax_full = np.transpose(softmax_full, (0, 3, 2, 1))
    return softmax_full


def main(args):
    # Load the data and mask the background
    img_nii = nib.load(args.input)
    img_nii.set_data_dtype(np.float32)  # Force data type to be float32
    img_np = img_nii.get_fdata().astype(np.float32)
    mask_np = nib.load(args.mask).get_fdata().astype(np.uint8)
    # Skull stripping
    mask_np = binary_dilation(mask_np, iterations=3)
    img_np[mask_np == 0] = 0.
    # Set Nan values to mean intensity value
    num_nans = np.count_nonzero(np.isnan(img_np))
    if num_nans > 0:
        logger.warning('%d NaN values were found in the image %s'
                % (num_nans, args.input))
        logger.warning('Replaced NaN values with the mean value of the image.')
        img_np[np.isnan(img_np)] = np.nanmean(img_np)

    # Save the processed image to the output folder
    tmp_folder = './tmp_%s' % args.task
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    img_name = os.path.split(os.path.dirname(args.input))[1]
    save_img_path = os.path.join(tmp_folder, '%s_0000.nii.gz' % img_name)
    img_save_nii = nib.Nifti1Image(img_np, img_nii.affine, img_nii.header)
    nib.save(img_save_nii, save_img_path)
    # Save the segmentation prior as additional input
    if args.seg_prior is not None:
        seg_prior = nib.load(args.seg_prior).get_fdata().astype(np.float32)
        for c in range(1, 9):
            seg_prior_c = seg_prior[:,:,:,c]
            seg_prior_c_nii = nib.Nifti1Image(seg_prior_c, img_nii.affine, img_nii.header)
            nib.save(
                seg_prior_c_nii,
                os.path.join(tmp_folder, img_name + "_" + str(c).zfill(4) + ".nii.gz")
            )
        os.system('rm %s' % args.seg_prior)


    # Run the autoseg with nnUNet and save it in the specified output folder
    if args.fold == 'all':
        folders = []
        # Compute the softmax predictions for each fold in separate tmp folders
        for fold in range(5):
            output_folder_fold = os.path.join(tmp_folder, 'fold%d' % fold)
            # folders.append(output_folder_fold)
            options = '-t %s -f %d -m %s -tr %s -p %s --save_npz' % \
                (args.task, fold, args.model, args.trainer, args.plan)
            cmd = 'nnUNet_predict -i %s -o %s %s' % (tmp_folder, output_folder_fold, options)
            os.system(cmd)
            if os.path.exists(os.path.join(output_folder_fold, '%s.npz' % img_name)):
                # Only if the fold model exists, the fold output is included
                folders.append(output_folder_fold)
        # Ensemble the predictions
        cmd = 'nnUNet_ensemble -f '
        for folder in folders:
            cmd += '%s ' % folder
        cmd += '-o %s ' % args.output_folder
        if args.save_npz:
            cmd += '--npz'
        os.system(cmd)
    else:
        options = '-t %s -f %s -m %s -tr %s -p %s' % \
              (args.task, args.fold, args.model, args.trainer, args.plan)
        if args.save_npz:  # Save the softmax prediction
            options += ' --save_npz'
        cmd = 'nnUNet_predict -i %s -o %s %s' % (tmp_folder, args.output_folder, options)
        os.system(cmd)

    if os.path.exists(tmp_folder):
        os.system('rm -r %s' % tmp_folder)


if __name__ == '__main__':
    args= parser.parse_args()
    main(args)