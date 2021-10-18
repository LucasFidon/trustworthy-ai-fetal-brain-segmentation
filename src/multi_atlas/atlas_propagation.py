import os
import nibabel as nib
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import binary_dilation
import numpy as np
from time import time
from src.utils.definitions import NIFTYREG_PATH


SIGMA = 0  # sigma for smoothing the segmentation prior


def probabilistic_segmentation_prior(image_nii, mask_nii,
                                     template_nii, template_seg_nii, template_mask_nii,
                                     mask_dilation=3, save_folder_path=None, use_affine=True,
                                     affine_only=False, grid_spacing=4, be=0.001, le=0.01, lp=3):
    """
    Summary of what this function does:
    1. register the template to the image (use concatenation with one-hot segmentation to help)
    2. propagate the labels and (optionally) smooth the prior

    The segmentation is used during registration with MSE if seg_nii is not None.
    """
    time_0 = time()
    # print('Use the atlas volumes in %s' % atlas_folder)
    print('Use NiftyReg version in %s' % NIFTYREG_PATH)

    # Step 0: Create the folder where to save the registration output
    tmp_folder = './tmp'
    if save_folder_path is None:  # in this case the tmp folder will be deleted
        save_folder_path = tmp_folder

    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    # Mask the image
    masked_image_nii = _mask_image(
        image_nii=image_nii,
        mask_nii=mask_nii,
        num_dilation=mask_dilation
    )

    # Register the template to the image
    affine_params, cpp_params_path = _register_atlas_to_img(
        image_nii=masked_image_nii,
        mask_nii=mask_nii,
        atlas_nii=template_nii,
        atlas_mask_nii=template_mask_nii,
        grid_spacing=grid_spacing,
        be=be,
        le=le,
        lp=lp,
        save_folder=save_folder_path,
        use_affine=use_affine,
        affine_only=affine_only,
    )

    # Propagate the labels
    proba_seg_prior = _propagate_labels(
        atlas_seg_nii=template_seg_nii,
        image_nii=masked_image_nii,
        aff_path=affine_params,  # will be None if use_affine == False
        cpp_path=cpp_params_path,
        save_folder=save_folder_path,
    )

    # Delete the tmp folder
    if os.path.exists(tmp_folder):
        os.system('rm -r %s' % tmp_folder)
    duration = int(time() - time_0)  # in seconds
    minutes = duration // 60
    seconds = duration - minutes * 60
    print('The atlas propagation has been performed in %dmin%dsec' % (minutes, seconds))

    return proba_seg_prior


def _mask_image(image_nii, mask_nii, num_dilation):
    image_np = image_nii.get_fdata().astype(np.float32)
    mask_np = mask_nii.get_fdata().astype(np.uint8)
    if num_dilation > 0:
        dilated_mask_np = binary_dilation(mask_np, iterations=num_dilation).astype(np.uint8)
    else:
        dilated_mask_np = mask_np
    image_np[dilated_mask_np == 0] = 0
    out_img_nii = nib.Nifti1Image(image_np, image_nii.affine, image_nii.header)
    return out_img_nii


def _convert_to_one_hot_and_smooth_seg_prior(segmentation_nii, smooth_sigma=SIGMA):
    seg_np = segmentation_nii.get_fdata().astype(np.uint8)
    # Convert the segmentation into one-hot representation
    hard_prior_seg_one_hot = np.eye(seg_np.max() + 1)[seg_np].astype(np.float32)  # numpy magic
    #TODO: what about filtering with a cubic B-spline filter instead?
    if smooth_sigma > 0.:  # Gaussian smoothing
        # Put the class dimension first (PyTorch convention)
        hard_prior_seg_one_hot = np.transpose(hard_prior_seg_one_hot, (3, 0, 1, 2))
        prior = np.stack(
            [gaussian_filter(hard_prior_seg_one_hot[c,...], sigma=smooth_sigma, order=0, mode='nearest')
            for c in range(hard_prior_seg_one_hot.shape[0])],
            axis=0,
        )
        # Normalize the smooth prior so that the entries sum to 1 for each voxel
        prior /= np.sum(prior, axis=0)
        prior = np.transpose(prior, (1, 2, 3, 0))
    else:
        prior = hard_prior_seg_one_hot
    prior_nii = nib.Nifti1Image(prior, segmentation_nii.affine)
    return prior_nii


def _register_atlas_to_img(image_nii, mask_nii,
                           atlas_nii, atlas_mask_nii,
                           grid_spacing, be, le, lp,
                           save_folder, use_affine, affine_only):
    """
    Affine registration + non-linear registration with stationary velocity fields
    are performed to register the atlas to the image.
    The segmentation is used with MSE if segmentation_nii is not None.
    """
    # Return the path to the output velocity field returned by NiftyReg
    def save_nifti(volume_np, affine, save_path):
        volume_nii = nib.Nifti1Image(volume_np, affine)
        nib.save(volume_nii, save_path)

    # Prepare the volumes to register
    img_seg_path = os.path.join(save_folder, 'img_seg.nii.gz')
    mask_path = os.path.join(save_folder, 'mask.nii.gz')
    atlas_img_seg_path = os.path.join(save_folder, 'atlas_img_seg.nii.gz')
    atlas_mask_path = os.path.join(save_folder, 'atlas_mask.nii.gz')
    save_nifti(image_nii.get_fdata(), image_nii.affine, img_seg_path)
    save_nifti(atlas_nii.get_fdata(), atlas_nii.affine, atlas_img_seg_path)
    save_nifti(
        mask_nii.get_fdata().astype(np.uint8),
        mask_nii.affine,
        mask_path
    )
    save_nifti(
        atlas_mask_nii.get_fdata().astype(np.uint8),
        atlas_mask_nii.affine,
        atlas_mask_path
    )

    # Affine registration
    if use_affine or affine_only:
        affine_path = os.path.join(save_folder, 'outputAffine.txt')
        affine_res_path = os.path.join(save_folder, 'affine_warped_atlas.nii.gz')
        affine_reg_cmd = '%s/reg_aladin -ref %s -rmask %s -flo %s -fmask %s -res %s -aff %s -comm -voff' % \
            (NIFTYREG_PATH, img_seg_path, mask_path, atlas_img_seg_path, atlas_mask_path, affine_res_path, affine_path)
        os.system(affine_reg_cmd)

        # Warp the atlas mask
        affine_res_mask_path = os.path.join(save_folder, 'affine_warped_atlas_mask.nii.gz')
        affine_warp_mask_cmd = '%s/reg_resample -ref %s -flo %s -trans %s -res %s -inter 0 -voff' % \
            (NIFTYREG_PATH, img_seg_path, atlas_mask_path, affine_path, affine_res_mask_path)
        os.system(affine_warp_mask_cmd)

        if affine_only:
            return affine_path, None

    else:  # no affine transformation
        affine_path = None
        affine_res_path = atlas_img_seg_path
        affine_res_mask_path = atlas_mask_path

    # Registration
    reg_loss_options = '-lncc 0 6'
    res_path = os.path.join(save_folder, 'warped_atlas_img_seg.nii.gz')
    cpp_path = os.path.join(save_folder, 'cpp.nii.gz')
    reg_options = '-be %f -le %f -sx %s -ln 3 -lp %d %s -voff' % \
        (be, le, grid_spacing, lp, reg_loss_options)
    reg_cmd = '%s/reg_f3d -ref %s -rmask %s -flo %s -fmask %s -res %s -cpp %s %s' % \
        (NIFTYREG_PATH, img_seg_path, mask_path, affine_res_path, affine_res_mask_path, res_path, cpp_path, reg_options)
    # print('Non linear registration command line:')
    # print(reg_cmd)
    os.system(reg_cmd)
    return affine_path, cpp_path


def _propagate_labels(atlas_seg_nii, image_nii, aff_path, cpp_path, save_folder):
    # Infere the tmp folder from input
    if cpp_path is not None:
        tmp_folder = os.path.split(cpp_path)[0]
    else:
        tmp_folder = os.path.split(aff_path)[0]
    image_path = os.path.join(tmp_folder, 'img.nii.gz')
    nib.save(image_nii, image_path)

    # Convert the atlas segmentation into one-hot representation
    atlas_seg_onehot_nii = _convert_to_one_hot_and_smooth_seg_prior(atlas_seg_nii)
    atlas_seg_path = os.path.join(tmp_folder, 'atlas_seg_onehot.nii.gz')
    nib.save(atlas_seg_onehot_nii, atlas_seg_path)

    # Affine deformation of the atlas segmentation
    if aff_path is not None:
        aff_warped_seg = os.path.join(save_folder, 'warped_atlas_seg_onehot_after_aff_only.nii.gz')
        cmd = '%s/reg_resample -ref %s -flo %s -trans %s -res %s -inter 1 -voff' % \
            (NIFTYREG_PATH, image_path, atlas_seg_path, aff_path, aff_warped_seg)
        os.system(cmd)
    else:
        aff_warped_seg = atlas_seg_path

    if cpp_path is not None:
        # Warp the atlas seg given a pre-computed transformation (vel) and save it
        warped_seg = os.path.join(save_folder, 'warped_atlas_seg_onehot.nii.gz')
        cmd = '%s/reg_resample -ref %s -flo %s -trans %s -res %s -inter 1 -voff' % \
            (NIFTYREG_PATH, image_path, aff_warped_seg, cpp_path, warped_seg)
        os.system(cmd)
    else:
        warped_seg = aff_warped_seg

    # Load and return the warped atlas proba numpy array
    warped_atlas_proba_nii = nib.load(warped_seg)
    warped_atlas_proba = warped_atlas_proba_nii.get_fdata().astype(np.float32)

    # Deal with the padding to 0
    # Change the padding from all 0 to one hot for the background
    sum_proba_map = np.sum(warped_atlas_proba, axis=-1)
    warped_atlas_proba[sum_proba_map == 0, 0] = 1.
    sum_proba_map = np.sum(warped_atlas_proba, axis=-1)
    warped_atlas_proba /= sum_proba_map[:, :, :, None]

    # Replace the warped atlas seg
    warped_atlas_proba_nii_post = nib.Nifti1Image(warped_atlas_proba, warped_atlas_proba_nii.affine)
    nib.save(warped_atlas_proba_nii_post, warped_seg)

    return warped_atlas_proba