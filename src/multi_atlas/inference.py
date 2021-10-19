import os
import numpy as np
import nibabel as nib
from src.multi_atlas.atlas_propagation import probabilistic_segmentation_prior
from src.multi_atlas.utils import compute_def_from_cpp
from src.multi_atlas.multi_atlas_fusion_weights import log_heat_kernel_GIF

SUPPORTED_MERGING_METHOD = [
    'mean',
    'GIF',
]


def _weights_from_log_heat_kernels(log_heat_kernels):
    max_heat = log_heat_kernels.max(axis=0)
    x = log_heat_kernels - max_heat[None,:,:,:]
    exp_x = np.exp(x)
    norm = np.sum(exp_x, axis=0)
    w = exp_x / norm[None,:,:,:]
    return w

def multi_atlas_segmentation(img_nii, mask_nii, atlas_folder_list,
        grid_spacing, be, le, lp, save_folder, only_affine, merging_method='GIF'):

    assert merging_method in SUPPORTED_MERGING_METHOD, \
        "Merging method %s not supported. Only %s supported." % (merging_method, str(SUPPORTED_MERGING_METHOD))

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    proba_seg_list = []  # list of atlas segmentations after registration
    log_heat_kernel_list = []

    # Register the atlas segmentations to the input image
    for folder in atlas_folder_list:
        atlas_name = os.path.split(folder)[1]
        save_folder_atlas = os.path.join(save_folder, atlas_name)
        expected_output = os.path.join(save_folder_atlas, 'warped_atlas_seg_onehot.nii.gz')
        if os.path.exists(expected_output):
            print('\n%s already exists.\nSkip registration.' % expected_output)
            proba_atlas_prior_nii = nib.load(expected_output)
            proba_atlas_prior = proba_atlas_prior_nii.get_fdata().astype(np.float32)
        else:
            template_nii = nib.load(os.path.join(folder, 'srr.nii.gz'))
            template_mask_nii = nib.load(os.path.join(folder, 'mask.nii.gz'))
            template_seg_nii = nib.load(os.path.join(folder, 'parcellation.nii.gz'))
            proba_atlas_prior = probabilistic_segmentation_prior(
                image_nii=img_nii,
                mask_nii=mask_nii,
                template_nii=template_nii,
                template_seg_nii=template_seg_nii,
                template_mask_nii=template_mask_nii,
                grid_spacing=grid_spacing,
                be=be,
                le=le,
                lp=lp,
                save_folder_path=save_folder_atlas,
                affine_only=only_affine,
            )

        # Add the warped atlas segmentation
        proba_seg_list.append(proba_atlas_prior)

        # Compute the heat kernel for the GIF-like fusion
        expected_heat_kernel = os.path.join(
            save_folder_atlas, 'heat_kernel.nii.gz')
        if os.path.exists(expected_heat_kernel):
            # Load the existing heat map (skip computation)
            log_heat_kernel_nii = nib.load(expected_heat_kernel)
            log_heat_kernel = log_heat_kernel_nii.get_fdata().astype(np.float32)
        else:
            # Compute the heat map
            expected_warped_atlas_path = os.path.join(
                save_folder_atlas, 'warped_atlas_img_seg.nii.gz')
            warped_atlas_nii = nib.load(expected_warped_atlas_path)
            warped_atlas = warped_atlas_nii.get_fdata().astype(np.float32)
            warped_atlas_mask = (np.argmax(proba_atlas_prior, axis=-1) > 0).astype(np.uint8)
            expected_cpp_path = os.path.join(save_folder_atlas, 'cpp.nii.gz')
            expected_img_path = os.path.join(save_folder_atlas, 'img.nii.gz')
            def_path = os.path.join(save_folder_atlas, 'def.nii.gz')
            compute_def_from_cpp(expected_cpp_path, expected_img_path, def_path)
            deformation = nib.load(def_path).get_fdata().astype(np.float32)
            log_heat_kernel = log_heat_kernel_GIF(
                image=img_nii.get_fdata().astype(np.float32),
                mask=mask_nii.get_fdata().astype(np.uint8),
                atlas_warped_image=warped_atlas,
                atlas_warped_mask=warped_atlas_mask,
                deformation_field=deformation,
            )
            # Save the heat kernel
            log_heat_kernel_nii = nib.Nifti1Image(log_heat_kernel, warped_atlas_nii.affine)
            nib.save(log_heat_kernel_nii, expected_heat_kernel)
        log_heat_kernel_list.append(log_heat_kernel)

    # Merge the proba predictions
    if merging_method == 'GIF':
        proba_seg = np.stack(proba_seg_list, axis=0)  # n_atlas, n_x, n_y, n_z, n_class
        log_heat_kernels = np.stack(log_heat_kernel_list, axis=0)  # n_atlas, n_x, n_y, n_z
        weights = _weights_from_log_heat_kernels(log_heat_kernels)
        weighted_proba_seg = weights[:,:,:,:, None] * proba_seg
        multi_atlas_proba_seg = np.sum(weighted_proba_seg, axis=0)
    else:  # Vanilla average
        multi_atlas_proba_seg = np.mean(np.stack(proba_seg_list, axis=0), axis=0)

    return multi_atlas_proba_seg
