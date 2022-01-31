import os
import numpy as np
import nibabel as nib
from src.multi_atlas.atlas_propagation import probabilistic_segmentation_prior
from src.multi_atlas.utils import compute_disp_from_cpp
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
        grid_spacing, be, le, lp, save_folder, only_affine,
        merging_method='GIF', reuse_existing_pred=False,
        force_recompute_heat_kernels=False):

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

        # List of files that should exist at the end of the segmentation
        expected_output = os.path.join(save_folder_atlas, 'warped_atlas_seg_onehot.nii.gz')
        predicted_segmentation = os.path.join(save_folder_atlas, 'predicted_seg.nii.gz')
        expected_warped_atlas_path = os.path.join(save_folder_atlas, 'warped_atlas_img.nii.gz')
        expected_disp_path = os.path.join(save_folder_atlas, 'disp.nii.gz')
        expected_heat_kernel = os.path.join(save_folder_atlas, 'log_heat_kernel.nii.gz')
        to_not_remove = [  # paths to filter during the cleaning at the end
            expected_output,
            predicted_segmentation,
            # expected_warped_atlas_path,
            # expected_disp_path,
            expected_heat_kernel
        ]

        # Try to see what results we can reuse to save time
        compute_registration = True
        compute_heat_map = True
        if reuse_existing_pred:
            compute_registration = False
            compute_heat_map = False
            for p in to_not_remove:
                if not os.path.exists(p):
                    compute_registration = True
                    compute_heat_map = True
        if force_recompute_heat_kernels:
            compute_heat_map = True
            if not os.path.exists(expected_warped_atlas_path) or not os.path.exists(expected_disp_path):
                compute_registration = True

        if not compute_registration:
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
            seg = np.argmax(proba_atlas_prior, axis=-1).astype(np.uint8)
            seg_nii = nib.Nifti1Image(seg, template_seg_nii.affine)
            nib.save(seg_nii, predicted_segmentation)

        # Add the warped atlas segmentation
        proba_seg_list.append(proba_atlas_prior)

        # Compute the heat kernel for the GIF-like fusion
        if not compute_heat_map:  # Load the existing heat map (skip computation)
            log_heat_kernel_nii = nib.load(expected_heat_kernel)
            log_heat_kernel = log_heat_kernel_nii.get_fdata().astype(np.float32)
        else:  # Compute the heat map
            warped_atlas_nii = nib.load(expected_warped_atlas_path)
            warped_atlas = warped_atlas_nii.get_fdata().astype(np.float32)
            warped_atlas_mask = (np.argmax(proba_atlas_prior, axis=-1) > 0).astype(np.uint8)
            if compute_registration:  # Recompute the displacement field if registration was run
                expected_cpp_path = os.path.join(save_folder_atlas, 'cpp.nii.gz')
                expected_img_path = os.path.join(save_folder_atlas, 'img.nii.gz')
                compute_disp_from_cpp(expected_cpp_path, expected_img_path, expected_disp_path)
            deformation = nib.load(expected_disp_path).get_fdata().astype(np.float32)
            log_heat_kernel, lssd, disp_norm = log_heat_kernel_GIF(
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

        # Cleaning - remove the files that we will not need anymore
        for f_n in os.listdir(save_folder_atlas):
            p = os.path.join(save_folder_atlas, f_n)
            if not p in to_not_remove:
                os.system('rm %s' % p)

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
