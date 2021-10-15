import os
import numpy as np
import nibabel as nib
from src.multi_atlas.atlas_propagation import probabilistic_segmentation_prior


def multi_atlas_segmentation(img_nii, mask_nii, atlas_folder_list,
        grid_spacing, be, le, lp, save_folder, only_affine):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Register the atlas segmentations to the input image
    proba_seg_list = []
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
        proba_seg_list.append(proba_atlas_prior)

    # Merge the proba predictions
    multi_atlas_proba_seg = np.mean(np.stack(proba_seg_list, axis=0), axis=0)

    #TODO: other merging methods

    return multi_atlas_proba_seg
