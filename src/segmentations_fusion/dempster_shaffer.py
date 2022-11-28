import os
from time import time
import numpy as np
from math import ceil
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from sklearn.mixture import GaussianMixture
import sys
sys.path.append('/workspace/trustworthy-ai-fetal-brain-segmentation')
from src.utils.definitions import *


def merge_deep_and_atlas_seg(deep_proba, atlas_seg, condition):
    assert condition in CONDITIONS, \
        'Only conditions %s are supported. Received %s.' % (str(CONDITIONS), condition)

    # Anatomical prior
    out = np.copy(deep_proba)
    if condition == 'Neurotypical':
        atlas_margin = ATLAS_MARGINS_CONTROL
    elif condition == 'Spina Bifida':
        atlas_margin = ATLAS_MARGINS_SPINA_BIFIDA
    else:  # other pathology
        atlas_margin = np.maximum(ATLAS_MARGINS_CONTROL, ATLAS_MARGINS_SPINA_BIFIDA)
    # Round the margins to the closest integer values
    atlas_margin = np.rint(atlas_margin).astype(np.int)
    print('\nApply atlas-based margins to the deep learning-based segmentation. ', atlas_margin)

    # We set the proba to zeros outside of "atlas mask + margin"
    for c in range(len(atlas_margin)):
        atlas_seg_c = (atlas_seg == c)
        atlas_seg_c = binary_dilation(atlas_seg_c, iterations=atlas_margin[c])
        out[c, np.logical_not(atlas_seg_c)] = 0

    # Normalize the probability
    out[:, ...] /= np.sum(out, axis=0)

    return out


def bilateral_filtering(image, mask, sigma_color=1, sigma_spatial=1):
    # Warning: Time complexity in O(sigma_spatial**3)
    MIN_WIN_SIZE = 2

    # Normalize the image before filtering
    img_for = image[mask == 1]
    image -= np.mean(img_for)
    image /= np.std(img_for)
    image[mask == 0] = 0

    # Apply the filter to the image
    denoised_image = np.zeros_like(image)
    normalization = np.zeros_like(image)
    win_size = max(MIN_WIN_SIZE, ceil(3 * sigma_spatial))

    for kx in range(-win_size, win_size+1):
        for ky in range(-win_size, win_size+1):
            for kz in range(-win_size, win_size+1):
                trans_img = np.roll(image, shift=(kx,ky,kz))
                trans_mask = np.roll(mask, shift=(kx,ky,kz))
                gauss_w = np.exp(-0.5 * (kx**2 + ky**2 + kz**2) / sigma_spatial**2)
                # Multiply the weights by the translated mask
                # because we want to ignore the voxels outside the mask
                w = gauss_w * trans_mask * np.exp(-0.5 * ((image - trans_img) / sigma_color)**2)
                denoised_image += w * trans_img
                normalization += w
    denoised_image[mask == 1] /= normalization[mask == 1]
    denoised_image[mask == 0] = 0

    return denoised_image


def dempster_add_intensity_prior(deep_proba, image, mask, denoise=False):
    out = np.copy(deep_proba)

    mask[np.isnan(image)] = 0  # mask nan values
    # Erode the mask for the intensity prior
    # because we want to make sure we do not include the background.
    # The risk is that some of the foreground voxels are missing but this is ok.
    mask_prior = binary_erosion(mask, iterations=3)

    if denoise:
        print('\n*** Apply bilateral filtering before adding the intensity prior.')
        t0 = time()
        image = bilateral_filtering(
            image=image,
            mask=mask_prior,
            sigma_color=1,
            sigma_spatial=1,
        )
        t1 = time()
        print('Bilateral filtering done is %.0f seconds.' % (t1 - t0))

    print('\nFit a GMM with two components for the intensity prior.')
    # Fit the GMM with two components
    X = image[mask_prior == 1]
    X = X[:, None]
    gm = GaussianMixture(n_components=2, random_state=0).fit(X)
    means = gm.means_.flatten()
    std = np.sqrt(gm.covariances_.flatten())

    # Identify the components
    argsort = np.argsort(means)
    mean_csf = means[argsort[1]]
    std_csf = std[argsort[1]]
    mean_mix = means[argsort[0]]
    std_mix = std[argsort[0]]

    img_fg = image[mask == 1]

    # Compute the prior probability
    m_csf = np.exp(-0.5 * np.square((img_fg - mean_csf) / std_csf)) / std_csf
    m_mix = np.exp(-0.5 * np.square((img_fg - mean_mix) / std_mix)) / std_mix

    labels_seen = []
    for roi_eval in list(LABELS.keys()):
        if roi_eval in ['intra_axial_csf', 'extra_axial_csf', 'background']:
            for i in LABELS[roi_eval]:
                if not i in labels_seen:
                    out[i, mask == 1] *= (m_csf + m_mix)
                    labels_seen.append(i)
        else:
            for i in LABELS[roi_eval]:
                if not i in labels_seen:
                    labels_seen.append(i)
                    out[i, mask == 1] *= m_mix

    # Normalize the probability
    out[:, ...] /= np.sum(out, axis=0)

    return out


if __name__ == '__main__':
    # Playing with bilateral filtering
    import nibabel as nib
    example_path = os.path.join(CORRECTED_ZURICH_DATA_DIR, 'sub-feta016')
    srr_path = os.path.join(example_path, 'srr.nii.gz')
    srr_nii = nib.load(srr_path)
    srr = srr_nii.get_fdata().astype(np.float32)
    mask_path = os.path.join(example_path, 'mask.nii.gz')
    mask_nii = nib.load(mask_path)
    mask = mask_nii.get_fdata().astype(np.uint8)
    t0 = time()
    srr_denoised = bilateral_filtering(srr, mask, sigma_color=1, sigma_spatial=1)
    t1 = time()
    print('Bilateral filtering done is %.0f seconds.' % (t1 - t0))
    srr_den_nii = nib.Nifti1Image(srr_denoised, srr_nii.affine)
    nib.save(srr_den_nii, 'test_denoise.nii.gz')
