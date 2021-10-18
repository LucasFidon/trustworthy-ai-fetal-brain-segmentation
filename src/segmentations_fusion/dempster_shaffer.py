import numpy as np
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from sklearn.mixture import GaussianMixture
from src.utils.definitions import LABELS


ATLAS_MARGIN = [2] * 9  # bg, wm, vent, cer, ext-csf, cgm, dgm, bs, cc (in voxels)


def merge_deep_and_atlas_seg(deep_proba, atlas_seg):
    out_score = np.copy(deep_proba)
    print('\nApply atlas-based margins to the deep learning-based segmentation. ', ATLAS_MARGIN)
    # We set the proba to zeros outside of "atlas mask + margin"
    for c in range(len(ATLAS_MARGIN)):
        atlas_seg_c = (atlas_seg == c)
        atlas_seg_c = binary_dilation(atlas_seg_c, iterations=ATLAS_MARGIN[c])
        out_score[c, np.logical_not(atlas_seg_c)] = 0

    # Normalize the probability
    out_score[:, ...] /= np.sum(out_score, axis=0)

    return out_score


def dempster_add_intensity_prior(deep_proba, image, mask):
    mask[np.isnan(image)] = 0  # mask nan values
    # Erode the mask for the intensity prior
    # because we want to make sure we do not include the background.
    # The risk is that some of the foreground voxels are missing but this is ok.
    mask_prior = binary_erosion(mask, iterations=3)

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

    img_fg = image[mask_prior == 1]

    # Compute the prior probability
    m_csf = np.exp(-0.5 * np.square((img_fg - mean_csf) / std_csf)) / std_csf
    m_mix = np.exp(-0.5 * np.square((img_fg - mean_mix) / std_mix)) / std_mix

    labels_seen = []
    for roi_eval in list(LABELS.keys()):
        if roi_eval in ['intra_axial_csf', 'extra_axial_csf']:
            for i in LABELS[roi_eval]:
                if not i in labels_seen:
                    deep_proba[i, mask_prior == 1] *= (m_csf + m_mix)
                    labels_seen.append(i)
        else: # Note that we want to include the background label here
            for i in LABELS[roi_eval]:
                if not i in labels_seen:
                    labels_seen.append(i)
                    print('class %d' % i)
                    deep_proba[i, mask_prior == 1] *= m_mix

    # Normalize the probability
    deep_proba[:, ...] /= np.sum(deep_proba, axis=0)

    return deep_proba
