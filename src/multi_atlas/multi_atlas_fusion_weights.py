import numpy as np
from scipy.ndimage import convolve, gaussian_filter
from scipy.ndimage.morphology import binary_dilation, binary_erosion

SIGMA_HIGH_PASS_FILTER = 20  # in mm. In the GIF paper they use 20mm.
# NORMALIZATION = 'percentiles'  # can also be 'z_score'
NORMALIZATION = 'z_score'  # z_score with mask erosion works best


def log_heat_kernel_GIF(image, mask, atlas_warped_image, atlas_warped_mask, deformation_field, spacing=[0.8]*3):
    def normalize_image(img, brain_mask, mode='z_score'):
        # Prepare brain mask
        fg_mask = (brain_mask > 0)
        fg_mask[np.isnan(img)] = False  # remove NaNs from the mask
        # Erode the mask because we want to make sure we do not include
        # the background voxels to compute the intensity statistic
        fg_mask_eroded = binary_erosion(fg_mask, iterations=3)

        # Compute useful image intensity stats
        img_fg = img[fg_mask_eroded]
        p999 = np.percentile(img_fg, 99.9)
        img_fg[img_fg > p999] = p999
        mean = np.mean(img_fg)

        # Normalize the image
        img[np.isnan(img)] = mean  # Set NaNs to mean values to allow > comparison
        img[img > p999] = p999
        if mode == 'z_score':
            std = np.std(img_fg)
            img = (img - mean) / std
        else:
            median = np.median(img_fg)
            p95 = np.percentile(img_fg, 95)
            p5 = np.percentile(img_fg, 5)
            print('Use percentiles normalization')
            img = (img - median) / (p95 - p5)

        # Set background voxels to 0
        img[np.logical_not(fg_mask)] = 0

        return img

    def apply_cubic_Bsplines_kernel(intensity_map):
        kernel1d = np.array([1./6, 2./3, 1./6])
        kernel3d = kernel1d[:,None,None] * kernel1d[None,:,None] * kernel1d[None,None,:]  # 3x3x3
        output = convolve(intensity_map, kernel3d, mode='nearest')
        return output

    def high_pass_filter(vector_map):
        vector_map = np.squeeze(vector_map)
        sigma = np.array([SIGMA_HIGH_PASS_FILTER / spacing[i] for i in range(3)] + [0.])
        low_fq = gaussian_filter(vector_map, sigma=sigma, mode='nearest')
        output = vector_map - low_fq
        return output

    # Normalize the input image
    img_norm = normalize_image(
        image, mask, mode=NORMALIZATION)

    # Normalize the atlas image intensity (zero mean, unit variance for each volume)
    atlas_img_norm = normalize_image(
        atlas_warped_image, atlas_warped_mask,mode=NORMALIZATION)

    # Compute the intensity term (LSSD)
    ssd = (atlas_img_norm - img_norm) ** 2
    lssd = apply_cubic_Bsplines_kernel(ssd)

    # Remove the low frequencies of the deformations
    disp = high_pass_filter(deformation_field)

    # Compute the displacement field norm (in mm)
    disp_norm = np.linalg.norm(disp, axis=-1)

    # Compute the heat kernel maps
    distance_map = 0.5 * lssd + 0.5 * disp_norm
    log_heat_kernel = -distance_map**2

    return log_heat_kernel, lssd, disp_norm
