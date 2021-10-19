import numpy as np
from scipy.ndimage import convolve, gaussian_filter


def log_heat_kernel_GIF(image, mask, atlas_warped_image, atlas_warped_mask, deformation_field, spacing=[0.8]*3):
    def normalize_image(img, brain_mask):
        # Prepare brain mask
        fg_mask = (brain_mask > 0)
        fg_mask[np.isnan(img)] = False  # remove NaNs from the mask

        # Compute useful image intensity stats
        img_fg = img[fg_mask]
        p99 = np.percentile(img_fg, 99)
        img_fg[img_fg > p99] = p99
        mean = np.mean(img_fg)
        std = np.std(img_fg)

        # Normalize the image
        img[np.isnan(img)] = mean  # Set NaNs to mean values to allow > comparison
        img[img > p99] = p99
        img = (img - mean) / std
        img[np.logical_not(fg_mask)] = 0

        return img

    def apply_cubic_Bsplines_kernel(intensity_map):
        kernel1d = np.array([1./6, 2./3, 1./6])
        kernel3d = kernel1d[:,None,None] * kernel1d[None,:,None] * kernel1d[None,None,:]  # 3x3x3
        output = convolve(intensity_map, kernel3d, mode='nearest')
        return output

    def high_pass_filter(vector_map):
        vector_map = np.squeeze(vector_map)
        sigma = np.array([20. / spacing[i] for i in range(3)] + [0.])
        low_fq = gaussian_filter(vector_map, sigma=sigma, mode='nearest')
        output = vector_map - low_fq
        return output

    # Normalize the input image
    img_norm = normalize_image(image, mask)

    # Normalize the atlas image intensity (zero mean, unit variance for each volume)
    atlas_img_norm = normalize_image(atlas_warped_image, atlas_warped_mask)

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

    return log_heat_kernel
