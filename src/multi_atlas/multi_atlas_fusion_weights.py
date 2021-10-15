import numpy as np
from scipy.ndimage import convolve, gaussian_filter


def merge_multi_atlas_seg_GIF(image, multi_atlas_proba_list, multi_atlas_warped_image_list,
                              deformation_field_list, spacing=[0.8]*3):
    def normalize_image(img):
        img[np.isnan(img)] = 0
        p99 = np.percentile(img, 99)
        img[img > p99] = p99
        mean = np.mean(img)
        std = np.std(img)
        img = (img - mean) / std
        return img

    def apply_cubic_Bsplines_kernel(vector_map):
        kernel1d = np.array([1./6, 2./3, 1./6])
        kernel3d = kernel1d[:,None,None] * kernel1d[None,:,None] * kernel1d[None,None,:]  # 3x3x3
        output = convolve(vector_map, kernel3d[:,:,:,None])
        return output

    def high_pass_filter(vector_map):
        sigma = 20. / np.array(spacing)
        low_fq = gaussian_filter(vector_map, sigma=sigma[:,:,:,None])
        output = vector_map - low_fq
        return output

    heat_kernel_list = []
    num_atlas = len(multi_atlas_proba_list)

    # Normalize the input image
    img_norm = normalize_image(image)

    # atlas_warped_img_norm_list = []
    for i in range(num_atlas):
        # Normalize the atlas image intensity (zero mean, unit variance for each volume)
        atlas_img_norm = normalize_image(multi_atlas_warped_image_list[i])

        # Compute the intensity term (LSSD)
        ssd = (atlas_img_norm - img_norm) ** 2
        lssd = apply_cubic_Bsplines_kernel(ssd)

        # Remove the low frequencies of the deformations
        disp = high_pass_filter(deformation_field_list[i])

        # Compute the displacement field norm (in mm)
        disp_norm = np.linalg.norm(disp * spacing[None,None,None,:], axis=-1)

        # Compute the heat kernel maps
        distance_map = 0.5 * lssd + 0.5 * disp_norm
        heat_kernel = np.exp(-distance_map**2)
        heat_kernel_list.append(heat_kernel)

    return heat_kernel_list
