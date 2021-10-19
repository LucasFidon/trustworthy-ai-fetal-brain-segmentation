"""
@brief  Evaluation metrics for segmentation applications.

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
@date   30 Oct 2019.
"""

import numpy as np
from scipy import ndimage
from src.evaluation.evaluation_metrics import lookup_tables


def _binarize(seg, fg_class):
    """
    Binarize a segmentation with label 1 for pixels/voxels the foreground class
    and label 0 for pixels/voxels the other classes.
    :param seg: int numpy array.
    :param fg_class: int or int list; class in seg corresponding to the foreground.
    :return: binary segmentation corresponding to seg for the foreground class fg_class.
    """
    bin_seg = np.zeros_like(seg, dtype=bool)
    if isinstance(fg_class, int):
        bin_seg[seg == fg_class] = True
    else:  # list of integers
        for c in fg_class:
            bin_seg[seg == c] = True
    return bin_seg


# Basic metrics
def true_positives(seg_pred, seg_gt):
    """
    Number of True Positives
    for the predicted segmentation seg_pred
    and the ground-truth segmentation seg_gt.
    :param seg_pred: numpy bool array.
    :param seg_gt: numpy bool array.
    :return: int; number of true positives.
    """
    assert seg_pred.dtype == np.bool, "seg_1 should be of type bool, " \
                                      "found %s instead." % seg_pred.dtype
    assert seg_gt.dtype == np.bool, "seg_2 should be of type bool, " \
                                    "found %s instead." % seg_gt.dtype
    num_tp = np.sum(seg_pred * seg_gt)
    return num_tp


def false_positives(seg_pred, seg_gt):
    """
    Number of False Positives
    for the predicted segmentation seg_pred
    and the ground-truth segmentation seg_gt.
    :param seg_pred: numpy bool array.
    :param seg_gt: numpy bool array.
    :return: int; number of false positives.
    """
    assert seg_pred.dtype == np.bool, "seg_1 should be of type bool, " \
                                      "found %s instead." % seg_pred.dtype
    assert seg_gt.dtype == np.bool, "seg_2 should be of type bool, " \
                                    "found %s instead." % seg_gt.dtype
    num_fp = np.sum(seg_pred * (1 - seg_gt))
    return num_fp


def false_negatives(seg_pred, seg_gt):
    """
    Number of False Negatives
    for the predicted segmentation seg_pred
    and the ground-truth segmentation seg_gt.
    :param seg_pred: numpy bool array.
    :param seg_gt: numpy bool array.
    :return: int; number of false negatives.
    """
    assert seg_pred.dtype == np.bool, "seg_1 should be of type bool, " \
                                      "found %s instead." % seg_pred.dtype
    assert seg_gt.dtype == np.bool, "seg_2 should be of type bool, " \
                                    "found %s instead." % seg_gt.dtype
    num_fn = np.sum((1 - seg_pred) * seg_gt)
    return num_fn


def true_negatives(seg_pred, seg_gt):
    """
    Number of True Negatives
    for the predicted segmentation seg_pred
    and the ground-truth segmentation seg_gt.
    :param seg_pred: numpy bool array.
    :param seg_gt: numpy bool array.
    :return: int; number of true negatives.
    """
    assert seg_pred.dtype == np.bool, "seg_1 should be of type bool, " \
                                      "found %s instead." % seg_pred.dtype
    assert seg_gt.dtype == np.bool, "seg_2 should be of type bool, " \
                                    "found %s instead." % seg_gt.dtype
    num_tn = np.sum((1 - seg_pred) * (1 - seg_gt))
    return num_tn


# Dice scores and variants
def dice_score(seg_1, seg_2, fg_class):
    """
    Compute the Dice score for class fg_class
    between the segmentations seg_1 and seg_2.
    For explanation about the formula used to compute the Dice score coefficient,
    see for example:
    "Generalised Wasserstein Dice Score for Imbalanced Multi-class Segmentation
    using Holistic Convolutional Networks", L. Fidon et al, BrainLes 2017.
    :param seg_1: numpy int array.
    :param seg_2: numpy int array.
    :param fg_class: int or int list.
    :return: float; Dice score value.
    """
    assert seg_1.shape == seg_2.shape, "seg_1 and seg_2 must have the same shape " \
                                       "to compute their dice score."
    # binarize the segmentations
    bin_seg_1 = _binarize(seg_1, fg_class=fg_class)
    bin_seg_2 = _binarize(seg_2, fg_class=fg_class)
    # compute the Dice score value
    tp = true_positives(bin_seg_1, bin_seg_2)
    fp = false_positives(bin_seg_1, bin_seg_2)
    fn = false_negatives(bin_seg_1, bin_seg_2)
    if tp + fp + fn == 0:  # empty foreground for seg_1 and seg_2
        dice_val = 1.
    else:
        dice_val = 2. * tp / (2. * tp + fp + fn)
    return dice_val


def mean_dice_score(seg_1, seg_2, labels_list=[0, 1]):
    """
    Compute the mean of the Dice scores for the labels in labels_list
    between the segmentations seg_1 and seg_2.
    :param seg_1: numpy int array.
    :param seg_2: numpy int array.
    :param labels_list: int list.
    :return:
    """
    assert len(labels_list) > 0, "the list of labels to consider for the mean dice score" \
                                 "must contain at least one label"
    dice_values = []
    for l in labels_list:
        dice = dice_score(seg_1, seg_2, fg_class=l)
        dice_values.append(dice)
    mean_dice = np.mean(dice_values)
    return mean_dice


# Jaccard index and variants
def jaccard(seg_1, seg_2, fg_class):
    """
    Compute the Jaccard for class fg_class
    between the segmentations seg_1 and seg_2.
    :param seg_1: numpy int array.
    :param seg_2: numpy int array.
    :param fg_class: int or int list.
    :return: float; Jaccard value.
    """
    assert seg_1.shape == seg_2.shape, "seg_1 and seg_2 must have the same shape " \
                                       "to compute their dice score"
    # binarize the segmentations
    bin_seg_1 = _binarize(seg_1, fg_class=fg_class)
    bin_seg_2 = _binarize(seg_2, fg_class=fg_class)
    # compute the Jaccard index value
    tp = true_positives(bin_seg_1, bin_seg_2)
    fp = false_positives(bin_seg_1, bin_seg_2)
    fn = false_negatives(bin_seg_1, bin_seg_2)
    if tp + fp + fn == 0:  # empty foreground for seg_1 and seg_2
        jaccard = 1.
    else:
        jaccard = tp / (tp + fp + fn)
    return jaccard


# Surface distances

def haussdorff_distance(mask_gt, mask_pred, fg_class,
                       percentile=100, spacing_mm=[0.8, 0.8, 0.8]):
    bin_mask_gt = np.squeeze(_binarize(mask_gt, fg_class=fg_class))
    bin_mask_pred = np.squeeze(_binarize(mask_pred, fg_class=fg_class))

    surface_distances = compute_surface_distances(
        bin_mask_gt, bin_mask_pred, spacing_mm)

    haussdorff_dist_value = compute_robust_hausdorff(surface_distances, percentile)

    return haussdorff_dist_value


def missing_coverage_distance(mask_gt, mask_pred, fg_class,
                              percentile=100, spacing_mm=[0.8, 0.8, 0.8]):
    # HD that penalizes only the false negatives
    bin_mask_gt = np.squeeze(_binarize(mask_gt, fg_class=fg_class))
    bin_mask_pred = np.squeeze(_binarize(mask_pred, fg_class=fg_class))

    # Take the union of the ground-truth and the predicted mask
    bin_mask_union = np.logical_or(bin_mask_gt, bin_mask_pred)

    surface_distances = compute_surface_distances(
        bin_mask_union, bin_mask_pred, spacing_mm)

    coverage_dist_value = compute_robust_hausdorff(surface_distances, percentile)

    return coverage_dist_value


def compute_surface_distances(mask_gt, mask_pred, spacing_mm):
    """
    Compute closest distances from all surface points to the other surface.
    Finds all surface elements "surfels" in the ground truth mask `mask_gt` and
    the predicted mask `mask_pred`, computes their area in mm^2 and the distance
    to the closest point on the other surface. It returns two sorted lists of
    distances together with the corresponding surfel areas. If one of the masks
    is empty, the corresponding lists are empty and all distances in the other
    list are `inf`.
    :param mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    :param mask_pred: 3-dim Numpy array of type bool. The predicted mask.
    :param spacing_mm: 3-element list-like structure. Voxel spacing in x0, x1 and x2
        direction.
    :return: A dict with:
    "distances_gt_to_pred": 1-dim numpy array of type float. The distances in mm
        from all ground truth surface elements to the predicted surface,
        sorted from smallest to largest.
    "distances_pred_to_gt": 1-dim numpy array of type float. The distances in mm
        from all predicted surface elements to the ground truth surface,
        sorted from smallest to largest.
    "surfel_areas_gt": 1-dim numpy array of type float. The area in mm^2 of
        the ground truth surface elements in the same order as
        distances_gt_to_pred
    "surfel_areas_pred": 1-dim numpy array of type float. The area in mm^2 of
        the predicted surface elements in the same order as
        distances_pred_to_gt
    """
    # compute the area for all 256 possible surface elements
    # (given a 2x2x2 neighbourhood) according to the spacing_mm
    neighbour_code_to_surface_area = np.zeros([256])
    for code in range(256):
        normals = np.array(lookup_tables.neighbour_code_to_normals[code])
        sum_area = 0
        for normal_idx in range(normals.shape[0]):
            # normal vector
            n = np.zeros([3])
            n[0] = normals[normal_idx, 0] * spacing_mm[1] * spacing_mm[2]
            n[1] = normals[normal_idx, 1] * spacing_mm[0] * spacing_mm[2]
            n[2] = normals[normal_idx, 2] * spacing_mm[0] * spacing_mm[1]
            area = np.linalg.norm(n)
            sum_area += area
        neighbour_code_to_surface_area[code] = sum_area

    # compute the bounding box of the masks to trim
    # the volume to the smallest possible processing subvolume
    mask_all = mask_gt | mask_pred
    bbox_min = np.zeros(3, np.int64)
    bbox_max = np.zeros(3, np.int64)

    # max projection to the x0-axis
    proj_0 = np.max(np.max(mask_all, axis=2), axis=1)
    idx_nonzero_0 = np.nonzero(proj_0)[0]
    if len(idx_nonzero_0) == 0:  # pylint: disable=g-explicit-length-test
        return {"distances_gt_to_pred": np.array([]),
                "distances_pred_to_gt": np.array([]),
                "surfel_areas_gt": np.array([]),
                "surfel_areas_pred": np.array([])}

    bbox_min[0] = np.min(idx_nonzero_0)
    bbox_max[0] = np.max(idx_nonzero_0)

    # max projection to the x1-axis
    proj_1 = np.max(np.max(mask_all, axis=2), axis=0)
    idx_nonzero_1 = np.nonzero(proj_1)[0]
    bbox_min[1] = np.min(idx_nonzero_1)
    bbox_max[1] = np.max(idx_nonzero_1)

    # max projection to the x2-axis
    proj_2 = np.max(np.max(mask_all, axis=1), axis=0)
    idx_nonzero_2 = np.nonzero(proj_2)[0]
    bbox_min[2] = np.min(idx_nonzero_2)
    bbox_max[2] = np.max(idx_nonzero_2)

    # crop the processing subvolume.
    # we need to zeropad the cropped region with 1 voxel at the lower,
    # the right and the back side. This is required to obtain the "full"
    # convolution result with the 2x2x2 kernel
    cropmask_gt = np.zeros((bbox_max - bbox_min)+2, np.uint8)
    cropmask_pred = np.zeros((bbox_max - bbox_min)+2, np.uint8)

    cropmask_gt[0:-1, 0:-1, 0:-1] = mask_gt[bbox_min[0]:bbox_max[0]+1,
                                            bbox_min[1]:bbox_max[1]+1,
                                            bbox_min[2]:bbox_max[2]+1]

    cropmask_pred[0:-1, 0:-1, 0:-1] = mask_pred[bbox_min[0]:bbox_max[0]+1,
                                                bbox_min[1]:bbox_max[1]+1,
                                                bbox_min[2]:bbox_max[2]+1]

    # compute the neighbour code (local binary pattern) for each voxel
    # the resulting arrays are spacially shifted by minus half a voxel in each
    # axis.
    # i.e. the points are located at the corners of the original voxels
    kernel = np.array([[[128, 64],
                          [32, 16]],
                         [[8, 4],
                          [2, 1]]])
    neighbour_code_map_gt = ndimage.filters.correlate(
        cropmask_gt.astype(np.uint8), kernel, mode="constant", cval=0)
    neighbour_code_map_pred = ndimage.filters.correlate(
        cropmask_pred.astype(np.uint8), kernel, mode="constant", cval=0)

    # create masks with the surface voxels
    borders_gt = ((neighbour_code_map_gt != 0) & (neighbour_code_map_gt != 255))
    borders_pred = ((neighbour_code_map_pred != 0) &
                    (neighbour_code_map_pred != 255))

    # compute the distance transform (closest distance of each voxel to the
    # surface voxels)
    if borders_gt.any():
        distmap_gt = ndimage.morphology.distance_transform_edt(
            ~borders_gt, sampling=spacing_mm)
    else:
        distmap_gt = np.Inf * np.ones(borders_gt.shape)

    if borders_pred.any():
        distmap_pred = ndimage.morphology.distance_transform_edt(
            ~borders_pred, sampling=spacing_mm)
    else:
        distmap_pred = np.Inf * np.ones(borders_pred.shape)

    # compute the area of each surface element
    surface_area_map_gt = neighbour_code_to_surface_area[neighbour_code_map_gt]
    surface_area_map_pred = neighbour_code_to_surface_area[
        neighbour_code_map_pred]

    # create a list of all surface elements with distance and area
    distances_gt_to_pred = distmap_pred[borders_gt]
    distances_pred_to_gt = distmap_gt[borders_pred]
    surfel_areas_gt = surface_area_map_gt[borders_gt]
    surfel_areas_pred = surface_area_map_pred[borders_pred]

    # sort them by distance
    if distances_gt_to_pred.shape != (0,):
        sorted_surfels_gt = np.array(
            sorted(zip(distances_gt_to_pred, surfel_areas_gt)))
        distances_gt_to_pred = sorted_surfels_gt[:, 0]
        surfel_areas_gt = sorted_surfels_gt[:, 1]

    if distances_pred_to_gt.shape != (0,):
        sorted_surfels_pred = np.array(
            sorted(zip(distances_pred_to_gt, surfel_areas_pred)))
        distances_pred_to_gt = sorted_surfels_pred[:, 0]
        surfel_areas_pred = sorted_surfels_pred[:, 1]

    return {"distances_gt_to_pred": distances_gt_to_pred,
            "distances_pred_to_gt": distances_pred_to_gt,
            "surfel_areas_gt": surfel_areas_gt,
            "surfel_areas_pred": surfel_areas_pred}


def compute_robust_hausdorff(surface_distances, percent):
    """
    Computes the robust Hausdorff distance.
    Computes the robust Hausdorff distance. "Robust", because it uses the
    `percent` percentile of the distances instead of the maximum distance. The
    percentage is computed by correctly taking the area of each surface element
    into account.
    Based on
    https://github.com/deepmind/surface-distance/blob/master/surface_distance/metrics.py
    :param surface_distances: dict with "distances_gt_to_pred", "distances_pred_to_gt"
      "surfel_areas_gt", "surfel_areas_pred" created by
      compute_surface_distances()
    :param percent: a float value between 0 and 100.
    :return: a float value. The robust Hausdorff distance in mm.
    """
    distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
    distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
    surfel_areas_gt = surface_distances["surfel_areas_gt"]
    surfel_areas_pred = surface_distances["surfel_areas_pred"]
    if len(distances_gt_to_pred) > 0:  # pylint: disable=g-explicit-length-test
        surfel_areas_cum_gt = np.cumsum(surfel_areas_gt) / np.sum(surfel_areas_gt)
        idx = np.searchsorted(surfel_areas_cum_gt, percent/100.0)
        perc_distance_gt_to_pred = distances_gt_to_pred[
            min(idx, len(distances_gt_to_pred)-1)]
    else:
        perc_distance_gt_to_pred = np.Inf

    if len(distances_pred_to_gt) > 0:  # pylint: disable=g-explicit-length-test
        surfel_areas_cum_pred = (np.cumsum(surfel_areas_pred) /
                                 np.sum(surfel_areas_pred))
        idx = np.searchsorted(surfel_areas_cum_pred, percent/100.0)
        perc_distance_pred_to_gt = distances_pred_to_gt[
            min(idx, len(distances_pred_to_gt)-1)]
    else:
        perc_distance_pred_to_gt = np.Inf

    return max(perc_distance_gt_to_pred, perc_distance_pred_to_gt)
