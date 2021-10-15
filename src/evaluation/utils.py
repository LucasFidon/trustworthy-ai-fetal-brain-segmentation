import os
import numpy as np
import nibabel as nib
import pickle
from src.utils.definitions import *
from src.evaluation.evaluation_metrics.segmentation_metrics import dice_score, haussdorff_distance


def compute_evaluation_metrics(pred_seg_path, gt_seg_path, dataset_path):
    def load_np(seg_path):
        seg = nib.load(seg_path).get_fdata().astype(np.uint8)
        return seg
    # Load the segmentations
    pred_seg_folder, pred_seg_name = os.path.split(pred_seg_path)
    pred_seg = load_np(pred_seg_path)
    gt_seg = load_np(gt_seg_path)
    # Compute the metrics
    dice_values = {}
    haus_values = {}
    for roi in DATASET_LABELS[dataset_path]:
        dice_values[roi] = 100 * dice_score(
            pred_seg,
            gt_seg,
            fg_class=LABELS[roi],
        )
        haus_values[roi] = min(
            MAX_HD,
            haussdorff_distance(
                pred_seg,
                gt_seg,
                fg_class=LABELS[roi],
                percentile=95,
            )
        )
    print('\n\033[92mEvaluation for %s\033[0m' % pred_seg_name)
    print('Dice scores:')
    print(dice_values)
    print('Hausdorff95 distances:')
    print(haus_values)
    return dice_values, haus_values


def print_results(metrics, method_names=METHOD_NAMES, save_path=None):
    print('\nGlobal statistics for the metrics')
    for method in method_names:
        print('\n\033[93m----------')
        print(method.upper())
        print('----------\033[0m')
        for roi in ALL_ROI:
            print('\033[92m%s\033[0m' % roi)
            for metric in METRIC_NAMES:
                key = '%s_%s' % (metric, roi)
                num_data = len(metrics[method][key])
                if num_data == 0:
                    print('No data for %s' % key)
                    continue
                print('%d cases' % num_data)
                mean = np.mean(metrics[method][key])
                std = np.std(metrics[method][key])
                median = np.median(metrics[method][key])
                q3 = np.percentile(metrics[method][key], 75)
                p95 = np.percentile(metrics[method][key], 95)
                q1 = np.percentile(metrics[method][key], 25)
                p5 = np.percentile(metrics[method][key], 25)
                print(key)
                if metric == 'dice':
                    print('mean=%.1f std=%.1f median=%.1f q1=%.1f p5=%.1f' % (mean, std, median, q1, p5))
                else:
                    print('mean=%.1f std=%.1f median=%.1f q3=%.1f p95=%.1f' % (mean, std, median, q3, p95))
            print('-----------')
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)