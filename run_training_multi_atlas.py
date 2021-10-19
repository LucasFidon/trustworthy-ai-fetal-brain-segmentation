import os
from time import time
import numpy as np
import nibabel as nib
from src.utils.definitions import *
from src.utils.utils import get_feta_info
from src.evaluation.utils import print_results, compute_evaluation_metrics
from src.multi_atlas.inference import multi_atlas_segmentation
from src.multi_atlas.utils import get_atlas_list

SAVE_FOLDER = '/data/saved_res_fetal_multiatlas21_v1'
ATLAS_FUSION_METHODS = ['GIF', 'mean']
ATLAS_METRIC_NAMES = METRIC_NAMES + ['missing_coverage']


def main(training_data_path, pred_folder):
    # Initialize the metric dict
    metrics_per_cond = {
     cond: {
        method: {'%s_%s' % (metric, roi): [] for roi in ALL_ROI for metric in ATLAS_METRIC_NAMES}
            for method in ATLAS_FUSION_METHODS
        }
        for cond in CONDITIONS
    }
    pred_dict = {}

    # Get data information
    patid_ga, patid_cond = get_feta_info()

    sample_folders = [n for n in os.listdir(training_data_path) if '.' not in n]
    for f_n in sample_folders[:10]:
        patid = f_n
        if not patid in list(patid_ga.keys()):
            print('\n*** Unknown GA. \nSkip %s.' % f_n)
            continue
        print('\n--------------')
        print('Start inference for case %s' % f_n)
        # Paths of input
        input_path = os.path.join(training_data_path, f_n, 'srr.nii.gz')
        mask_path = os.path.join(training_data_path, f_n, 'mask.nii.gz')
        output_path = os.path.join(pred_folder, f_n)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # Info about the case
        ga = patid_ga[patid]  # GA rounded to the closest week
        cond = patid_cond[patid]
        print('GA: %d weeks, Condition: %s' % (ga, cond))

        img_nii = nib.load(input_path)
        mask_nii = nib.load(mask_path)

        # Compute the automatic segmentations
        atlas_list = get_atlas_list(ga=ga, condition=cond, ga_delta_max=1)
        print('\nStart atlas propagation using the volumes')
        print(atlas_list)
        atlas_pred_save_folder = os.path.join(output_path, 'atlas_pred')
        for fusion_method in ATLAS_FUSION_METHODS:
            pred_atlas_path = os.path.join(output_path, '%s_atlas_%s.nii.gz' % (f_n, fusion_method))
            pred_proba_atlas = multi_atlas_segmentation(
                img_nii,
                mask_nii,
                atlas_folder_list=atlas_list,
                grid_spacing=GRID_SPACING,
                be=BE,
                le=LE,
                lp=LP,
                save_folder=atlas_pred_save_folder,
                only_affine=False,
                merging_method=fusion_method,
            )
            pred_atlas = np.argmax(pred_proba_atlas, axis=3).astype(np.uint8)
            pred_atlas_nii = nib.Nifti1Image(pred_atlas, img_nii.affine)
            nib.save(pred_atlas_nii, pred_atlas_path)
            pred_dict[fusion_method] = pred_atlas_path

        # Evaluation
        gt_seg_path = os.path.join(training_data_path, f_n, 'parcellation.nii.gz')
        for method in ATLAS_FUSION_METHODS:
            dice, haus, cove = compute_evaluation_metrics(
                pred_dict[method],
                gt_seg_path,
                dataset_path=training_data_path,
                compute_coverage_distance=True,
            )
            for roi in DATASET_LABELS[training_data_path]:
                metrics_per_cond[cond][method]['dice_%s' % roi].append(dice[roi])
                metrics_per_cond[cond][method]['hausdorff_%s' % roi].append(haus[roi])
                metrics_per_cond[cond][method]['missing_coverage_%s' % roi].append(cove[roi])

    # Save and print the metrics aggregated
    for cond in CONDITIONS:
        print('\n%s\n-------' % cond)
        save_metrics_path = os.path.join(pred_folder, 'metrics_%s.pkl' % cond.replace(' ', '_'))
        print_results(
            metrics_per_cond[cond],
            method_names=ATLAS_FUSION_METHODS,
            save_path=save_metrics_path,
        )


if __name__ == '__main__':
    t_start = time()
    if not os.path.exists(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)
    main(TRAINING_DATA_DIR, SAVE_FOLDER)
    total_time = int(time() - t_start)
    print('\nTotal time=%dmin%dsec' % (total_time // 60, total_time % 60))
