import os
from time import time
import numpy as np
import nibabel as nib
from src.utils.definitions import *
from src.utils.utils import get_feta_info
from src.evaluation.utils import print_results, compute_evaluation_metrics, print_summary_results
from src.multi_atlas.inference import multi_atlas_segmentation
from src.multi_atlas.utils import get_atlas_list

SAVE_FOLDER = '/data/saved_res_fetal_multiatlas21_v1'
ATLAS_FUSION_METHODS = ['GIF', 'mean']
ATLAS_METRIC_NAMES = METRIC_NAMES + ['missing_coverage', 'number_registrations']

# OPTIONS
ATLAS_SELECTION = ['ALL', 'CONDITION']
ATLAS_FUSION = ['GIF', 'mean']
GA_DELTA_MAX = 4  # max 4
ATLAS_MARGINS = []  #todo
#todo more pre-processing on the atlas segmentation? bilateral filtering, nb components, closing operation
# but maybe this is already reflected in the registration parameters..


def convert_to_patid(folder_name):
    patid = folder_name.replace('-', '_')
    if not 'Study' in folder_name:
        patid = '%s_Study1' % (patid.replace('_', ''))
    return patid

def get_fold0_data():
    def is_in_fold0(folder_name):
        patid = convert_to_patid(folder_name)
        if patid in FOLD_0:
            return True
        else:
            return False
    data_paths = [
        CDH_DOAA_DEC19, CONTROLS_DOAA_OCT20,
        DOAA_BRAIN_LONGITUDINAL_SRR_AND_SEG,
        LEUVEN_MMC, CDH, CONTROLS_WITH_EXT_CSF,
    ]
    fold0_cases_folders = []
    for data_p in data_paths:
        for folder_n in os.listdir(data_p):
            if is_in_fold0(folder_n):
                folder_p = os.path.join(data_p, folder_n)
                fold0_cases_folders.append(folder_p)
    print('Found %d cases for fold0' % len(fold0_cases_folders))
    return fold0_cases_folders


def main(pred_folder):
    # Initialize the metric dict
    metrics_per_cond = {cond: {} for cond in CONDITIONS}
    method_names = []
    for fusion_method in ATLAS_FUSION_METHODS:
        for selection_method in ATLAS_SELECTION:
            for ga_delta in range(0, GA_DELTA_MAX+1):
                method_n = '%s_%s_GAdelta%d' % (fusion_method, selection_method, ga_delta)
                method_names.append(method_n)
                for cond in CONDITIONS:
                    metrics_per_cond[cond][method_n] = {
                        '%s_%s' % (metric, roi): []
                        for roi in ALL_ROI for metric in ATLAS_METRIC_NAMES
                    }
    pred_dict = {}

    # Get data information
    patid_ga, patid_cond = get_feta_info()
    sample_folders = get_fold0_data()

    #todo add AI seg for the fold0 model
    # if we want to evaluate: bilateral filtering, etc...
    for sample_f in sample_folders:
        patid = convert_to_patid(os.path.split(sample_f)[1])
        if not patid in list(patid_ga.keys()):
            print('\n*** Unknown GA. \nSkip %s.' % sample_f)
            continue
        print('\n--------------')
        print('Start inference for case %s' % sample_f)
        # Paths of input
        input_path = os.path.join(sample_f, 'srr.nii.gz')
        mask_path = os.path.join(sample_f, 'mask.nii.gz')
        gt_seg_path = os.path.join(sample_f, 'parcellation.nii.gz')
        output_path = os.path.join(pred_folder, patid)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # Info about the case
        ga = patid_ga[patid]  # GA rounded to the closest week
        cond = patid_cond[patid]
        print('GA: %d weeks, Condition: %s' % (ga, cond))

        img_nii = nib.load(input_path)
        mask_nii = nib.load(mask_path)

        # Compute the automatic segmentations
        atlas_list = get_atlas_list(
            ga=ga,
            condition='Pathological',  # To use all the atlases
            ga_delta_max=GA_DELTA_MAX,
        )
        print('\nStart atlas propagation using the volumes')
        print(atlas_list)
        atlas_pred_save_folder = os.path.join(output_path, 'atlas_pred')
        for fusion_method in ATLAS_FUSION_METHODS:
            for selection_method in ATLAS_SELECTION:
                for ga_delta in range(0, GA_DELTA_MAX+1):
                    method_n = '%s_%s_GAdelta%d' % (fusion_method, selection_method, ga_delta)
                    print('\n***\033[93m Start inference for method %s\033[0m' % method_n)
                    pred_atlas_path = os.path.join(
                        output_path,
                        '%s_atlas_%s.nii.gz' % (patid, method_n),
                    )
                    pred_dict[method_n] = pred_atlas_path
                    # if os.path.exists(pred_atlas_path):
                    #     print('Skip. %s already exists.' % pred_atlas_path)
                    #     continue

                    # Multi-atlas segmentation
                    atlas_list = get_atlas_list(
                        ga=ga,
                        condition=cond if (selection_method == 'CONDITION') else 'Pathological',
                        ga_delta_max=ga_delta,
                    )
                    if not os.path.exists(pred_atlas_path):
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
                            reuse_existing_pred=True,
                        )
                        pred_atlas = np.argmax(pred_proba_atlas, axis=3).astype(np.uint8)
                        pred_atlas_nii = nib.Nifti1Image(pred_atlas, img_nii.affine)
                        nib.save(pred_atlas_nii, pred_atlas_path)

                    # Evaluation
                    dice, haus, cove = compute_evaluation_metrics(
                        pred_dict[method_n],
                        gt_seg_path,
                        dataset_path=TRAINING_DATA_PREPROCESSED_DIR,  # only used to know what ROIs to evaluate...
                        compute_coverage_distance=True,
                    )
                    for roi in ALL_ROI:
                        metrics_per_cond[cond][method_n]['dice_%s' % roi].append(dice[roi])
                        metrics_per_cond[cond][method_n]['hausdorff_%s' % roi].append(haus[roi])
                        metrics_per_cond[cond][method_n]['missing_coverage_%s' % roi].append(cove[roi])
                        metrics_per_cond[cond][method_n]['number_registrations_%s' % roi].append(len(atlas_list))


        # # Evaluation
        # gt_seg_path = os.path.join(sample_f, 'parcellation.nii.gz')
        # for fusion_method in ATLAS_FUSION_METHODS:
        #     for selection_method in ATLAS_SELECTION:
        #         for ga_delta in range(0, GA_DELTA_MAX+1):
        #             method = '%s_%s_GAdelta%d' % (fusion_method, selection_method, ga_delta)
        #             dice, haus, cove = compute_evaluation_metrics(
        #                 pred_dict[method],
        #                 gt_seg_path,
        #                 dataset_path=TRAINING_DATA_PREPROCESSED_DIR,  # only used to know what ROIs to evaluate...
        #                 compute_coverage_distance=True,
        #             )
        #             for roi in ALL_ROI:
        #                 metrics_per_cond[cond][method]['dice_%s' % roi].append(dice[roi])
        #                 metrics_per_cond[cond][method]['hausdorff_%s' % roi].append(haus[roi])
        #                 metrics_per_cond[cond][method]['missing_coverage_%s' % roi].append(cove[roi])

    # Save and print the metrics aggregated
    for cond in CONDITIONS:
        if cond == 'Pathological':  # no other ABN in fol0
            continue
        print('\n%s\n-------' % cond)
        save_metrics_path = os.path.join(pred_folder, 'metrics_%s.pkl' % cond.replace(' ', '_'))
        print_results(
            metrics_per_cond[cond],
            method_names=method_names,
            metric_names=ATLAS_METRIC_NAMES,
            save_path=save_metrics_path,
        )

    # Print the average mean metrics across ROI
    for cond in CONDITIONS:
        if cond == 'Pathological':  # no other ABN in fol0
            continue
        print('\n%s\n-------' % cond)
        print_summary_results(
            metrics_per_cond[cond],
            method_names=method_names,
            metric_names=ATLAS_METRIC_NAMES,
        )


if __name__ == '__main__':
    t_start = time()
    if not os.path.exists(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)
    main(SAVE_FOLDER)
    total_time = int(time() - t_start)
    print('\nTotal time=%dmin%dsec' % (total_time // 60, total_time % 60))
