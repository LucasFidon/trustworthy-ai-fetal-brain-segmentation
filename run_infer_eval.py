import os
import numpy as np
import nibabel as nib
from time import time
from src.utils.definitions import *
from src.utils.utils import get_feta_info
from src.evaluation.utils import print_results, compute_evaluation_metrics
from src.deep_learning.inference_nnunet import load_softmax
from src.multi_atlas.inference import multi_atlas_segmentation
from src.multi_atlas.utils import get_atlas_list
from src.segmentations_fusion.dempster_shaffer import merge_deep_and_atlas_seg, dempster_add_intensity_prior

# DATA_DIR = [FETA_CHALLENGE_DIR]
# DATA_DIR = [CORRECTED_ZURICH_DATA_DIR, EXCLUDED_ZURICH_DATA_DIR, FETA_CHALLENGE_DIR]
# DATA_DIR = [CDH_LEUVEN_TESTINGSET, DATA_FOLDER_CONTROLS2_PARTIAL_FULLYSEG, SB_FRED]
# DATA_DIR = [DATA_FOLDER_THOMAS_GROUP1, DATA_FOLDER_THOMAS_GROUP2]
# DATA_DIR = [
#     CDH_LEUVEN_TESTINGSET, DATA_FOLDER_CONTROLS2_PARTIAL_FULLYSEG, SB_FRED,
#     DATA_FOLDER_THOMAS_GROUP1, DATA_FOLDER_THOMAS_GROUP2,
# ]
DATA_DIR = [SB_FRED]
SAVE_FOLDER = '/data/saved_res_fetal_trust21_v3'
DO_EVAL = True
MERGING_MULTI_ATLAS = 'GIF'  # Can be 'GIF' or 'mean'
# MERGING_MULTI_ATLAS = 'mean'


def main(dataset_path_list):
    pred_folder = os.path.join(SAVE_FOLDER, 'nnunet_task225')
    if not os.path.exists(pred_folder):
        os.mkdir(pred_folder)

    # Initialize the metric dict
    metrics_per_cond = {
     cond: {
        method: {'%s_%s' % (metric, roi): [] for roi in ALL_ROI for metric in METRIC_NAMES}
            for method in METHOD_NAMES
        }
        for cond in CONDITIONS
    }
    pred_dict = {}

    # Get data info
    patid_ga, patid_cond = get_feta_info()

    # Run the batch inference
    for dataset in dataset_path_list:
        sample_folders = [n for n in os.listdir(dataset) if '.' not in n]
        for f_n in sample_folders:
            patid = f_n.replace('feta', '')
            if not patid in list(patid_ga.keys()):
                print('\n*** Unknown GA. \nSkip %s.' % f_n)
                continue
            # if patid != 'sub-052':
            #     print('Skip %s' % f_n)
            #     continue
            print('\n--------------')
            # Paths of input
            input_path = os.path.join(dataset, f_n, 'srr.nii.gz')
            if not os.path.exists(input_path):
                input_path = os.path.join(dataset, f_n, 'srr_template.nii.gz')
            mask_path = os.path.join(dataset, f_n, 'mask.nii.gz')
            if not os.path.exists(mask_path):
                mask_path = os.path.join(dataset, f_n, 'srr_template_mask.nii.gz')
            output_path = os.path.join(pred_folder, f_n)

            # Info about the case
            ga = patid_ga[patid]  # GA rounded to the closest week
            cond = patid_cond[patid]

            # Set the predictions paths
            pred_path = os.path.join(output_path, '%s.nii.gz' % f_n)  # pred segmentation using CNN only
            pred_atlas_path = os.path.join(output_path, '%s_atlas.nii.gz' % f_n)
            pred_trustworthy_atlas_only_path = os.path.join(output_path, '%s_trustworthy_atlas_only.nii.gz' % f_n)
            pred_trustworthy_path = os.path.join(output_path, '%s_trustworthy.nii.gz' % f_n)
            pred_dict['cnn'] = pred_path
            pred_dict['atlas'] = pred_atlas_path
            pred_dict['trustworthy_atlas_only'] = pred_trustworthy_atlas_only_path
            pred_dict['trustworthy'] = pred_trustworthy_path
            pred_softmax_path = os.path.join(output_path, '%s.npz' % f_n)
            volume_info_path = os.path.join(output_path, '%s.pkl' % f_n)  # info about the volume and preprocessing doen by nnUNet
            if not os.path.exists(output_path):
                os.mkdir(output_path)

            # Inference
            # skip_inference = False
            skip_inference = os.path.exists(pred_path) and os.path.exists(pred_atlas_path) and os.path.exists(pred_trustworthy_path) and os.path.exists(pred_trustworthy_atlas_only_path)
            if skip_inference:
                print('Skip inference for %s.\nThe predictions already exists.' % f_n)
            else:
                print('Start inference for case %s' % f_n)
                # CNN inference
                cmd_options = '--input %s --mask %s --output_folder %s --fold all --task Task225_FetalBrain3dTrust --save_npz' % \
                    (input_path, mask_path, output_path)
                cmd = 'python %s/src/deep_learning/inference_nnunet.py %s' % (REPO_PATH, cmd_options)
                print(cmd)
                os.system(cmd)

                # Trustworthy - atlas
                # Load the softmax prediction, img and mask
                img_nii = nib.load(input_path)
                img = img_nii.get_fdata().astype(np.float32)
                mask_nii = nib.load(mask_path)
                mask = mask_nii.get_fdata().astype(np.uint8)
                softmax = load_softmax(pred_softmax_path, volume_info_path)

                # Propagate the atlas volumes segmentation
                atlas_list = get_atlas_list(ga=ga, condition=cond, ga_delta_max=1)
                print('\nStart atlas propagation using the volumes')
                print(atlas_list)
                atlas_pred_save_folder = os.path.join(output_path, 'atlas_pred')
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
                    merging_method=MERGING_MULTI_ATLAS,
                )

                # Save the atlas-based prediction
                pred_atlas = np.argmax(pred_proba_atlas, axis=3).astype(np.uint8)
                pred_atlas_nii = nib.Nifti1Image(pred_atlas, img_nii.affine)
                nib.save(pred_atlas_nii, pred_atlas_path)

                # Transpose the atlas proba to match PyTorch convention
                pred_proba_atlas = np.transpose(pred_proba_atlas, axes=(3, 0, 1, 2))

                # Take a weighted average of the CNN and atlas predicted proba
                pred_proba_trustworthy = 5 * softmax + pred_proba_atlas  # 5=nb of CNNs in the ensemble
                pred_proba_trustworthy /= 6

                # Apply Dempster's rule with the atlas prior
                pred_proba_trustworthy = merge_deep_and_atlas_seg(
                    deep_proba=pred_proba_trustworthy,
                    atlas_seg=pred_atlas,
                )

                # Save the trustworthy (atlas only) prediction
                pred_trustworthy = np.argmax(pred_proba_trustworthy, axis=0).astype(np.uint8)
                pred_trustworthy_nii = nib.Nifti1Image(pred_trustworthy, img_nii.affine)
                nib.save(pred_trustworthy_nii, pred_trustworthy_atlas_only_path)

                # Trustworthy AI with the intensity prior
                pred_proba_trustworthy = dempster_add_intensity_prior(
                    deep_proba=pred_proba_trustworthy,
                    image=img,
                    mask=mask,
                )
                # Save the trustworthy prediction
                pred_trustworthy = np.argmax(pred_proba_trustworthy, axis=0).astype(np.uint8)
                pred_trustworthy_nii = nib.Nifti1Image(pred_trustworthy, img_nii.affine)
                nib.save(pred_trustworthy_nii, pred_trustworthy_path)

                # Clean folder
                if os.path.exists(pred_softmax_path):  # Remove the npz file (it takes a lot of space)
                    os.system('rm %s' % pred_softmax_path)
                if os.path.exists(volume_info_path):  # Delete the pkl file
                    os.system('rm %s' % volume_info_path)

            # Evaluation
            if DO_EVAL:
                gt_seg_path = os.path.join(dataset, f_n, 'parcellation.nii.gz')
                for method in METHOD_NAMES:
                    dice, haus = compute_evaluation_metrics(pred_dict[method], gt_seg_path, dataset_path=dataset)
                    for roi in DATASET_LABELS[dataset]:
                        metrics_per_cond[cond][method]['dice_%s' % roi].append(dice[roi])
                        metrics_per_cond[cond][method]['hausdorff_%s' % roi].append(haus[roi])

    # Save and print the metrics aggregated
    for cond in CONDITIONS:
        print('\n%s\n-------' % cond)
        save_metrics_path = os.path.join(pred_folder, 'metrics_%s.pkl' % cond.replace(' ', '_'))
        print_results(metrics_per_cond[cond], method_names=METHOD_NAMES, save_path=save_metrics_path)


if __name__ == '__main__':
    t_start = time()
    if not os.path.exists(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)
    main(DATA_DIR)
    total_time = int(time() - t_start)
    print('\nTotal time=%dmin%dsec' % (total_time // 60, total_time % 60))
