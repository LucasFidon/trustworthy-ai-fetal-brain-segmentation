import os
import numpy as np
import nibabel as nib
from time import time
import SimpleITK as sitk
from src.utils.definitions import *
from src.utils.utils import get_feta_info
from src.evaluation.utils import print_results, compute_evaluation_metrics
from src.deep_learning.inference_nnunet import load_softmax
from src.multi_atlas.inference import multi_atlas_segmentation
from src.multi_atlas.utils import get_atlas_list
from src.segmentations_fusion.dempster_shaffer import merge_deep_and_atlas_seg, dempster_add_intensity_prior

# DATA_DIR = [CORRECTED_ZURICH_DATA_DIR, EXCLUDED_ZURICH_DATA_DIR, FETA_IRTK_DIR]
# DATA_DIR = [CDH_LEUVEN_TESTINGSET, DATA_FOLDER_CONTROLS2_PARTIAL_FULLYSEG]
# DATA_DIR = [DATA_FOLDER_THOMAS_GROUP1, DATA_FOLDER_THOMAS_GROUP2]
DATA_DIR = [
    CDH_LEUVEN_TESTINGSET, DATA_FOLDER_CONTROLS2_PARTIAL_FULLYSEG, SB_FRED,
    DATA_FOLDER_THOMAS_GROUP1, DATA_FOLDER_THOMAS_GROUP2,
    CORRECTED_ZURICH_DATA_DIR, EXCLUDED_ZURICH_DATA_DIR, FETA_IRTK_DIR
]
# DATA_DIR = [SB_FRED]

SAVE_FOLDER = '/data/saved_res_fetal_trust21_v3'
DO_BIAS_FIELD_CORRECTION = True  # Will be ignored for data from Leuven
MERGING_MULTI_ATLAS = 'GIF'  # Can be 'GIF' or 'mean'
# MERGING_MULTI_ATLAS = 'mean'
DO_BILATERAL_FILTERING = False
REUSE_CNN_PRED = True  # Set to False if you want to force recomputing the trustworthy segmentations
REUSE_ATLAS_PRED = True  # Set to False if you want to force recomputing the registration
FORCE_RECOMPUTE_HEAT_MAP = False  # This might lead to recomputing the registrations


def apply_bias_field_corrections(img_path, mask_path, save_img_path):
    input_img = sitk.ReadImage(img_path, sitk.sitkFloat32)
    mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetBiasFieldFullWidthAtHalfMaximum(0.15)
    corrector.SetConvergenceThreshold(1e-6)
    corrector.SetSplineOrder(3)
    corrector.SetWienerFilterNoise(0.11)
    t0 = time()
    print('Estimate the bias field inhomogeneity...')
    output = corrector.Execute(input_img, mask)
    t1 = time()
    print('Bias field inhomogeneity estimated in %.0fsec' % (t1 - t0))
    sitk.WriteImage(output, save_img_path)


def main(dataset_path_list):
    pred_folder = os.path.join(SAVE_FOLDER, 'nnunet_task225')
    if not os.path.exists(pred_folder):
        os.mkdir(pred_folder)

    # Initialize the metric dict
    metrics = {
        center: {
            cond: {
                method: {'%s_%s' % (metric, roi): [] for roi in ALL_ROI for metric in METRIC_NAMES}
                for method in METHOD_NAMES
            }
            for cond in CONDITIONS
        }
        for center in CENTERS
    }
    pred_dict = {}

    # Get all data info
    patid_ga, patid_cond = get_feta_info()

    # Run the batch inference
    for dataset in dataset_path_list:
        sample_folders = [n for n in os.listdir(dataset) if '.' not in n]
        center_val = DATASET_GROUPS[dataset]
        for f_n in sample_folders:
            # Get case info
            patid = f_n.replace('feta', '')
            if not patid in list(patid_ga.keys()):
                print('\n*** Unknown GA. \nSkip %s.' % f_n)
                continue
            # if patid != 'sub-052':
            #     print('Skip %s' % f_n)
            #     continue
            print('\n--------------')
            ga = patid_ga[patid]  # GA rounded to the closest week
            cond = patid_cond[patid]

            # Paths of input
            input_path = os.path.join(dataset, f_n, 'srr.nii.gz')
            if not os.path.exists(input_path):
                input_path = os.path.join(dataset, f_n, 'srr_template.nii.gz')
            mask_path = os.path.join(dataset, f_n, 'mask.nii.gz')
            if not os.path.exists(mask_path):
                mask_path = os.path.join(dataset, f_n, 'srr_template_mask.nii.gz')
            output_path = os.path.join(pred_folder, f_n)
            if not os.path.exists(output_path):
                os.mkdir(output_path)

            # Preprocessing
            if DO_BIAS_FIELD_CORRECTION and center_val == 'out':
                print('\n*** Use bias field correction for %s' % patid)
                pre_input_path = os.path.join(output_path, 'srr_preprocessed.nii.gz')
                if not os.path.exists(pre_input_path):
                    apply_bias_field_corrections(input_path, mask_path, pre_input_path)
                input_path = pre_input_path

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

            # Inference
            skip_inference = False
            if REUSE_CNN_PRED and REUSE_ATLAS_PRED and not FORCE_RECOMPUTE_HEAT_MAP:
                skip_inference = os.path.exists(pred_path) and os.path.exists(pred_atlas_path) and os.path.exists(pred_trustworthy_path) and os.path.exists(pred_trustworthy_atlas_only_path)
            if skip_inference:
                print('Skip inference for %s.\nThe predictions already exists.' % f_n)
            else:
                print('\nStart inference for case %s' % f_n)
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
                if cond == 'Pathological':
                    atlas_list = get_atlas_list(ga=ga, condition='Neurotypical', ga_delta_max=DELTA_GA_CONTROL)
                    atlas_list += get_atlas_list(ga=ga, condition='Spina Bifida', ga_delta_max=DELTA_GA_SPINA_BIFIDA)
                elif cond == 'Neurotypical':
                    atlas_list = get_atlas_list(ga=ga, condition='Neurotypical', ga_delta_max=DELTA_GA_CONTROL)
                else:
                    assert cond == 'Spina Bifida', 'Unknown condition %s' % cond
                    atlas_list = get_atlas_list(ga=ga, condition='Spina Bifida', ga_delta_max=DELTA_GA_SPINA_BIFIDA)
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
                    reuse_existing_pred=REUSE_ATLAS_PRED,
                    force_recompute_heat_kernels=FORCE_RECOMPUTE_HEAT_MAP,
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
                    condition=cond,  # Used to know which margins to use
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
                    denoise=DO_BILATERAL_FILTERING,
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
            gt_seg_path = os.path.join(dataset, f_n, 'parcellation.nii.gz')
            for method in METHOD_NAMES:
                dice, haus = compute_evaluation_metrics(pred_dict[method], gt_seg_path, dataset_path=dataset)
                for roi in DATASET_LABELS[dataset]:
                    if not roi in ALL_ROI:
                        continue
                    metrics[center_val][cond][method]['dice_%s' % roi].append(dice[roi])
                    metrics[center_val][cond][method]['hausdorff_%s' % roi].append(haus[roi])

    # Save and print the metrics aggregated
    for center in CENTERS:
        print('=======\n%s\n=======' % center)
        for cond in CONDITIONS:
            print('\n%s\n-------' % cond)
            save_metrics_path = os.path.join(
                pred_folder,
                'metrics_%s-distribution_%s.pkl' % (center, cond.replace(' ', '_')),
            )
            print_results(
                metrics[center][cond],
                method_names=METHOD_NAMES,
                save_path=save_metrics_path,
            )


if __name__ == '__main__':
    t_start = time()
    if not os.path.exists(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)
    main(DATA_DIR)
    total_time = int(time() - t_start)
    print('\nTotal time=%dmin%dsec' % (total_time // 60, total_time % 60))
