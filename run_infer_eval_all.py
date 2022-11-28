import os
import numpy as np
import nibabel as nib
from time import time
import SimpleITK as sitk
import pandas as pd
from src.utils.definitions import *
from src.utils.utils import get_feta_info
from src.evaluation.utils import print_results, compute_evaluation_metrics
from src.deep_learning.inference_nnunet import load_softmax
from src.multi_atlas.inference import multi_atlas_segmentation
from src.multi_atlas.utils import get_atlas_list
from src.segmentations_fusion.dempster_shaffer import merge_deep_and_atlas_seg, dempster_add_intensity_prior

# Training set (used to generate the segmentation prior for training)
# DATA_DIR = [
#     TRAINING_DATA_PREPROCESSED_DIR,
# ]
# Testing set
DATA_DIR = [
    CONTROLS_KCL,
    SB_FRED2,
    DOAA_BRAIN_LONGITUDINAL_SRR_AND_SEG2,
    UCLH_MMC_2,
    ZURICH_TEST_DATA_DIR,
    SB_VIENNA,
    CDH_LEUVEN_TESTINGSET, DATA_FOLDER_CONTROLS2_PARTIAL_FULLYSEG, SB_FRED,
    DATA_FOLDER_THOMAS_GROUP1, DATA_FOLDER_THOMAS_GROUP2,
    CORRECTED_ZURICH_DATA_DIR, EXCLUDED_ZURICH_DATA_DIR, FETA_IRTK_DIR,
]

# SAVE_FOLDER = '/data/saved_res_fetal_trust22_training'
SAVE_FOLDER = '/data/saved_res_fetal_trust21_v3'
DO_BIAS_FIELD_CORRECTION = True  # Will be ignored for data from Leuven
MERGING_MULTI_ATLAS = 'GIF'  # Can be 'GIF' or 'mean'
DO_BILATERAL_FILTERING = False  # Not used; option for the data pre-processing in the intensity-based contracts
REUSE_CNN_PRED = True  # Set to False if you want to force recomputing the trustworthy segmentations
REUSE_ATLAS_PRED = True  # Set to False if you want to force recomputing the registration
FORCE_RECOMPUTE_HEAT_MAP = False  # This might lead to recomputing the registrations
INFERENCE_ONLY = False  # Set to true if you do not want to compute the evaluation metrics
ATLAS_ONLY = False  # True to run only the atlas-based inference; Need to use INFERENCE_ONLY=True with that

DEEP_LEARNING_MODELS = ['nnUNet', 'nnUNetSegPrior', 'SwinUNETR']
# DEEP_LEARNING_MODELS = ['SwinUNETR']
TASK_NAME = {
    '225': 'Task225_FetalBrain3dTrust',
    '235': 'Task235_FetalBrain3dTrustSegPrior',
}

METRICS_COLUMN = ['Study', 'GA', 'Condition', 'Center type', 'Methods', 'ROI', 'dice', 'hausdorff']


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


def save_seg(seg, affine, save_path):
    save_folder = os.path.split(save_path)[0]
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    seg_nii = nib.Nifti1Image(seg, affine)
    nib.save(seg_nii, save_path)


def apply_TWAI(ai, fallback, atlas_seg, cond, img, mask_foreground, affine,
               save_path, save_path_intensity_only=None, save_path_atlas_only=None, eps=0.01):
    pred_proba = ai + eps * fallback

    # TWAI atlas
    pred_proba_trustworthy_atlas = merge_deep_and_atlas_seg(
        deep_proba=pred_proba,
        atlas_seg=atlas_seg,
        condition=cond,  # Used to know which margins to use
    )
    if save_path_atlas_only is not None:
        seg_trustworthy_atlas = np.argmax(pred_proba_trustworthy_atlas, axis=0).astype(np.uint8)
        save_seg(seg_trustworthy_atlas, affine=affine, save_path=save_path_atlas_only)

    # TWAI intensity
    if save_path_intensity_only is not None:
        pred_proba_trustworthy_intensity = dempster_add_intensity_prior(
            deep_proba=pred_proba,
            image=img,
            mask=mask_foreground,
            denoise=DO_BILATERAL_FILTERING,
        )
        pred_trustworthy_intensity = np.argmax(pred_proba_trustworthy_intensity, axis=0).astype(np.uint8)
        save_seg(pred_trustworthy_intensity, affine=affine, save_path=save_path_intensity_only)

    # Trustworthy AI with the intensity prior + atlas prior
    pred_proba_trustworthy = dempster_add_intensity_prior(
        deep_proba=pred_proba_trustworthy_atlas,
        image=img,
        mask=mask_foreground,
        denoise=DO_BILATERAL_FILTERING,
    )
    # Save the trustworthy prediction
    pred_trustworthy = np.argmax(pred_proba_trustworthy, axis=0).astype(np.uint8)
    save_seg(pred_trustworthy, affine=affine, save_path=save_path)


def main(dataset_path_list):
    pred_folder = os.path.join(SAVE_FOLDER, 'all')
    if not os.path.exists(pred_folder):
        os.makedirs(pred_folder)

    metric_data = []
    pred_dict = {}

    # Get all data info
    patid_sample = get_feta_info()

    # Run the batch inference
    for dataset in dataset_path_list:
        sample_folders = [n for n in os.listdir(dataset) if '.' not in n]
        for f_n in sample_folders:
            # Get case info
            patid = f_n.replace('feta', '')
            if not patid in list(patid_sample.keys()):
                print('\n*** Unknown sample. \nSkip %s.' % f_n)
                continue
            print('\n--------------')
            sample = patid_sample[patid]
            ga_ori = sample.ga
            # GA is rounded to the closest week and clipped to the range of GA of the atlases
            ga = int(round(ga_ori))
            if ga > MAX_GA:
                print('Found ga=%d for %s. Change it to %d (max value accepted)' % (ga, patid, MAX_GA))
                ga = MAX_GA
            if ga < MIN_GA:
                print('Found ga=%d for %s. Change it to %d (min value accepted)' % (ga, patid, MIN_GA))
                ga = MIN_GA
            cond = sample.cond
            center_val = sample.center

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
            if DO_BIAS_FIELD_CORRECTION and dataset in [FETA_IRTK_DIR, CORRECTED_ZURICH_DATA_DIR, EXCLUDED_ZURICH_DATA_DIR, ZURICH_TEST_DATA_DIR]:
                print('\n*** Use bias field correction for %s' % patid)
                pre_input_path = os.path.join(output_path, 'srr_preprocessed.nii.gz')
                if not os.path.exists(pre_input_path):
                    apply_bias_field_corrections(input_path, mask_path, pre_input_path)
                input_path = pre_input_path

            # Set the predictions paths
            pred_dict['atlas'] = os.path.join(output_path, '%s_atlas.nii.gz' % f_n)
            for deep_model in DEEP_LEARNING_MODELS:
                pred_dict[deep_model] = os.path.join(output_path, deep_model, '%s.nii.gz' % f_n)
                pred_dict['%s_add_fusion' % deep_model] = os.path.join(
                    output_path, '%s_%s_add_fusion.nii.gz' % (f_n, deep_model))
                pred_dict['%s_mult_fusion' % deep_model] = os.path.join(
                    output_path, '%s_%s_mult_fusion.nii.gz' % (f_n, deep_model))
                for ai in [deep_model, '%s_add_fusion' % deep_model, '%s_mult_fusion' % deep_model]:
                    pred_dict['%s_trustworthy_atlas_only' % ai] = os.path.join(
                        output_path, '%s_%s_trustworthy_atlas_only.nii.gz' % (f_n, ai))
                    pred_dict['%s_trustworthy_intensity_only' % ai] = os.path.join(
                        output_path, '%s_%s_trustworthy_intensity_only.nii.gz' % (f_n, ai))
                    pred_dict['%s_trustworthy' % ai] = os.path.join(
                        output_path, '%s_%s_trustworthy.nii.gz' % (f_n, ai))

            pred_softmax_path = {
                'nnUNet': os.path.join(output_path, 'nnUNet', '%s.npz' % f_n),
                'nnUNetSegPrior': os.path.join(output_path, 'nnUNetSegPrior', '%s.npz' % f_n),
                'SwinUNETR': os.path.join(SWINUNETR_TEST_PRED[0], '%s.nii.gz' % f_n),
            }

            volume_info_path = {
                'nnUNet': os.path.join(output_path, 'nnUNet', '%s.pkl' % f_n),
                'nnUNetSegPrior': os.path.join(output_path, 'nnUNetSegPrior', '%s.pkl' % f_n),
            }

            # Inference
            skip_inference = False
            if REUSE_CNN_PRED and REUSE_ATLAS_PRED and not FORCE_RECOMPUTE_HEAT_MAP:
                # The softmax is not stored to save space
                # that's why we need to run the deep learning inference if any prediction is missing
                skip_inference = True
                for p in pred_dict.values():
                    if not os.path.exists(p):
                        skip_inference = False
                        break
            if skip_inference:
                print('Skip inference for %s.\nThe predictions already exist.' % f_n)
            else:
                print('\nStart inference for case %s' % f_n)

                # Load the img and mask
                img_nii = nib.load(input_path)
                img = img_nii.get_fdata().astype(np.float32)
                mask_nii = nib.load(mask_path)
                mask = mask_nii.get_fdata().astype(np.uint8)

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
                # Save folder for the intermediate results of the atlas-based segmentation
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
                save_seg(pred_atlas, affine=img_nii.affine, save_path=pred_dict['atlas'])

                # CNNs inference
                for deep_model in DEEP_LEARNING_MODELS:
                    out_folder = os.path.split(pred_dict[deep_model])[0]
                    if deep_model == 'nnUNetSegPrior':
                        task = TASK_NAME['235']
                        trainer = 'nnUNetTrainerV2_Seg_Prior'
                        seg_prior_path = './tmp_%s/seg_prior.nii.gz' % task
                        if not os.path.exists('tmp_%s' % task):
                            os.mkdir('tmp_%s' % task)
                        save_seg(pred_proba_atlas, affine=img_nii.affine, save_path=seg_prior_path)
                        cmd_options = '--input %s --mask %s --output_folder %s --fold all --task %s --trainer %s --save_npz' % \
                            (input_path, mask_path, out_folder, task, trainer)
                        if os.path.exists(seg_prior_path):
                            cmd_options += ' --seg_prior %s' % seg_prior_path
                        cmd = 'python %s/src/deep_learning/inference_nnunet.py %s' % (REPO_PATH, cmd_options)
                        print(cmd)
                        os.system(cmd)
                    elif deep_model == 'nnUNet':
                        task = TASK_NAME['225']
                        trainer = 'nnUNetTrainerV2'
                        cmd_options = '--input %s --mask %s --output_folder %s --fold all --task %s --trainer %s --save_npz' % \
                            (input_path, mask_path, out_folder, task, trainer)
                        cmd = 'python %s/src/deep_learning/inference_nnunet.py %s' % (REPO_PATH, cmd_options)
                        print(cmd)
                        os.system(cmd)
                    elif deep_model == 'SwinUNETR':
                        softmax = nib.load(pred_softmax_path[deep_model]).get_fdata().astype(np.float32)
                        pred_seg_swin = np.argmax(softmax, axis=3).astype(np.uint8)
                        save_seg(
                            pred_seg_swin,
                            affine=img_nii.affine,
                            save_path=pred_dict[deep_model],
                        )
                    else:
                        raise ValueError('Unknown deep learning model %s' % deep_model)

                # Transpose the atlas proba to match PyTorch convention
                pred_proba_atlas = np.transpose(pred_proba_atlas, axes=(3, 0, 1, 2))

                # Compute the TWAI predictions
                for deep_model in DEEP_LEARNING_MODELS:
                    # Load the deep learning softmax prediction
                    if deep_model == 'SwinUNETR':
                        softmax = nib.load(pred_softmax_path[deep_model]).get_fdata().astype(np.float32)
                        softmax = np.transpose(softmax, axes=(3, 0, 1, 2))
                    else:
                        softmax = load_softmax(pred_softmax_path[deep_model], volume_info_path[deep_model])

                    # ADDITIVE FUSION
                    pred_proba_add_fusion = 0.5 * (softmax + pred_proba_atlas)
                    pred_add_fusion = np.argmax(pred_proba_add_fusion, axis=0).astype(np.uint8)
                    save_seg(
                        pred_add_fusion,
                        affine=img_nii.affine,
                        save_path=pred_dict['%s_add_fusion' % deep_model],
                    )

                    # MULTIPLICATIVE FUSION
                    pred_proba_mult_fusion = softmax * pred_proba_atlas
                    pred_proba_mult_fusion += 0.001
                    # Normalize the probability
                    pred_proba_mult_fusion[:, ...] /= np.sum(pred_proba_mult_fusion, axis=0)
                    pred_mult_fusion = np.argmax(pred_proba_mult_fusion, axis=0).astype(np.uint8)
                    save_seg(
                        pred_mult_fusion,
                        affine=img_nii.affine,
                        save_path=pred_dict['%s_mult_fusion' % deep_model],
                    )

                    apply_TWAI(
                        ai=softmax,
                        fallback=pred_proba_atlas,
                        atlas_seg=pred_atlas,
                        cond=cond,
                        img=img,
                        mask_foreground=mask,
                        affine=img_nii.affine,
                        save_path=pred_dict['%s_trustworthy' % deep_model],
                        save_path_intensity_only=pred_dict['%s_trustworthy_intensity_only' % deep_model],
                        save_path_atlas_only=pred_dict['%s_trustworthy_atlas_only' % deep_model],
                    )

                    apply_TWAI(
                        ai=pred_proba_add_fusion,
                        fallback=pred_proba_atlas,
                        atlas_seg=pred_atlas,
                        cond=cond,
                        img=img,
                        mask_foreground=mask,
                        affine=img_nii.affine,
                        save_path=pred_dict['%s_add_fusion_trustworthy' % deep_model],
                        save_path_intensity_only=pred_dict['%s_add_fusion_trustworthy_intensity_only' % deep_model],
                        save_path_atlas_only=pred_dict['%s_add_fusion_trustworthy_atlas_only' % deep_model],
                    )

                    apply_TWAI(
                        ai=pred_proba_add_fusion,
                        fallback=pred_proba_atlas,
                        atlas_seg=pred_atlas,
                        cond=cond,
                        img=img,
                        mask_foreground=mask,
                        affine=img_nii.affine,
                        save_path=pred_dict['%s_mult_fusion_trustworthy' % deep_model],
                        save_path_intensity_only=pred_dict['%s_mult_fusion_trustworthy_intensity_only' % deep_model],
                        save_path_atlas_only=pred_dict['%s_mult_fusion_trustworthy_atlas_only' % deep_model],
                    )


                    # Clean folder
                    if 'nnUNet' in deep_model:
                        if os.path.exists(pred_softmax_path[deep_model]):  # Remove the npz file (it takes a lot of space)
                            os.system('rm %s' % pred_softmax_path[deep_model])
                        if os.path.exists(volume_info_path[deep_model]):  # Delete the pkl file
                            os.system('rm %s' % volume_info_path[deep_model])

            # Evaluation
            if INFERENCE_ONLY:
                continue
            gt_seg_path = os.path.join(dataset, f_n, 'parcellation.nii.gz')
            # for method in METHOD_NAMES:
            for method in list(pred_dict.keys()):
                dice, haus = compute_evaluation_metrics(pred_dict[method], gt_seg_path, dataset_path=dataset)
                for roi in DATASET_LABELS[dataset]:
                    if not roi in ALL_ROI:
                        continue
                    line = [patid, ga_ori, cond, center_val, method, roi, dice[roi], haus[roi]]
                    metric_data.append(line)

    # Save and print the metrics aggregated
    if not INFERENCE_ONLY:
        df = pd.DataFrame(metric_data, columns=METRICS_COLUMN)
        csv_path = os.path.join(pred_folder, 'metrics_all.csv')
        df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    t_start = time()
    if not os.path.exists(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)
    main(DATA_DIR)
    total_time = int(time() - t_start)
    print('\nTotal time=%dmin%dsec' % (total_time // 60, total_time % 60))
