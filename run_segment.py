import os
from argparse import ArgumentParser
from loguru import logger
import nibabel as nib
from src.utils.definitions import *
from run_infer_eval_all import apply_bias_field_corrections
from src.deep_learning.inference_nnunet import load_softmax
from src.multi_atlas.inference import multi_atlas_segmentation
from src.multi_atlas.utils import get_atlas_list
from src.segmentations_fusion.dempster_shaffer import merge_deep_and_atlas_seg, dempster_add_intensity_prior

SUPPORTED_CONDITIONS = CONDITIONS
MERGING_MULTI_ATLAS = 'GIF'

parser = ArgumentParser(description='Segment a fetal brain 3D MRI for the backbone AI, '
                                    'the fallback, and the Trustworthy AI algorithms.')
parser.add_argument('--input', type=str,
                    help='Path to the fetal brain 3D MRI to be segmented.')
parser.add_argument('--mask', type=str,
                    help='Path to the 3D brain mask of the fetal brain 3D MRI to be segmented.')
parser.add_argument('--output_folder', type=str,
                    help='Path of the folder where the output will be saved.')
parser.add_argument('--ga', type=float,
                    help='Gestational age at the time of acquisition of the fetal brain 3D MRI to be segmented.')
parser.add_argument('--condition', type=str,
                    help='Brain condition of the fetal brain 3D MRI to be segmented. '
                         'Must be one of %s.' % str(SUPPORTED_CONDITIONS))
parser.add_argument('--bfc', action='store_true',
                    help='Allow to use intensity bias field correction. '
                         'Recommended if no correction for intensity bias field correction has been '
                         'performed before.')
parser.add_argument('--force_rerun', action='store_true',
                    help='Force to re-run all the segmentations including all the image registrations required'
                         ' for the fallback. By default segmentations are re-used if they exist already')


def _get_atlas_volumes_path_list(condition, ga):
    if condition == 'Pathological':
        atlas_list = get_atlas_list(ga=ga, condition='Neurotypical', ga_delta_max=DELTA_GA_CONTROL)
        atlas_list += get_atlas_list(ga=ga, condition='Spina Bifida', ga_delta_max=DELTA_GA_SPINA_BIFIDA)
    elif condition == 'Neurotypical':
        atlas_list = get_atlas_list(ga=ga, condition='Neurotypical', ga_delta_max=DELTA_GA_CONTROL)
    else:
        assert condition == 'Spina Bifida', 'Unknown condition %s' % condition
        atlas_list = get_atlas_list(ga=ga, condition='Spina Bifida', ga_delta_max=DELTA_GA_SPINA_BIFIDA)
    return atlas_list

def _preproces_GA(ga):
    out = int(round(ga))
    if out > MAX_GA:
        out = MAX_GA
    if out < MIN_GA:
        out = MIN_GA
    return out


def main(args):
    input_path = args.input
    mask_path = args.mask
    output_path = args.output_folder
    logger.info('Input 3D T2w-MRI: %s' % input_path)
    logger.info('Input 3D brain mask: %s' % mask_path)
    logger.info('Output folder: %s' % output_path)

    # GA is rounded to the closest week and clipped to the range of GA of the atlases
    ga = _preproces_GA(args.ga)
    logger.info('Use GA=%d weeks (rounded to the closest integer in [%d, %d])' % (ga, MIN_GA, MAX_GA))
    cond = args.condition
    logger.info('Condition: %s' % cond)

    assert cond in SUPPORTED_CONDITIONS, \
        '--condition argument must be in %s.\nFound %s' % (str(SUPPORTED_CONDITIONS), cond)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Bias field correction (optional)
    if args.bfc:
        logger.warning('Intensity bias field correction will be applied to the input 3D MRI prior to segmentation')
        pre_input_path = os.path.join(output_path, 'srr_preprocessed.nii.gz')
        apply_bias_field_corrections(input_path, mask_path, pre_input_path)
        input_path = pre_input_path

    logger.info('Start inference...')

    # Backbone AI inference
    output_backboneAI_path = os.path.join(output_path, 'backboneAI')
    cmd_options = '--input %s --mask %s --output_folder %s --fold all --task Task225_FetalBrain3dTrust --save_npz' % \
        (input_path, mask_path, output_backboneAI_path)
    cmd = 'python %s/src/deep_learning/inference_nnunet.py %s' % (REPO_PATH, cmd_options)
    logger.info('Backbone AI inference:\n%s' % cmd)
    os.system(cmd)

     # Load the softmax prediction, img and mask
    img_nii = nib.load(input_path)
    img = img_nii.get_fdata().astype(np.float32)
    mask_nii = nib.load(mask_path)
    mask = mask_nii.get_fdata().astype(np.uint8)
    f_n = os.path.split(output_path)[1]
    pred_softmax_path = os.path.join(output_backboneAI_path, '%s.npz' % f_n)
    volume_info_path = os.path.join(output_backboneAI_path, '%s.pkl' % f_n)  # info about the volume and preprocessing done by nnUNet
    softmax = load_softmax(pred_softmax_path, volume_info_path)

    # Fallback inference
    # Propagate the atlas volumes segmentation
    atlas_list = _get_atlas_volumes_path_list(cond, ga)
    logger.info('Fallback inference')
    logger.info('Atlas propagation will be computed for the volumes:')
    for p in atlas_list:
        logger.info(p)
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
        reuse_existing_pred=not args.force_rerun,
        force_recompute_heat_kernels=False,
    )

    # Save the atlas-based prediction
    pred_atlas = np.argmax(pred_proba_atlas, axis=3).astype(np.uint8)
    pred_atlas_nii = nib.Nifti1Image(pred_atlas, img_nii.affine)
    output_fallback_path = os.path.join(output_path, 'fallback')
    if not os.path.exists(output_fallback_path):
        os.mkdir(output_fallback_path)
    pred_atlas_path = os.path.join(output_fallback_path, '%s.nii.gz' % f_n)
    nib.save(pred_atlas_nii, pred_atlas_path)

    # Transpose the atlas proba to match PyTorch convention
    pred_proba_atlas = np.transpose(pred_proba_atlas, axes=(3, 0, 1, 2))

    # Take the average of the CNN and atlas predicted proba as backbone AI
    pred_proba_trustworthy = 0.5 * (softmax + pred_proba_atlas)

    # Apply Dempster's rule with the atlas prior
    pred_proba_trustworthy = merge_deep_and_atlas_seg(
        deep_proba=pred_proba_trustworthy,
        atlas_seg=pred_atlas,
        condition=cond,  # Used to know which margins to use
    )

    # Apply Dempster's rule with the intensity prior
    pred_proba_trustworthy = dempster_add_intensity_prior(
        deep_proba=pred_proba_trustworthy,
        image=img,
        mask=mask,
        denoise=False,
    )
    # Save the trustworthy prediction
    pred_trustworthy = np.argmax(pred_proba_trustworthy, axis=0).astype(np.uint8)
    pred_trustworthy_nii = nib.Nifti1Image(pred_trustworthy, img_nii.affine)
    output_twai_path = os.path.join(output_path, 'trustworthyAI')
    if not os.path.exists(output_twai_path):
        os.mkdir(output_twai_path)
    pred_trustworthy_path = os.path.join(output_twai_path, '%s.nii.gz' % f_n)
    nib.save(pred_trustworthy_nii, pred_trustworthy_path)

    logger.success('Backbone AI (nnU-Net-DRO) segmentation has been saved in %s' % output_backboneAI_path)
    logger.success('Fallback segmentation has been saved in %s' % output_fallback_path)
    logger.success('Trustworthy AI segmentation has been saved in %s' % output_twai_path)

    # Clean folder
    if os.path.exists(pred_softmax_path):  # Remove the npz file (it takes a lot of space)
        os.system('rm %s' % pred_softmax_path)
    if os.path.exists(volume_info_path):  # Delete the pkl file
        os.system('rm %s' % volume_info_path)


if __name__ == '__main__':
    paper_ref = 'L.Fidon et al. "A Dempster-Shafer approach to trustworthy AI with application to fetal brain MRI segmentation". arXiv preprint arXiv:2204.02779 (2022).'
    git_repo = "https://github.com/LucasFidon/trustworthy-ai-fetal-brain-segmentation"
    logger.success("Please cite the following paper when using this code:\n%s" % paper_ref)
    logger.success("If you have questions or suggestions, please open an issue at %s" % git_repo)
    logger.success("Have a good day!")
    args = parser.parse_args()
    main(args)
