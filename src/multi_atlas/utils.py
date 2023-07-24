import os
import numpy as np
import nibabel as nib
from loguru import logger
from src.utils.definitions import NIFTYREG_PATH, ATLAS_SB, ATLAS_CONTROL_HARVARD, ATLAS_CONTROL_CHINESE, CONDITIONS


def compute_disp_from_cpp(cpp_path, ref_path, save_disp_path):
    save_folder = os.path.split(save_disp_path)[0]

    # Convert the cpp into a deformation field
    save_def_path = os.path.join(save_folder, 'tmp_def.nii.gz')
    cmd = '%s/reg_transform -ref %s -def %s %s > /dev/null' % (NIFTYREG_PATH, ref_path, cpp_path, save_def_path)
    os.system(cmd)

    # Create the identity transformation to get the displacement
    cpp_id = os.path.join(save_folder, 'output_cpp_identity.nii.gz')
    res_id = os.path.join(save_folder, 'srr_identity.nii.gz')
    cmd = 'reg_f3d -ref %s -flo %s -res %s -cpp %s -be 1. -le 0. -ln 3 -voff' % \
          (ref_path, ref_path, res_id, cpp_id)
    os.system(cmd)
    save_id_path = os.path.join(save_folder, 'tmp_id_def.nii.gz')
    cmd = '%s/reg_transform -ref %s -def %s %s > /dev/null' % (NIFTYREG_PATH, ref_path, cpp_id, save_id_path)
    os.system(cmd)

    # Substract the identity to get the displacement field
    deformation_nii = nib.load(save_def_path)
    deformation = deformation_nii.get_fdata().astype(np.float32)
    identity = nib.load(save_id_path).get_fdata().astype(np.float32)
    disp = deformation - identity
    disp_nii = nib.Nifti1Image(disp, deformation_nii.affine)
    nib.save(disp_nii, save_disp_path)


def get_atlas_list(ga, condition, ga_delta_max=1):
    assert condition in CONDITIONS, \
        'Found %s but only %s are supported' % (condition, str(CONDITIONS))

    atlas_list = []
    for ga_shift in range(-ga_delta_max, ga_delta_max+1):
        if condition == 'Spina Bifida':
            if ga + ga_shift <= 25:
                template_path_notop = os.path.join(
                    ATLAS_SB,
                    'fetal_SB_atlas_GA%d_notoperated' % (ga + ga_shift),
                )
                if os.path.exists(template_path_notop):
                    atlas_list.append(template_path_notop)
                else:
                    logger.warning('%s not found.' % template_path_notop)
            if ga + ga_shift >= 25:
                template_path_op = os.path.join(
                    ATLAS_SB,
                    'fetal_SB_atlas_GA%d_operated' % (ga + ga_shift),
                )
                if os.path.exists(template_path_op):
                    atlas_list.append(template_path_op)
                else:
                    logger.warning('%s not found.' % template_path_op)

        elif condition == 'Neurotypical':
            template_harvard_path = os.path.join(
                ATLAS_CONTROL_HARVARD,
                'HarvardSTA%d_Study1' % (ga + ga_shift),
            )
            if os.path.exists(template_harvard_path):
                atlas_list.append(template_harvard_path)
            else:
                logger.warning('%s not found.' % template_harvard_path)
            template_chinese_path = os.path.join(
                ATLAS_CONTROL_CHINESE,
                'Chinese%d_Study1' % (ga + ga_shift),
            )
            if os.path.exists(template_chinese_path):
                atlas_list.append(template_chinese_path)
            else:
                logger.warning('%s not found.' % template_chinese_path)

    return atlas_list
