import os
from src.utils.definitions import NIFTYREG_PATH, ATLAS_SB, ATLAS_CONTROL_HARVARD, ATLAS_CONTROL_CHINESE, CONDITIONS


def compute_def_from_cpp(cpp_path, ref_path, save_def_path):
    cmd = '%s/reg_transform -ref %s -def %s %s > /dev/null' % (NIFTYREG_PATH, ref_path, cpp_path, save_def_path)
    os.system(cmd)


def get_atlas_list(ga, condition, ga_delta_max=1):
    assert condition in CONDITIONS, \
        'Found %s but only %s are supported' % (condition, str(CONDITIONS))

    atlas_list = []
    for ga_shift in range(-ga_delta_max, ga_delta_max+1):
        if condition == 'Spina Bifida' or condition == 'Pathological':
            template_path_notop = os.path.join(
                ATLAS_SB,
                'fetal_SB_atlas_GA%d_notoperated' % (ga + ga_shift),
            )
            if os.path.exists(template_path_notop):
                atlas_list.append(template_path_notop)
            template_path_op = os.path.join(
                ATLAS_SB,
                'fetal_SB_atlas_GA%d_operated' % (ga + ga_shift),
            )
            if os.path.exists(template_path_op):
                atlas_list.append(template_path_op)

        if condition == 'Neurotypical' or condition == 'Pathological':  # Control / Neurotypical
            template_harvard_path = os.path.join(
                ATLAS_CONTROL_HARVARD,
                'HarvardSTA%d_Study1' % (ga + ga_shift),
            )
            if os.path.exists(template_harvard_path):
                atlas_list.append(template_harvard_path)
            template_chinese_path = os.path.join(
                ATLAS_CONTROL_CHINESE,
                'Chinese%d_Study1' % (ga + ga_shift),
            )
            if os.path.exists(template_chinese_path):
                atlas_list.append(template_chinese_path)

    return atlas_list
