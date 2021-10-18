import os
from src.utils.definitions import NIFTYREG_PATH


def compute_def_from_cpp(cpp_path, ref_path, save_def_path):
    cmd = '%s/reg_transform -ref %s -def %s %s' % (NIFTYREG_PATH, ref_path, cpp_path, save_def_path)
    os.system(cmd)
