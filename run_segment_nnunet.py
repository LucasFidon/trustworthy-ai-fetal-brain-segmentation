import os
from argparse import ArgumentParser
from src.utils.definitions import *
from run_infer_eval import apply_bias_field_corrections

parser = ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--mask', type=str)
parser.add_argument('--output_folder', type=str)
parser.add_argument('--bfc', action='store_true', help='Bias Field Correction')


def main(args):
    input_path = args.input
    mask_path = args.mask
    output_path = args.output_folder

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Bias field correction (optional)
    if args.bfc:
        pre_input_path = os.path.join(output_path, 'srr_preprocessed.nii.gz')
        apply_bias_field_corrections(input_path, mask_path, pre_input_path)
        input_path = pre_input_path

    # nnUNet prediction
    cmd_options = '--input %s --mask %s --output_folder %s --fold all --task Task225_FetalBrain3dTrust --save_npz' % \
        (input_path, mask_path, output_path)
    cmd = 'python %s/src/deep_learning/inference_nnunet.py %s' % (REPO_PATH, cmd_options)
    os.system(cmd)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
