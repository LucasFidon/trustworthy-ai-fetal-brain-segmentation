# import torch
import numpy as np
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
# from generalized_wasserstein_dice_loss.loss import GeneralizedWassersteinDiceLoss
from nnunet.training.loss_functions.generalized_wasserstein_dice_loss import GeneralizedWassersteinDiceLoss

# OLD from NiftyNet (warning: different class order!)
# 2: Edema
# 1: Non-enhancing Tumor (NET)
# 3: Enhancing Tumor (ET)
# dist(ET, NET) = 0.5
# dist(edema, NET) = 0.6
# dist(edema, ET) = 0.7
# DIST_MATRIX = np.array(
#     [[0.0, 1.0, 1.0, 1.0],
#     [1.0, 0.0, 0.6, 0.5],
#     [1.0, 0.6, 0.0, 0.7],
#     [1.0, 0.5, 0.7, 0.0]],
#     dtype=np.float64
# )

# Order of the classes must be:
# 1: Edema
# 2: Non-enhancing Tumor (NET)
# 3: Enhancing Tumor (ET)
# DIST_MATRIX = np.array(
#     [[0.0, 1.0, 1.0, 1.0],
#     [1.0, 0.0, 0.75, 0.75],
#     [1.0, 0.75, 0.0, 0.5],
#     [1.0, 0.75, 0.5, 0.0]],
#     dtype=np.float64
# )
DIST_MATRIX = np.array(
   [[0.0, 1.0, 1.0, 1.0],
    [1.0, 0.0, 0.6, 0.7],
    [1.0, 0.6, 0.0, 0.5],
    [1.0, 0.7, 0.5, 0.0]],
    dtype=np.float64
)


class nnUNetTrainerV2_GWDL(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        print('The Generalized Wasserstein Dice Loss is used')
        self.loss = GeneralizedWassersteinDiceLoss(dist_matrix=DIST_MATRIX, alpha_mode='equal')