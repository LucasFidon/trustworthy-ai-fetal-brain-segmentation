from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.loss_functions.dice_loss import DL_and_Focal_loss


class nnUNetTrainerV2_DL_and_Focal(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        print('Dice Loss + Focal Loss is used')
        dl_params = {}
        focal_params = {}
        self.loss = DL_and_Focal_loss(dl_kwargs=dl_params, focal_kwargs=focal_params)
