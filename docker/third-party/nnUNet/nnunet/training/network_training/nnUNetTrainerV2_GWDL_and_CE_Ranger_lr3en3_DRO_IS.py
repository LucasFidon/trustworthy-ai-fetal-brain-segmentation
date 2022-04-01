from nnunet.training.network_training.nnUNetTrainerV2_DRO_IS import nnUNetTrainerV2_DRO_IS
from nnunet.training.loss_functions.dice_loss import GWDL_and_CE_loss
from nnunet.training.network_training.nnUNetTrainerV2_GWDL import DIST_MATRIX
from nnunet.training.optimizer.ranger import Ranger


class nnUNetTrainerV2_GWDL_and_CE_Ranger_lr3en3_DRO_IS(nnUNetTrainerV2_DRO_IS):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        print('The Generalized Wasserstein Dice Loss + CE is used')
        gwdl_params = {'dist_matrix': DIST_MATRIX}
        ce_params = {}
        self.loss = GWDL_and_CE_loss(gwdl_kwargs=gwdl_params, ce_kwargs=ce_params)
        self.initial_lr = 3e-3

    def initialize_optimizer_and_scheduler(self):
        self.optimizer = Ranger(self.network.parameters(), self.initial_lr, k=6, N_sma_threshhold=5,
                                weight_decay=self.weight_decay)
        self.lr_scheduler = None

