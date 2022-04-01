import torch
from nnunet.training.network_training.nnUNetTrainerV2_DRO import nnUNetTrainerV2_DRO


class nnUNetTrainerV2Adam_DRO(nnUNetTrainerV2_DRO):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        print("Use Adam")
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            amsgrad=False
        )
        self.lr_scheduler = None
