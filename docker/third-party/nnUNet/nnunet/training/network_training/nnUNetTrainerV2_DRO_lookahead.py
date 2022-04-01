import torch
from nnunet.training.network_training.nnUNetTrainerV2_DRO import nnUNetTrainerV2_DRO
from nnunet.training.optimizer.lookahead import Lookahead


class nnUNetTrainerV2_DRO_lookahead(nnUNetTrainerV2_DRO):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                         unpack_data, deterministic, fp16)

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        # add lookahead
        self.optimizer = Lookahead(self.optimizer, la_steps=5, la_alpha=0.5)
        self.lr_scheduler = None
