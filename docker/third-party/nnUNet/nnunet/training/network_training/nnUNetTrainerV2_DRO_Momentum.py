from nnunet.training.network_training.nnUNetTrainerV2_DRO import nnUNetTrainerV2_DRO


class nnUNetTrainerV2_DRO_Momentum(nnUNetTrainerV2_DRO):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                         unpack_data, deterministic, fp16)
        self.dro_momentum = 0.5
        print('Use DRO momentum=%f' % self.dro_momentum)
