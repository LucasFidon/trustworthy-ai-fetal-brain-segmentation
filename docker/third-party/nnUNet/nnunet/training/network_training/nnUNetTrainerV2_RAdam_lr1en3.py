from nnunet.training.network_training.nnUNetTrainerV2_RAdam import nnUNetTrainerV2_RAdam

class nnUNetTrainerV2_RAdam_lr1en3(nnUNetTrainerV2_RAdam):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.initial_lr = 1e-3  # default is 0.01
