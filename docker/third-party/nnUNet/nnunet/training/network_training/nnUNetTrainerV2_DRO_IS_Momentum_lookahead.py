from nnunet.training.network_training.nnUNetTrainerV2_DRO import CLIP_IMPORTANCE_WEIGHTS
from nnunet.training.network_training.nnUNetTrainerV2_DRO_lookahead import nnUNetTrainerV2_DRO_lookahead


class nnUNetTrainerV2_DRO_IS_Momentum_lookahead(nnUNetTrainerV2_DRO_lookahead):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                         unpack_data, deterministic, fp16)
        # parameter for dro
        self.importance_sampling = True
        print('DRO importance sampling is used')
        print('Importance weights are clipped to [%f, %f].' %
            (CLIP_IMPORTANCE_WEIGHTS[0], CLIP_IMPORTANCE_WEIGHTS[1]))
        self.dro_momentum = 0.5
        print('Use DRO momentum=%f' % self.dro_momentum)