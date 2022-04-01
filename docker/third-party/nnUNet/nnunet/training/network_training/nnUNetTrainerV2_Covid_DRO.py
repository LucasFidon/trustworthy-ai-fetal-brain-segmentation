from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.training.data_augmentation.default_data_augmentation import get_moreDA_augmentation
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
from torch.nn.utils import clip_grad_norm_
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.network_training.nnUNetTrainerV2_DRO import nnUNetTrainerV2_DRO
from nnunet.training.loss_functions.generalized_wasserstein_dice_loss import GeneralizedWassersteinDiceLossCovid
# from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D, unpack_dataset
# from nnunet.training.dataloading.dro_dataset_loading import HardnessWeightedDataLoader3D
# from nnunet.training.dataloading.dro.unreduced_loss_function import UnreducedLossFunction

try:
    from apex import amp
except ImportError:
    amp = None


LABELS = {
    # Basic lesion classes
    'ggo': 1,
    'consolidation': 2,
    'crazy_paving_pattern': 3,
    'linear_opacity': 2,
    # Super classes
    'combined_pattern': 4,
    'reversed_halo_sign': 4,
    'other_abnormal_tissue': 5,
    'lungs': 6,
    'background': 0,
}
# Bounds for the importance weights used for DRO
CLIP_IMPORTANCE_WEIGHTS = [0.1, 10.]


class nnUNetTrainerV2_Covid_DRO(nnUNetTrainerV2_DRO):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        print('Use the Generalized Wasserstein Dice Loss')
        self.loss = GeneralizedWassersteinDiceLossCovid()

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        keys = data_dict['keys']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        output = self.network(data)
        masked_output = []
        for y in output:
            # Set combined pattern prediction proba to 0
            mask_comb = torch.zeros_like(y)
            mask_comb[:, LABELS['combined_pattern'], :, :, :] = 1.
            if torch.cuda.is_available():
                mask_comb = to_cuda(mask_comb)
            y = (1. - mask_comb) * y - 1000. * mask_comb
            # Set other abnormal tissue prediction proba to 0
            mask_other_abn = torch.zeros_like(y)
            mask_other_abn[:, LABELS['other_abnormal_tissue'], :, :, :] = 1.
            if torch.cuda.is_available():
                mask_other_abn = to_cuda(mask_other_abn)
            y = (1. - mask_other_abn) * y - 1000. * mask_other_abn
            masked_output.append(y)
        masked_output = tuple(masked_output)

        del data
        loss = self.loss(masked_output, target)

        if run_online_evaluation:
            self.run_online_evaluation(masked_output, target)
        del target

        # DRO stuff
        if not run_online_evaluation:  # only for training
            loss_batch_no_grad = self.loss.loss.double().cpu().detach()
            # If importance sampling is used the loss is rescaled
            # as a function of the observed deprecation between the actual loss
            # and the current sampler weights for the current batch.
            if self.importance_sampling:
                # Get the importance sampling weights for the current batch:
                # scale = beta * (1 - momentum) * (observed_loss - stale_loss)
                scale_cpu = data_generator.generator.get_importance_sampling_weights(
                    loss_batch_no_grad, keys)
                # Clip the importance sampling weights:
                # the min clip assures we make some use of all the samples.
                # the max clip avoids exploding gradient.
                # Sampling without replacement also has similar effects
                # and avoid exploding gradient caused by batch filled
                # with only one sample
                scale_cpu = torch.clamp(
                    scale_cpu,
                    min=CLIP_IMPORTANCE_WEIGHTS[0],
                    max=CLIP_IMPORTANCE_WEIGHTS[1],
                )
                # Move the scaling factors to gpu.
                scale = scale_cpu.to(torch.device("cuda"))
                # Rescale the loss (it will appear in the loss log)
                loss = (scale * self.loss.loss).mean()

            # Update the sampling weights/loss history for the samples in the
            # current batch.
            data_generator.generator.update_weights(loss_batch_no_grad, keys)

        if do_backprop:
            if not self.fp16 or amp is None or not torch.cuda.is_available():
                loss.backward()
            else:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            _ = clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return loss.detach().cpu().numpy()