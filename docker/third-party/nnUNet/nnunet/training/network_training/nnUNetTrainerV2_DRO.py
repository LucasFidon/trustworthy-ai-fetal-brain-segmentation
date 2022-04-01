from collections import OrderedDict
from typing import Tuple
import os
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
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D, DataLoader2D, unpack_dataset
from nnunet.training.dataloading.dro_dataset_loading import HardnessWeightedDataLoader3D
from nnunet.training.dataloading.dro.unreduced_loss_function import UnreducedLossFunction


try:
    from apex import amp
except ImportError:
    amp = None

# Bounds for the importance weights used for DRO
CLIP_IMPORTANCE_WEIGHTS = [0.1, 10.]

class nnUNetTrainerV2_DRO(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None,
                 batch_dice=True, stage=None, unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                         unpack_data, deterministic, fp16)
        self.max_num_epochs = 1000
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None
        # parameter for dro (default)
        self.beta = 100
        self.init_weights = 2.
        self.importance_sampling = False
        self.dro_momentum = 0.  # no momentum for the update of the DRO weights

        self.pin_memory = True

    def get_basic_generators(self):
        self.load_dataset()
        self.do_split()

        if self.threeD:
            # Use data loader for DRO
            dl_tr = HardnessWeightedDataLoader3D(
                self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                False, oversample_foreground_percent=self.oversample_foreground_percent, pad_mode="constant",
                pad_sides=self.pad_all_sides, beta=self.beta, sampling_weights_init=self.init_weights,
                dro_momentum=self.dro_momentum,
            )
            dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides)
        else:
            print("DRO not implemented for 2D data sampling")
            dl_tr = DataLoader2D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                                 transpose=None,  # self.plans.get('transpose_forward'),
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 pad_mode="constant", pad_sides=self.pad_all_sides)
            dl_val = DataLoader2D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size,
                                  transpose=None,  # self.plans.get('transpose_forward'),
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  pad_mode="constant", pad_sides=self.pad_all_sides)
        return dl_tr, dl_val

    def initialize(self, training=True, force_load_plans=False):
        """
        Same as nnUNetTrainerV2:
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        New for DRO:
        - use a HardnessWeightedDataLoader3D as training data generator
        - wrap the loss function to make it 'unreduced' (get access to the batch of loss and not just the mean)

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            # Wrap the loss to be an UnreducedLoss for DRO
            self.loss = UnreducedLossFunction(self.loss, reduction='mean')

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                # Create the data loaders
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")
                # Add data augmentation
                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

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

        del data
        loss = self.loss(output, target)

        if run_online_evaluation:
            self.run_online_evaluation(output, target)
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

        # Update the parameters of the CNN
        if do_backprop:
            if not self.fp16 or amp is None or not torch.cuda.is_available():
                loss.backward()
            else:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            _ = clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return loss.detach().cpu().numpy()

    def save_checkpoint(self, fname, save_optimizer=True):
        super().save_checkpoint(fname, save_optimizer)
        # Save weights of the sampler
        dir_save, model_save_name = os.path.split(fname)
        weights_save_name = '%s_dro_weights.pt7' % model_save_name.split('.')[0]
        save_path_dro_weights = os.path.join(dir_save, weights_save_name)
        self.dl_tr.save_weights(save_path_dro_weights)

    def load_sampling_weights(self, weights_path):
        self.dl_tr.load_weights(weights_path)

    def load_latest_checkpoint(self, train=True):
        # Load latest model parameters
        super().load_latest_checkpoint(train)
        # Load the latest weights of the sampler
        if train:
            if isfile(join(self.output_folder, "model_final_checkpoint_dro_weights.pt7")):
                return self.load_sampling_weights(
                    join(self.output_folder, "model_final_checkpoint_dro_weights.pt7"))
            if isfile(join(self.output_folder, "model_latest_dro_weights.pt7")):
                return self.load_sampling_weights(
                    join(self.output_folder, "model_latest_dro_weights.pt7"))
            if isfile(join(self.output_folder, "model_best_dro_weights.pt7")):
                return self.load_sampling_weights(
                    join(self.output_folder, "model_best_dro_weights.pt7"))
            raise RuntimeError("No checkpoint for the sampling weights found")
