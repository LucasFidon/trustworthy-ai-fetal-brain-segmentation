from typing import Tuple, List
import numpy as np
import torch
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch import nn
from torch.nn.utils import clip_grad_norm_
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.training.loss_functions.dice_loss_covid import DC_and_CE_loss
try:
    from apex import amp
except ImportError:
    amp = None

LABELS = {
    # Basic lesion classes
    'lung': 1,
    'lesion': 2,
    'unknown': 3,
    'background': 0,
}

class nnUNetTrainerV2_CovidChallengeDiceCE(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        print('The masked Dice Loss + CE is used')
        dl_params = {}
        ce_params = {}
        self.loss = DC_and_CE_loss(soft_dice_kwargs=dl_params, ce_kwargs=ce_params)

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

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        output = self.network(data)
        masked_output = []
        for y in output:
            # Set the unknown score to a very low value
            # The CE will not propagate gradient for 'unknown' voxels
            mask = torch.zeros_like(y)
            mask[:, LABELS['unknown'], :, :, :] = 1.
            if torch.cuda.is_available():
                mask = to_cuda(mask)
            y = (1. - mask) * y - 1000. * mask
            masked_output.append(y)
        masked_output = tuple(masked_output)

        del data
        loss = self.loss(masked_output, target)

        if run_online_evaluation:
            self.run_online_evaluation(masked_output, target)
        del target

        if do_backprop:
            if not self.fp16 or amp is None or not torch.cuda.is_available():
                loss.backward()
            else:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            _ = clip_grad_norm_(self.network.parameters(), 10)
            self.optimizer.step()

        return loss.detach().cpu().numpy()

    # def preprocess_predict_nifti(self, input_files, output_file=None, softmax_ouput_file=None):
    #     """
    #     Use this to predict new data
    #     :param input_files:
    #     :param output_file:
    #     :param softmax_ouput_file:
    #     :return:
    #     """
    #     print("preprocessing...")
    #     d, s, properties = self.preprocess_patient(input_files)
    #     print("predicting...")
    #     pred = self.predict_preprocessed_data_return_seg_and_softmax(d, self.data_aug_params["do_mirror"],
    #                                                                  self.data_aug_params['mirror_axes'], True, 0.5,
    #                                                                  True, 'constant', {'constant_values': 0},
    #                                                                  self.patch_size, True)[1]
    #     print("set 'unknown' proba to 0")
    #     #TODO not sure about the dimensions here
    #     mask = torch.zeros_like(pred)
    #     mask[LABELS['unknown'], :, :, :] = 1.
    #     if torch.cuda.is_available():
    #         mask = to_cuda(mask)
    #     pred = (1. - mask) * pred - 1000. * mask
    #
    #     pred = pred.transpose([0] + [i + 1 for i in self.transpose_backward])
    #
    #     print("resampling to original spacing and nifti export...")
    #     save_segmentation_nifti_from_softmax(pred, output_file, properties, 3, None, None, None, softmax_ouput_file,
    #                                          None)
    #     print("done")

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None, use_sliding_window: bool = True,
                                                         step_size: float = 0.5, use_gaussian: bool = True,
                                                         pad_border_mode: str = 'constant', pad_kwargs: dict = None,
                                                         all_in_gpu: bool = True,
                                                         verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wrapper for masking the unknown prediction
        """
        # Compute the class proba map (this is the prediction after the non-linearity)
        pred = super().predict_preprocessed_data_return_seg_and_softmax(
            data, do_mirroring, mirror_axes, use_sliding_window, step_size, use_gaussian,
            pad_border_mode, pad_kwargs, all_in_gpu, verbose)
        #TODO not sure about the dimensions here but should be c,x,y,z according to predict_3D in neural_network.py
        mask = torch.zeros_like(pred)
        mask[LABELS['unknown'], :, :, :] = 1.
        if torch.cuda.is_available():
            mask = to_cuda(mask)
        #TODO: we are setting the unknown proba to -1000... it works but that is ugly
        pred = (1. - mask) * pred - 1000. * mask
        return pred
