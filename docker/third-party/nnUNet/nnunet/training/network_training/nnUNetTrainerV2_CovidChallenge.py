# from typing import Tuple, List
import numpy as np
import torch
# from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch import nn
# from torch.nn.utils import clip_grad_norm_
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from generalized_wasserstein_dice_loss.loss import GeneralizedWassersteinDiceLoss
# from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
from nnunet.training.loss_functions.dice_loss_covid import CrossentropyND
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.network_architecture.generic_UNet import Generic_UNet, ConvDropoutNormNonlin
from nnunet.network_architecture.initialization import InitWeights_He
# from nnunet.training.loss_functions.ND_Crossentropy import CrossentropyND

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

DIST_MATRIX = np.array(
    [[0.0, 1.0, 1.0, 1.0],
     [1.0, 0.0, 0.9, 0.0],
     [1.0, 0.9, 0.0, 0.0],
     [1.0, 0.0, 0.0, 0.0]],
    dtype=np.float32
)


class GWDL_and_CE_loss(nn.Module):
    def __init__(self, gwdl_kwargs, ce_kwargs, aggregate="sum",
                 weight_ce=1, weight_gwdl=1, reduction='mean'):
        super(GWDL_and_CE_loss, self).__init__()
        self.weight_gwdl = weight_gwdl
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = CrossentropyND(**ce_kwargs)
        self.gwdl = GeneralizedWassersteinDiceLoss(**gwdl_kwargs)
        self.ce.reduction = reduction
        self.gwdl.reduction = reduction

    def forward(self, net_output, target):
        # mask voxels labeled 'unknown' for the CE
        mask = torch.ones_like(target).long()
        mask[target == LABELS['unknown']] = 0
        gwdl_loss = self.gwdl(net_output, target) if self.weight_gwdl != 0 else 0
        ce_loss = self.ce(net_output, target, loss_mask=mask) if self.weight_ce != 0 else 0
        result = self.weight_ce * ce_loss + self.weight_gwdl * gwdl_loss
        return result

    @property
    def reduction(self):
        return self.ce.reduction  # should be the same as self.dc.reduction

    @reduction.setter
    def reduction(self, reduction):
        self.ce.reduction = reduction
        self.gwdl.reduction = reduction

class Generic_UNetMasked(Generic_UNet):
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        super(Generic_UNetMasked, self).__init__(
            input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage,
                 feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs,
                 dropout_op, dropout_op_kwargs,
                 nonlin, nonlin_kwargs, deep_supervision, dropout_in_localization,
                 final_nonlin, weightInitializer, pool_op_kernel_sizes, conv_kernel_sizes,
                 upscale_logits, convolutional_pooling, convolutional_upsampling,
                 max_num_features, basic_block, seg_output_use_bias
            )

    def forward(self, x):
        skips = []
        seg_outputs = []
        # Encoder
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)
        # Decoder
        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            score = self.seg_outputs[u](x)
            # Mask the 'unknown' label
            score[:, LABELS['unknown'], :, :, :] = -1000
            seg_outputs.append(self.final_nonlin(score))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]


class nnUNetTrainerV2_CovidChallenge(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        print('The Generalized Wasserstein Dice Loss + CE is used')
        weighting_mode = 'default'
        print('Use the official implementation of the GWDL in mode %s' % weighting_mode)
        gwdl_params = {'dist_matrix': DIST_MATRIX, 'weighting_mode': weighting_mode}
        ce_params = {}
        self.loss = GWDL_and_CE_loss(gwdl_kwargs=gwdl_params, ce_kwargs=ce_params)

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNetMasked(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    # def initialize_network(self):
    #     self.num_classes = self.num_classes - 1  # remove the 'unknown' label
    #     super().initialize_network()


    # def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
    #     """
    #     gradient clipping improves training stability
    #
    #     :param data_generator:
    #     :param do_backprop:
    #     :param run_online_evaluation:
    #     :return:
    #     """
    #     data_dict = next(data_generator)
    #     data = data_dict['data']
    #     target = data_dict['target']
    #
    #     data = maybe_to_torch(data)
    #     target = maybe_to_torch(target)
    #
    #     if torch.cuda.is_available():
    #         data = to_cuda(data)
    #         target = to_cuda(target)
    #
    #     self.optimizer.zero_grad()
    #
    #     output = self.network(data)
    #     masked_output = []
    #     for y in output:
    #         # Set the unknown score to a very low value
    #         # The CE will not propagate gradient for 'unknown' voxels
    #         mask = torch.zeros_like(y)
    #         mask[:, LABELS['unknown'], :, :, :] = 1.
    #         # if torch.cuda.is_available():
    #         #     mask = to_cuda(mask)
    #         y = (1. - mask) * y - 1000. * mask
    #         masked_output.append(y)
    #     masked_output = tuple(masked_output)
    #
    #     del data
    #     loss = self.loss(masked_output, target)
    #
    #     if run_online_evaluation:
    #         self.run_online_evaluation(masked_output, target)
    #     del target
    #
    #     if do_backprop:
    #         if not self.fp16 or amp is None or not torch.cuda.is_available():
    #             loss.backward()
    #         else:
    #             with amp.scale_loss(loss, self.optimizer) as scaled_loss:
    #                 scaled_loss.backward()
    #         _ = clip_grad_norm_(self.network.parameters(), 10)
    #         self.optimizer.step()
    #
    #     return loss.detach().cpu().numpy()

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

    # def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
    #                                                      mirror_axes: Tuple[int] = None, use_sliding_window: bool = True,
    #                                                      step_size: float = 0.5, use_gaussian: bool = True,
    #                                                      pad_border_mode: str = 'constant', pad_kwargs: dict = None,
    #                                                      all_in_gpu: bool = True,
    #                                                      verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Wrapper for masking the unknown prediction
    #     """
    #     # Compute the class proba map (this is the prediction after the non-linearity)
    #     pred = super().predict_preprocessed_data_return_seg_and_softmax(
    #         data, do_mirroring, mirror_axes, use_sliding_window, step_size, use_gaussian,
    #         pad_border_mode, pad_kwargs, all_in_gpu, verbose)
    #     #TODO not sure about the dimensions here but should be c,x,y,z according to predict_3D in neural_network.py
    #     mask = torch.zeros_like(pred)
    #     mask[LABELS['unknown'], :, :, :] = 1.
    #     if torch.cuda.is_available():
    #         mask = to_cuda(mask)
    #     #TODO: we are setting the unknown proba to -1000... it works but that is ugly
    #     pred = (1. - mask) * pred - 1000. * mask
    #     return pred
