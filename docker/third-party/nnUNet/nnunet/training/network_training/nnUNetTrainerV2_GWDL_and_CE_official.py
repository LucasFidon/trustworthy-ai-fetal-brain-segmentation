import torch
from torch import nn
from generalized_wasserstein_dice_loss.loss import GeneralizedWassersteinDiceLoss
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.loss_functions.ND_Crossentropy import CrossentropyND
from nnunet.training.network_training.nnUNetTrainerV2_GWDL import DIST_MATRIX


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
        gwdl_loss = self.gwdl(net_output, target) if self.weight_gwdl != 0 else 0
        ce_loss = self.ce(net_output, target) if self.weight_ce != 0 else 0
        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_gwdl * gwdl_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

    @property
    def reduction(self):
        return self.ce.reduction  # should be the same as self.dc.reduction

    @reduction.setter
    def reduction(self, reduction):
        self.ce.reduction = reduction
        self.gwdl.reduction = reduction


class nnUNetTrainerV2_GWDL_and_CE_official(nnUNetTrainerV2):
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
