"""
@brief  PyTorch code for Unreduced Loss Function.

        Usual loss functions in PyTorch returned one scalar loss value per batch
        (e.g. average or sum of the losses for the samples of the batch).
        However, we may want instead to get a batch of scalar loss values
        with one scalar loss value per sample.

        Unreduced Loss Functions still return the reduced loss (mean or sum) when called,
        but in addition they store internally an unreduced loss value for the last batch
        with one scalar value per sample.

        Unreduced loss functions are required for the Hardness Weighted Sampler
        (see loss_functions/unreduced_loss_function.py).

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
@date   30 Oct 2019.
"""

import torch
import torch.nn as nn
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2


# Should UnreducedLossFunction inherit from the nn.Module class of PyTorch?
class UnreducedLossFunction:
    def __init__(self, reduced_loss_func, reduction='mean'):
        """
        Wrapper for PyTorch loss functions that keeps the batch of loss values in memory
        instead of returning only a reduced version of the batch loss (e.g. mean or sum).
        :param reduced_loss_func: instance of nn.Module; a PyTorch loss function
        """
        assert isinstance(reduced_loss_func, nn.Module), \
            "reduced_loss_func must be an instance of nn.Module. " \
            "Found an instance of %s instead." % type(reduced_loss_func)
        # internal pytorch loss function
        self.func = reduced_loss_func
        # internal unreduced loss for the last batch
        self.loss = None
        assert reduction in ['mean', 'sum'], "Only 'mean' and 'sum' are supported " \
                                             "for the reduction parameter."
        self.reduction = reduction

    def cuda(self, device=None):
        """
        Moves all the internal loss function parameters and buffers to the GPU.
        :param device: int; (optional) if specified, all parameters will be
                copied to that device
        """
        self.func.cuda(device)

    def cpu(self):
        """
        Moves all the internal loss function parameters and buffers to the CPU.
        """
        self.func.cpu()

    def __post_init__(self):
        """
        Called once after the __init__ function.
        """
        # set the reduction parameter of the internal PyTorch loss function to
        # 'none'
        self.func.reduction = 'none'

    def __call__(self, input, target):
        """
        Return the reduced batch loss (mean or sum)
        and saved the batch loss in self.loss
        :param input: pytorch tensor
        :param target: pytorch tensor
        :return: pytorch scalar tensor
        """
        # change the reduction attribute of the internal loss function to none
        # to get a batch loss instead of a sum or a mean
        if isinstance(self.func, MultipleOutputLoss2):
            # Rq: would be better if MultipleOutputLoss2 would expose
            # the attributes of the loss wrapped for deep supervision,
            # but I prefer not to change the API of nnUNet.
            self.func.loss.reduction = 'none'
        else:
            self.func.reduction = 'none'

        # store the unreduced batch loss
        self.loss = self.func.forward(input, target)

        # make sure the loss is averaged over the pixel/voxel positions
        # self.loss should be a 1D tensor of size = batch size
        while len(self.loss.shape) > 1:
            # average over the last dimension
            self.loss = torch.mean(self.loss, -1)

        # then return the reduced loss
        if self.reduction == 'sum':
            return self.loss.sum()
        else:  # self.reduction == 'mean'
            return self.loss.mean()

    def __getattr__(self, name):
        return getattr(self.func, name)

    def __setstate__(self, data):
        self.__dict__.update(data)
