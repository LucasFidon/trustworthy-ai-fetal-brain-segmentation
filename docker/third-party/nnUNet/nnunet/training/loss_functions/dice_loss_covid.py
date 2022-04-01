import torch
import torch.nn
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.training.loss_functions.dice_loss import get_tp_fp_fn_tn

# CLASS_TO_USE = torch.tensor([0, 1, 2, 3, 6]).long()
CLASS_TO_USE = torch.tensor([0, 1, 2]).long()
UNKNOWN_CLASS = [3]


class SoftDiceLoss(torch.nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=False, smooth=1e-5, reduction='mean'):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.reduction = reduction
        self.class_to_use = CLASS_TO_USE.cuda()

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        # nominator = 2 * tp + self.smooth
        nominator = 2 * tp
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / denominator

        # Remove the 'unknown' classes from the dice
        if self.batch_dice:
            dc = torch.index_select(dc, dim=0, index=self.class_to_use)
        else:
            dc = torch.index_select(dc, dim=1, index=self.class_to_use)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        # Take the mean over the classes
        dc = dc.mean(dim=1)

        # Apply the right reduction of the batch of loss values
        if self.reduction == 'none':
            return -dc
        elif self.reduction == 'sum':
            dc = dc.sum()
            return -dc
        else:  # default is mean reduction
            dc = dc.mean()
            return -dc


class CrossentropyND(torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """
    def forward(self, inp, target, loss_mask=None):
        target = target.long()
        num_batches = inp.size()[0]
        num_classes = inp.size()[1]

        inp = inp.view(num_batches, num_classes, -1)
        target = target.view(num_batches, -1)

        self.reduction = 'none'
        ce = super(CrossentropyND, self).forward(inp, target)

        # Mask the ce loss map (this flatten the tensor)
        if loss_mask is not None:
            mask = loss_mask.view(num_batches, -1)
            ce = ce[mask == 1]
        mean_ce = ce.mean()

        return mean_ce


class DC_and_CE_loss(torch.nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum",
                 weight_ce=1, weight_dice=1, reduction='mean'):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = CrossentropyND(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        self.ce.reduction = reduction
        self.dc.reduction = reduction

    def forward(self, net_output, target):
        # mask voxels labeled 'unknown'
        mask = torch.ones_like(target).long()
        for c in UNKNOWN_CLASS:
            mask[target == c] = 0

        # compute the loss for the voxels and class that are not 'unknown'
        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target, loss_mask=mask) if self.weight_ce != 0 else 0
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss

        return result

    @property
    def reduction(self):
        return self.ce.reduction  # should be the same as self.dc.reduction

    @reduction.setter
    def reduction(self, reduction):
        self.ce.reduction = reduction
        self.dc.reduction = reduction
