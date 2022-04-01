"""
@brief  Pytorch implementation of the Generalized Wasserstein Dice Loss [1]
        References:
        [1] "Generalised Wasserstein Dice Score for Imbalanced Multi-class
        Segmentation using Holistic Convolutional Networks",
        Fidon L. et al. MICCAI BrainLes 2017

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
@date   April 2020
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SUPPORTED_ALPHA_MODE = ['equal', 'simple']


class GeneralizedWassersteinDiceLoss(nn.Module):
    def __init__(self, dist_matrix, alpha_mode='equal', reduction='mean'):
        super(GeneralizedWassersteinDiceLoss, self).__init__()
        print('WARNING! Internal implementation of the Generalized Wasserstein Dice Loss is used')
        self.M = dist_matrix
        if isinstance(self.M, np.ndarray):
            self.M = torch.from_numpy(self.M).float()
        if torch.cuda.is_available():
            self.M = self.M.cuda()
        self.num_classes = self.M.size(0)
        assert alpha_mode in SUPPORTED_ALPHA_MODE
        self.alpha_mode = alpha_mode
        if alpha_mode == 'equal':
            print('Use the original definition of the Generalized Wasserstein Dice Loss from 2017')
        elif alpha_mode == 'simple':
            print('Use the Generalized Wasserstein Dice Loss with weights as in the Generalized Dice Loss')
        self.reduction = reduction

    def forward(self, input, target):
        epsilon = np.spacing(1)  # smaller number available
        assert input.dim() >= 4, "Input must be at least 4D scores Tensor " \
                                 "of shape (N, C, H, W) in 2d " \
                                 "and (N, C, H, W, D) in 3d."
        assert target.dim() >= 3, "Target must be at least 3D segmentation Tensor " \
                                  "of shape (N, H, W) or (N, 1, H, W) in 2d " \
                                  "and (N, H, W, D) or (N, 1, H, W, D) in 3d."
        # Aggregate spatial dimensions
        target = target.long()
        flat_input = input.view(input.size(0), input.size(1), -1)  # b,c,s
        flat_target = target.view(target.size(0), -1)  # b,s
        # Apply the softmax to the input scores map
        probs = F.softmax(flat_input, dim=1)  # b,c,s
        # Compute the Wasserstein distance map
        wass_dist_map = self.wasserstein_distance_map(probs, flat_target)  # b,s
        # Compute the weights alpha (one weight per class for each batch)
        alpha = self.compute_alpha_generalized_true_positives(flat_target)  # b,c

        # Compute the Wasserstein Dice
        if self.alpha_mode == 'simple':
            # use GDL-style alpha weights (i.e. normalize by the volume of each class)
            # contrary to [1] we also use alpha in the "generalized all error".
            true_pos = self.compute_generalized_true_positive(alpha, flat_target, wass_dist_map)
            denom = self.compute_denominator(alpha, flat_target, wass_dist_map)
            wass_dice = (2. * true_pos + epsilon) / (denom + epsilon)
        else:  # default: as in [1] (i.e. alpha=1 for all foreground classes and 0 for the background)
            # Compute the generalised number of true positives
            true_pos = self.compute_generalized_true_positive(alpha, flat_target, wass_dist_map)
            all_error = torch.sum(wass_dist_map, dim=1)
            wass_dice = (2. * true_pos + epsilon) / (2 * true_pos + all_error + epsilon)

        wass_dice_loss = 1. - wass_dice
        if self.reduction == 'sum':
            return wass_dice_loss.sum()
        elif self.reduction == 'none':
            return wass_dice_loss
        # default is mean reduction
        else:
            return wass_dice_loss.mean()

    def wasserstein_distance_map(self, flat_proba, flat_target):
        """
        Compute the voxel-wise Wasserstein distance (eq. 6 in [1])
        between the flattened prediction and the flattened labels (ground_truth) with respect
        to the distance matrix on the label space M.
        References:
        [1] "Generalised Wasserstein Dice Score for Imbalanced Multi-class
        Segmentation using Holistic Convolutional Networks",
        Fidon L. et al. MICCAI BrainLes 2017
        """
        # Turn the distance matrix to a map of identical matrix
        M_extended = torch.unsqueeze(self.M, dim=0)  # C,C -> 1,C,C
        M_extended = torch.unsqueeze(M_extended, dim=3)  # 1,C,C -> 1,C,C,1
        M_extended = M_extended.expand(
            (flat_proba.size(0), M_extended.size(1), M_extended.size(2), flat_proba.size(2))
        )
        # Expand the feature dimensions of the target
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)  # b,s -> b,1,s
        flat_target_extended = flat_target_extended.expand(  # b,1,s -> b,C,s
            (flat_target.size(0), M_extended.size(1), flat_target.size(1))
        )
        flat_target_extended = torch.unsqueeze(flat_target_extended, dim=1)  # b,C,s -> b,1,C,s
        # Extract the vector of class distances for the ground-truth label at each voxel
        M_extended = torch.gather(M_extended, dim=1, index=flat_target_extended)  # b,C,C,s -> b,1,C,s
        M_extended = torch.squeeze(M_extended, dim=1)  # b,1,C,s -> b,C,s
        # Compute the wasserstein distance map
        wasserstein_map = M_extended * flat_proba
        # Sum over the classes
        wasserstein_map = torch.sum(wasserstein_map, dim=1)  # b,C,s -> b,s
        return wasserstein_map

    def compute_generalized_true_positive(self, alpha, flat_target, wasserstein_distance_map):
        # Extend alpha to a map and select value at each voxel according to flat_target
        # alpha_extended = torch.unsqueeze(alpha, dim=0)  # C -> 1,C
        alpha_extended = torch.unsqueeze(alpha, dim=2)  # b,C -> b,C,1
        alpha_extended = alpha_extended.expand(  # b,C,1 -> b,C,s
            (flat_target.size(0), self.num_classes, flat_target.size(1))
        )
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)  # b,s -> b,1,s
        alpha_extended = torch.gather(
            alpha_extended, index=flat_target_extended, dim=1)  # b,C,s -> b,1,s

        # Get distance to background at each voxel according to flat_target
        # dist_background = torch.unsqueeze(self.get_distances_to_background(), dim=0)
        # dist_background = torch.unsqueeze(dist_background, dim=2)
        # dist_background = dist_background.expand(
        #     (flat_target.size(0), dist_background.size(1), flat_target.size(1))
        # )
        # dist_background = torch.gather(dist_background, index=flat_target_extended, dim=1)

        # Compute the generalized true positive as in eq. 9
        generalized_true_pos = torch.sum(
            alpha_extended * (1. - wasserstein_distance_map),
            dim=[1, 2],
        )
        return generalized_true_pos

    # def compute_generalized_error(self, alpha, flat_target, wasserstein_distance_map):
    #     # Extend alpha to a map and select value at each voxel according to flat_target
    #     # alpha_extended = torch.unsqueeze(alpha, dim=0)  # C -> 1,C
    #     alpha_extended = torch.unsqueeze(alpha, dim=2)  # b,C -> b,C,1
    #     alpha_extended = alpha_extended.expand(  # b,C,1 -> b,C,s
    #         (flat_target.size(0), self.num_classes, flat_target.size(1))
    #     )
    #     flat_target_extended = torch.unsqueeze(flat_target, dim=1)  # b,s -> b,1,s
    #     alpha_extended = torch.gather(
    #         alpha_extended, index=flat_target_extended, dim=1)  # b,C,s -> b,1,s
    #     # Compute the generalized true positive as in eq. 9
    #     generalized_true_pos = torch.sum(
    #         alpha_extended * wasserstein_distance_map,
    #         dim=[1, 2],
    #     )
    #     return generalized_true_pos

    def compute_denominator(self, alpha, flat_target, wasserstein_distance_map):
        # Extend alpha to a map and select value at each voxel according to flat_target
        # alpha_extended = torch.unsqueeze(alpha, dim=0)  # C -> 1,C
        alpha_extended = torch.unsqueeze(alpha, dim=2)  # b,C -> b,C,1
        alpha_extended = alpha_extended.expand(  # b,C,1 -> b,C,s
            (flat_target.size(0), self.num_classes, flat_target.size(1))
        )
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)  # b,s -> b,1,s
        alpha_extended = torch.gather(
            alpha_extended, index=flat_target_extended, dim=1)  # b,C,s -> b,1,s
        # Compute the generalized true positive as in eq. 9
        generalized_true_pos = torch.sum(
            alpha_extended * (2. - wasserstein_distance_map),
            dim=[1, 2],
        )
        return generalized_true_pos

    # def get_distances_to_background(self):
    #     # We assume that label 0 corresponds to the background
    #     dist_to_background = self.M[0, :]
    #     return dist_to_background

    def compute_alpha_generalized_true_positives(self, flat_target):
        """
        Compute the weights \alpha_l of eq. 9 in [1]
        References:
        [1] "Generalised Wasserstein Dice Score for Imbalanced Multi-class
        Segmentation using Holistic Convolutional Networks",
        Fidon L. et al. MICCAI BrainLes 2017
        """
        if self.alpha_mode == 'simple':  # GDL style
            # Define alpha like in the generalized dice loss
            # i.e. the inverse of the volume of each class.
            # Convert target to one-hot class encoding.
            one_hot = F.one_hot(  # shape: b,c,s
                flat_target, num_classes=self.num_classes).permute(0, 2, 1).float()
            volumes = torch.sum(one_hot, dim=2)  # b,c
            alpha = 1. / (volumes + 1.)
        else:  # equal; i.e. as in [1]
            # alpha weights are 0 for the background and 1 otherwise
            alpha_np = np.ones((flat_target.size(0), self.num_classes))
            alpha_np[:, 0] = 0.
            alpha = torch.from_numpy(alpha_np).float()
            if torch.cuda.is_available():
                alpha = alpha.cuda()
        return alpha


class GeneralizedWassersteinDiceLossCovid(GeneralizedWassersteinDiceLoss):
    def __init__(self, reduction='mean'):
        dist_matrix = np.array([
            [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 0.8, 0.8, 0.0, 0.0, 1.0],
            [1.0, 0.8, 0.0, 0.8, 0.0, 0.0, 1.0],
            [1.0, 0.8, 0.8, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        ])
        super(GeneralizedWassersteinDiceLossCovid, self).__init__(dist_matrix, reduction)
