"""
@brief  PyTorch code for Hardness Weighted Sampler.

        It is required to use an Unreduced Loss functions
        with a Hardness Weighted Sampler
        (see loss_functions/unreduced_loss_function.py).

@author Lucas Fidon (lucas.fidon@kcl.ac.uk)
@date   30 Oct 2019.
"""

import random
import torch
from torch.utils.data import Sampler, BatchSampler
import torch.nn.functional as F
import numpy as np


class WeightedSampler(Sampler):
    def __init__(self, beta, weights_init, num_samples=-1, momentum=0.):
        """
        The weighted sampler allows to sample examples in a dataset
        with respect to a custom distribution computed as:
            \f$
            distrib = \softmax(\beta \textup{weights})
            \f$
        There is one weight per example in the dataset.
        The weights can be updated dynamically during training.
        The weighted sampling is optional and will be used only when
        self.use_uniform_sampler is False.
        This can be used to initialize the weights during the first epoch (warm-up).
        :param num_samples: int; number of samples in the dataset.
        :param beta: float; robustness parameter (must be positive).
        It allows to interpolate between empirical risk minimization (beta=0),
        and worst case minimization (beta=+inf).
        :param weights_init: float or 1d tensor;
        initialization for the weights of the sampling.
        Its size should be equal to num_samples.
        """
        print('WARNING! You are using a DEPRECATED implementation of the hardness weighted sampler')
        self.num_samples = num_samples

        # robustness parameter
        self.beta = beta

        # momentum used for the update of the loss history.
        # Only used in weighted sampling mode
        self.momentum = momentum

        # params for the robust sampling
        self.weights = None
        if isinstance(weights_init, float) or isinstance(weights_init, int):
            assert num_samples > 0, "The number of samples should be specified if a constant weights_init is used"
            print('Initialize the weights of the sampler to the value', weights_init)
            # Add some gaussian noise on top of the initial weight value.
            self.weights = torch.tensor(
                np.random.normal(loc=weights_init, scale=0.001*weights_init, size=num_samples))
        else:
            assert len(weights_init.shape) == 1, "initial weights should be a 1d tensor"
            self.weights = weights_init.float()
            if self.num_samples <= 0:
                self.num_samples = weights_init.shape[0]
            else:
                assert self.num_samples == weights_init.shape[0], \
                    "weights_init should have a size equal to num_samples"

    def get_distribution(self):
        # Apply softmax to the weights vector.
        # This seems to be the most numerically stable way
        # to compute the softmax
        distribution = F.log_softmax(
            self.beta * self.weights, dim=0).data.exp()
        return distribution

    def draw_samples(self, n):
        """
        Draw n samples with respect to the sample weights.
        """
        eps = 0.0001 / self.num_samples
        # Get the distribution (softmax)
        distribution = self.get_distribution()
        p = distribution.numpy()
        # Set min proba to epsilon for stability
        p[p <= eps] = eps
        p /= p.sum()
        # Use numpy implementation of multinomial sampling because it is much faster
        sample_list = np.random.choice(
            self.num_samples,
            n,
            p=p,
            replace=False,
        ).tolist()
        return sample_list

    def update_weight(self, idx, new_weight):
        """
        Update the weight of sample idx for new_weight.
        :param idx: int; index of the sample of which the weight have to be updated.
        :param new_weight: float; new weight value for idx.
        """
        # momentum for the update
        self.weights[idx] = self.momentum * self.weights[idx] \
                            + (1. - self.momentum) * new_weight

    def __iter__(self):
        sample_list = self.draw_samples(self.num_samples)
        return iter(sample_list)

    def __len__(self):
        return self.num_samples

    def save_weights(self, save_path):
        torch.save(self.weights, save_path)

    def load_weights(self, weights_path):
        print('Load the sampling weights from %s' % weights_path)
        weights = torch.load(weights_path)
        self.weights = weights
