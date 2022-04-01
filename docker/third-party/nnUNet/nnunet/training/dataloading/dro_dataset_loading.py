import torch
from nnunet.training.dataloading.dataset_loading import DataLoader3D
# from nnunet.training.dataloading.dro.weighted_sampler import WeightedSampler
from hardness_weighted_sampler.sampler.weighted_sampler import WeightedSampler


class HardnessWeightedDataLoader3D(DataLoader3D):
    def __init__(self, data, patch_size, final_patch_size, batch_size,
                 has_prev_stage=False, oversample_foreground_percent=0.,
                 memmap_mode="r", pad_mode="edge", pad_kwargs_data=None,
                 pad_sides=None, beta=0, sampling_weights_init=2., dro_momentum=0.):
        super(HardnessWeightedDataLoader3D, self).__init__(
            data, patch_size, final_patch_size, batch_size,
            has_prev_stage, oversample_foreground_percent, memmap_mode,
            pad_mode, pad_kwargs_data, pad_sides
        )
        self.idx_sampler = WeightedSampler(
            beta=beta,
            weights_init=sampling_weights_init,
            num_samples=len(self.list_of_keys),
            momentum=dro_momentum,
        )
        self.reverse_list_of_keys = {
            key: i for i, key in enumerate(self.list_of_keys)
        }
        print("DRO is used with beta=%f and initial weights value=%f" %
              (beta, sampling_weights_init))

    def draw_random_keys(self, n):
        keys_idx = self.idx_sampler.draw_samples(n)
        keys = [self.list_of_keys[i] for i in keys_idx]
        return keys

    def update_weights(self, batch_new_weights, batch_keys):
        """
        Update the weights for the last batch.
        The indices corresponding the the weights in batch_new_weights
        should be the indices that have been copied into self.batch
        :param batch_new_weights: float or double array; new weights value for the last batch.
        :param indices: int list; indices of the samples to update.
        """
        # assert len(batch_keys) == batch_new_weights.size()[0], "number of weights in " \
        #                                                        "input batch does not " \
        #                                                        "correspond to the number " \
        #                                                        "of keys."
        # Update the weights for all the indices in self.batch
        for key, new_weight in zip(batch_keys, batch_new_weights):
            idx = self.reverse_list_of_keys[key]
            self.idx_sampler.update_weight(idx, new_weight)

    def get_importance_sampling_weights(self, batch_new_weights, batch_keys):
        # assert len(batch_keys) == batch_new_weights.size()[0], "number of weights in " \
        #                                                        "input batch does not " \
        #                                                        "correspond to the number " \
        #                                                        "of indices."
        batch_indices = [self.reverse_list_of_keys[key] for key in batch_keys]
        # for key in batch_keys:
        #     idx = self.reverse_list_of_keys[key]
        #     batch_indices.append(idx)
        importance_weights = self.idx_sampler.get_importance_sampling_weights(
            batch_new_weights, batch_indices)
        # log_importance_weights = []
        # for key, new_weight in zip(batch_keys, batch_new_weights):
        #     idx = self.reverse_list_of_keys[key]
        #     w = self.beta * (1. - self.momentum) * (new_weight - self.idx_sampler.weights[idx])
        #     log_importance_weights.append(w)
        # importance_weights = torch.tensor(
        #     log_importance_weights, requires_grad=False).exp().float()
        return importance_weights

    @property
    def beta(self):
        return self.idx_sampler.beta

    @property
    def momentum(self):
        return self.idx_sampler.momentum

    def save_weights(self, save_path):
        self.idx_sampler.save_weights(save_path)

    def load_weights(self, weights_path):
        self.idx_sampler.load_weights(weights_path)
