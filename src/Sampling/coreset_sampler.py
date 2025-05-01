
from comet_ml import Experiment
import time
import numpy as np
from sklearn.metrics import pairwise_distances

import torch
from torch.utils.data import DataLoader

from Models.unet_modules import max_pooling_2d
from Models.UNet import UNet

class CoresetsSampler:
    """
    Coresets sampler
    Implementation adapted from https://github.com/anonneurips-8435/Active_Learning/blob/eba1acddf0eeddabce3ee618349369e89c4f31dd/main/active_learning_strategies/core_set.py
    A diversity-based approach using coreset selection. The embedding of each example is computed by the network’s
    penultimate layer and the samples at each round are selected using a greedy furthest-first
    traversal conditioned on all labeled examples.

            pooling_kwargs = {'kernel_size': 4, 'stride': 4, 'padding': 0}

            sampler = CoresetsSampler(self.budget)
            querry_indices = sampler.sample(
                self.models_dic['model'], unlabeled_dataloader, self.device, self.experiment, self.finite_labeled_dataloader, pooling_kwargs)

    """

    def __init__(self, budget, trainer, **kwargs):
        self.budget = budget
        self.trainer = trainer
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        print(best_model_path)
        self.model = UNet.load_from_checkpoint(best_model_path)
        self.device = kwargs.get('device')
        self.model.to(self.device)

    def sample(self, unlabeled_dataloader, labeled_dataloader, pooling_kwargs):
        sampling_start_time = time.time()

        embedding_unlabeled, idx_unlabeled = get_embedding(self.model, unlabeled_dataloader, pooling_kwargs, self.device) # embedding_unlabeled=[153, 1036288], idx_unlabeled=[153]
        embedding_labeled, idx_labeled = get_embedding(self.model, labeled_dataloader, pooling_kwargs, self.device) # embedding_labeled=[10, 1036288], idx_labeled=[10]

        chosen_indices = furthest_first(embedding_unlabeled, embedding_labeled, self.budget)
        print('chosen_indices {}'.format(chosen_indices))

        querry_pool_indices = [idx_unlabeled[idx] for idx in chosen_indices]
        sampling_time = time.time() - sampling_start_time
        return querry_pool_indices, sampling_time


def furthest_first(unlabeled_set, labeled_set, budget):
    """
    Selects points with maximum distance

    Parameters
    ----------
    unlabeled_set: numpy array
        Embeddings of unlabeled set
    labeled_set: numpy array
        Embeddings of labeled set
    budget: int
        Number of points to return
    Returns
    ----------
    idxs: list
        List of selected data point indexes with respect to unlabeled_x
    """
    m = np.shape(unlabeled_set)[0] # 153
    if np.shape(labeled_set)[0] == 0:
        min_dist = np.tile(float("inf"), m)
    else:
        dist_ctr = pairwise_distances(unlabeled_set, labeled_set)
        min_dist = np.amin(dist_ctr, axis=1)

    idxs = []

    for i in range(budget):
        idx = min_dist.argmax()
        idxs.append(idx)
        dist_new_ctr = pairwise_distances(
            unlabeled_set, unlabeled_set[[idx], :])
        for j in range(m):
            min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

    return idxs


def get_embedding(model, dataloader, pooling_kwargs, device):
    model.eval()
    # embedding = torch.zeros([dataloader.shape[0], model.get_embedding_dim()])
    embedding_list = []
    idx_list = []

    pool = max_pooling_2d(**pooling_kwargs)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            with torch.no_grad():
                x, y, img_idx = batch['data'], batch['label'], batch['idx']
                x = x.to(device)
                _, [enc_1, enc_2, enc_3, center, dec_1, dec_2, dec_3] = model(x)
                ##embedding[idxs] = features.data.cpu()
                # cur_features = dec_3.view(dec_3.size(0), -1)
                # Downsampling for Dimensionality Reduction
                pooled_dec = pool(dec_3)
                # flatten
                cur_features = pooled_dec.view(pooled_dec.size(0), -1)
                # add to embedding list
                # embedding_list.append(cur_features.data.cpu())
                # idx_list.append(img_idx.item())
                embedding_list.extend(cur_features.detach().cpu().numpy())
                idx_list.extend(img_idx.detach().cpu().numpy())
    # embedding = np.concatenate(embedding_list, axis=0)
    embedding = np.array(embedding_list)
    print('embedding {}'.format(embedding.shape))
    print(idx_list)
    return embedding, idx_list
