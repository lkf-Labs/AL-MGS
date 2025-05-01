
from comet_ml import Experiment
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
from torch.utils.data import DataLoader

from Models.unet_modules import max_pooling_2d
from Models.UNet import UNet

class SDSSampler:
    def __init__(self, budget, trainer, similarity_metric, **kwargs):
        self.budget = budget
        self.trainer = trainer
        self.similarity_metric = similarity_metric
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        print(best_model_path)
        self.model = UNet.load_from_checkpoint(best_model_path)
        self.device = kwargs.get('device')

        self.model.to(self.device)

    def sample(self, unlabeled_dataloader, labeled_dataloader, pooling_kwargs):
        sampling_start_time = time.time()

        embedding_unlabeled, idx_unlabeled = get_embedding(self.model, unlabeled_dataloader, pooling_kwargs, self.device) # embedding_unlabeled=[153, 1036288], idx_unlabeled=[153]
        embedding_labeled, idx_labeled = get_embedding(self.model, labeled_dataloader, pooling_kwargs, self.device) # embedding_labeled=[10, 1036288], idx_labeled=[10]
        if self.similarity_metric == 'CosineSimilarity':
            chosen_indices = select_lowest_similarity_with_cos(embedding_unlabeled, embedding_labeled, self.budget)
        elif self.similarity_metric == 'EuclideanDistance':
            chosen_indices = select_lowest_similarity_with_EuclideanDistance(embedding_unlabeled, embedding_labeled, self.budget)
        elif self.similarity_metric == 'ManhattanDistance':
            chosen_indices = select_lowest_similarity_with_ManhattanDistance(embedding_unlabeled, embedding_labeled, self.budget)
        print('chosen_indices {}'.format(chosen_indices))
        # t-SNE visualization
        visualize_with_tsne(embedding_unlabeled, embedding_labeled, chosen_indices)

        querry_pool_indices = [idx_unlabeled[idx] for idx in chosen_indices]
        sampling_time = time.time() - sampling_start_time
        return querry_pool_indices, sampling_time

    def get_vector(self, unlabeled_dataloader, labeled_dataloader, pooling_kwargs):
        embedding_unlabeled, idx_unlabeled = get_embedding(self.model, unlabeled_dataloader, pooling_kwargs,  self.device)  # embedding_unlabeled=[153, 1036288], idx_unlabeled=[153]
        embedding_labeled, idx_labeled = get_embedding(self.model, labeled_dataloader, pooling_kwargs, self.device)  # embedding_labeled=[10, 1036288], idx_labeled=[10]
        return embedding_unlabeled, embedding_labeled

def select_lowest_similarity_with_cos(unlabeled_set, labeled_set, budget):
    """
    Selects samples with lowest similarity

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
    # Calculate intra-group similarity using cos
    cos_sim_matrix = cosine_similarity(unlabeled_set, unlabeled_set)
    np.fill_diagonal(cos_sim_matrix, -np.inf)
    max_intra_similarity = np.amax(cos_sim_matrix, axis=1)
    # Calculate the similarity with the labeled group
    cos_sim_matrix = cosine_similarity(unlabeled_set, labeled_set)
    max_exter_similarity = np.amax(cos_sim_matrix, axis=1)
    total_max_similarity = max_intra_similarity + max_exter_similarity
    # Get the indices of the smallest `budget` values, sorted in ascending order
    idxs = np.argsort(total_max_similarity)[:budget]

    return idxs


def get_embedding(model, dataloader, pooling_kwargs, device):
    model.eval()
    embedding_list = []
    idx_list = []

    pool = max_pooling_2d(**pooling_kwargs)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            with torch.no_grad():
                x, y, img_idx = batch['data'], batch['label'], batch['idx']
                x = x.to(device)
                _, [enc_1, enc_2, enc_3, center, dec_1, dec_2, dec_3] = model(x)
                pooled_dec = pool(dec_3) # 4x64x88x184
                cur_features = pooled_dec.view(pooled_dec.size(0), -1)
                embedding_list.extend(cur_features.detach().cpu().numpy())
                idx_list.extend(img_idx.detach().cpu().numpy())
    embedding = np.array(embedding_list)
    print('embedding {}'.format(embedding.shape))
    print(idx_list)
    return embedding, idx_list

def select_lowest_similarity_with_EuclideanDistance(unlabeled_set, labeled_set, budget):
    """
    Selects samples with lowest similarity

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
    # Calculate intra-group similarity using Euclidean Distance
    ed_sim_matrix = euclidean_distances(unlabeled_set, unlabeled_set)
    np.fill_diagonal(ed_sim_matrix, -np.inf)
    max_intra_distance = np.amax(ed_sim_matrix, axis=1)
    # Calculate the similarity with the labeled group
    ed_sim_matrix = euclidean_distances(unlabeled_set, labeled_set)
    max_exter_distance = np.amax(ed_sim_matrix, axis=1)
    total_max_distance = max_intra_distance + max_exter_distance
    # Get the indices of the smallest `budget` values, sorted in descending order
    idxs = np.argsort(total_max_distance)[::-1][:budget]

    return idxs


def select_lowest_similarity_with_ManhattanDistance(unlabeled_set, labeled_set, budget):
    """
    Selects samples with lowest similarity

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
    # Calculate intra-group similarity using Manhattan Distance
    md_sim_matrix = manhattan_distances(unlabeled_set, unlabeled_set)
    np.fill_diagonal(md_sim_matrix, -np.inf)
    max_intra_distance = np.amax(md_sim_matrix, axis=1)
    # Calculate the similarity with the labeled group
    md_sim_matrix = manhattan_distances(unlabeled_set, labeled_set)
    max_exter_distance = np.amax(md_sim_matrix, axis=1)
    total_max_distance = max_intra_distance + max_exter_distance
    # Get the indices of the smallest `budget` values, sorted in descending order
    idxs = np.argsort(total_max_distance)[::-1][:budget]

    return idxs


def visualize_with_tsne(embedding_unlabeled, embedding_labeled, querry_indices,
                        perplexity=30, n_iter=1000, random_state=42):
    """
    Visualize sample distribution using t-SNE dimensionality reduction

    Args:
        embedding_unlabeled (np.array): Feature embeddings of unlabeled samples [n_unlabeled, dim]
        embedding_labeled (np.array): Feature embeddings of labeled samples [n_labeled, dim]
        querry_indices (list): Indices of queried samples in embedding_unlabeled
        perplexity (int): t-SNE perplexity parameter (controls neighborhood size)
        n_iter (int): Number of optimization iterations
        random_state (int): Random seed for reproducibility
    """
    # Concatenate all embeddings for joint t-SNE processing
    all_embeddings = np.concatenate([embedding_unlabeled, embedding_labeled], axis=0)

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=perplexity,
                n_iter=n_iter, random_state=random_state)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Split back into respective sets
    n_unlabeled = len(embedding_unlabeled)
    unlabeled_2d = embeddings_2d[:n_unlabeled]
    labeled_2d = embeddings_2d[n_unlabeled:]

    # Extract coordinates of queried samples
    querry_2d = unlabeled_2d[querry_indices]

    # Create mask to exclude queried samples from unlabeled set
    mask = np.ones(len(unlabeled_2d), dtype=bool)
    mask[querry_indices] = False
    unlabeled_2d = unlabeled_2d[mask]

    # Initialize figure
    plt.figure(figsize=(10, 8))

    # 1. Plot unlabeled samples (blue circles)
    plt.scatter(unlabeled_2d[:, 0], unlabeled_2d[:, 1],
                c=np.array([[157 / 255, 195 / 255, 230 / 255]]),  # RGB(157,195,230)
                edgecolors=np.array([[89 / 255, 128 / 255, 198 / 255]]),  # Border RGB(89,128,198)
                linewidths=1.5,
                alpha=0.8,
                label='Unlabeled Samples',
                s=200)

    # 2. Plot labeled samples (purple circles)
    plt.scatter(labeled_2d[:, 0], labeled_2d[:, 1],
                c=np.array([[194 / 255, 152 / 255, 194 / 255]]),
                edgecolors=np.array([[122 / 255, 61 / 255, 163 / 255]]),
                linewidths=1.5,
                alpha=0.8,
                label='Labeled Samples',
                s=200)

    # 3. Highlight queried samples (blue stars)
    plt.scatter(querry_2d[:, 0], querry_2d[:, 1],
                marker='*',
                c=np.array([[157 / 255, 195 / 255, 230 / 255]]),
                edgecolors=np.array([[89 / 255, 128 / 255, 198 / 255]]),
                linewidths=1.5,
                s=200,
                label='Selected Samples')

    # Configure plot aesthetics
    plt.legend(fontsize='x-large')
    plt.title('t-SNE Visualization of SDS Sample Selection', fontsize=22)
    plt.xticks(fontsize='xx-large')
    plt.yticks(fontsize='xx-large')
    plt.grid(alpha=0.2)

    # Save and display
    plt.tight_layout()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'tsne_visualization_{timestamp}.png', dpi=300)
    plt.show()