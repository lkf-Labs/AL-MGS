import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class RedundancyThresholdEWMA:
    def __init__(self, alpha=1.0, gamma=0.8):
        """
        Adaptive redundancy threshold calculator with EWMA smoothing.

        Args:
            alpha (float): Multiplier for standard deviation in threshold calculation
            gamma (float): Decay factor for EWMA (0.8 means 80% weight to history)
        """
        self.alpha = alpha
        self.gamma = gamma
        self.mu_prev = None  # Will be initialized in first cycle
        self.sigma_prev = None  # Will be initialized in first cycle
        self.threshold = 0.8  # Current adaptive threshold
        self.is_initialized = False

    def calculate_exter_similarity(self, top_uncertainty_indices,
                                   embedding_unlabeled,
                                   embedding_labeled,
                                   al_cycle):
        """
        Calculate external similarity and update adaptive threshold.

        Args:
            top_uncertainty_indices: Indices of candidate samples
            embedding_unlabeled: Unlabeled sample embeddings [n_unlabeled, dim]
            embedding_labeled: Labeled sample embeddings [n_labeled, dim]
            al_cycle: Current active learning cycle number (1-based)

        Returns:
            tuple: (current_mean, current_std, adaptive_threshold)
        """
        # 1. Calculate current batch statistics
        candidate_embeddings = embedding_unlabeled[top_uncertainty_indices]
        cos_sim_matrix = cosine_similarity(candidate_embeddings, embedding_labeled)
        max_similarities = np.amax(cos_sim_matrix, axis=1)

        current_mean = np.mean(max_similarities)
        current_std = np.std(max_similarities)

        # 2. Update EWMA estimates
        # Special handling for first cycle
        if al_cycle == 0 or not self.is_initialized:
            self.mu_prev = current_mean
            self.sigma_prev = current_std
            self.is_initialized = True
        else:
            # EWMA update for cycles > 1
            # Mean update: μ_t = γ*μ_{t-1} + (1-γ)*current_mean
            self.mu_prev = self.gamma * self.mu_prev + (1 - self.gamma) * current_mean
            # Std update: σ_t = sqrt(γ*σ_{t-1}^2 + (1-γ)*current_std^2)
            self.sigma_prev = np.sqrt(
                self.gamma * self.sigma_prev ** 2 +
                (1 - self.gamma) * current_std ** 2
            )

        # 3. Compute adaptive threshold: τ = μ + α*σ
        self.threshold = self.mu_prev + self.alpha * self.sigma_prev

        return current_mean, current_std, self.threshold


# Usage Example
if __name__ == "__main__":
    # Initialize with alpha=1.0, gamma=0.8
    threshold_calculator = RedundancyThresholdEWMA(alpha=1.0, gamma=0.8)

    # Mock data
    embedding_unlabeled = np.random.rand(100, 256)
    embedding_labeled = np.random.rand(20, 256)

    # Simulate 3 AL cycles
    for cycle in [1, 2, 3]:
        # Randomly select candidates (replace with actual uncertainty sampling)
        top_indices = np.random.choice(100, size=5, replace=False)

        # Calculate threshold
        mean, std, threshold = threshold_calculator.calculate_exter_similarity(
            top_indices,
            embedding_unlabeled,
            embedding_labeled,
            al_cycle=cycle
        )

        print(f"Cycle {cycle}: μ={mean:.3f}, σ={std:.3f}, Threshold={threshold:.3f}")