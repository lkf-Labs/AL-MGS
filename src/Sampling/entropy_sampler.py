
import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from PIL import Image

from Utils.uncertainty_utils import entropy
from Utils.utils import normalize
from Utils.plot_utils import  plot_all_uncertain_samples_from_lists

from Models.UNet import UNet


class EntropySampler:
    """
    Returns 2 lists of indices and corresponding uncertainty (based on mean entropy over pixels), and sampling time
    """

    def __init__(self, budget, trainer, **kwargs):
        self.budget = budget
        self.trainer = trainer
        # self.model = self.trainer.model
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        print(best_model_path)
        self.model = UNet.load_from_checkpoint(best_model_path)
        self.device = kwargs.get('device')
        self.model.to(self.device)

    def get_uncertainty(self, unlabeled_dataloader):
        """ We sample the images with mean entropy on output softmax"""

        sampling_start_time = time.time()

        self.model.eval()

        uncertainty_map_list = []
        indice_list = []
        data_list = []
        logits_list = []
        target_list = []

        # We iterate through the unlabeled dataloader
        for batch_idx, batch in enumerate(unlabeled_dataloader):
            with torch.no_grad():
                x, y, img_idx = batch['data'], batch['label'], batch['idx']
                x = x.to(self.device)

                logits, _ = self.model(x) # 4x2x352x736

                # We get output probability and prediction
                prob = F.softmax(logits, dim=1) # 4x2x352x736

                # We compute the entropy for each pixels of the image
                cur_uncertainty_map = entropy(prob, dim=1)   # 4x352x736 # shape (BS, H x W)

                original_img, heatmap = self.prepare_visualization_data(x, cur_uncertainty_map, idx=0)
                # generate entropy heatmap
                self.save_uncertainty_heatmap(
                    original_image=original_img,
                    uncertainty_map=heatmap,
                    save_path=f'entropy_heatmap/entropyheatmap_{img_idx[0]}.png'
                )


                uncertainty_map_list.extend(cur_uncertainty_map.detach().cpu().numpy())  # [(352,736),(352,736),......]

                # We keep track of all dataloader results
                indice_list.extend(img_idx.detach().cpu().numpy())
                data_list.extend(x.detach().cpu().numpy())
                logits_list.extend(logits.detach().cpu().numpy())
                target_list.extend(y.detach().cpu().numpy())

        uncertainty_map_array = np.stack(uncertainty_map_list, axis=0) # 153 x 352 x 736
        mean_uncertainty_list = np.mean(uncertainty_map_array, axis=(1, 2)) # 153

        sampling_time = time.time() - sampling_start_time

        # We plot top uncertainty samples and their uncertainty map
        # plot_all_uncertain_samples_from_lists(indice_list, data_list, logits_list, target_list, uncertainty_map_list,
        #                                       self.budget, self.trainer, title='Entropy of query images',
        #                                       model_out_channels=self.model.out_channels)

        return indice_list, mean_uncertainty_list, sampling_time

    def prepare_visualization_data(self, x_tensor, uncertainty_tensor, idx=0):
        """
        Convert tensors to numpy format for visualization

        Args:
            x_tensor: Input image tensor [batch_size, channels, height, width]
            uncertainty_tensor: Uncertainty heatmap tensor [batch_size, height, width]
            idx: Index of sample in batch to process

        Returns:
            Tuple of (original_img, uncertainty_map) containing:
            - Original image array (normalized)
            - Uncertainty heatmap array (normalized)
        """
        # Select specified sample and convert to CPU numpy array
        x_sample = x_tensor[idx].cpu().detach()  # Detach from computation graph and move to CPU
        uncertainty_sample = uncertainty_tensor[idx].cpu().detach()

        # Process image channels (supports both grayscale and RGB)
        if x_sample.shape[0] == 1:  # Single-channel grayscale image
            original_img = x_sample.squeeze().numpy()  # Reduce to [H, W]
        else:  # Three-channel color image
            original_img = x_sample.permute(1, 2, 0).numpy()  # Transpose to [H, W, C]

        # Get uncertainty heatmap
        uncertainty_map = uncertainty_sample.numpy()

        # Normalize data to [0,1] range
        original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
        uncertainty_map = (uncertainty_map - uncertainty_map.min()) / (uncertainty_map.max() - uncertainty_map.min())

        return original_img, uncertainty_map

    def save_uncertainty_heatmap(self, original_image, uncertainty_map, save_path, dpi=300):
        """
        Generate and save uncertainty heatmap overlay visualization

        Args:
            original_image: Original image array (single-channel [H,W] or three-channel [H,W,3])
            uncertainty_map: Uncertainty heatmap array [H,W]
            save_path: Path to save image (e.g., './heatmap.png')
            dpi: Output resolution (default 300dpi)
        """
        plt.figure(figsize=(8, 6))  # Create 8x6 inch figure

        # Auto-detect image type and display
        if original_image.ndim == 3 and original_image.shape[2] == 3:
            plt.imshow(original_image)  # Direct display for RGB images
        else:
            plt.imshow(original_image, cmap='gray')  # Grayscale images use grayscale colormap

        # Overlay semi-transparent heatmap (using jet colormap, 40% opacity)
        heatmap = plt.imshow(uncertainty_map,
                             cmap='jet',
                             alpha=0.4,  # Transparency setting
                             vmin=0,  # Min value anchor
                             vmax=1)  # Max value anchor

        # Optional colorbar display (currently commented out)
        # cbar = plt.colorbar(heatmap, shrink=0.8)
        # cbar.set_label('Uncertainty Level', rotation=270, labelpad=15)

        plt.axis('off')  # Turn off axes
        plt.savefig(save_path,
                    bbox_inches='tight',  # Remove white borders
                    pad_inches=0,
                    dpi=dpi)
        plt.close()