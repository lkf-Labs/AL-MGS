
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
import os


from Utils.load_utils import load_single_image, get_dict_from_config
from Utils.utils import natural_keys
from Data.transform import Preprocess

from Utils.load_utils import load_mg_img, load_mg_label


class MyDataset(Dataset):
    def __init__(self, data_folder, dataset_name, type='train', partition=3, kwargs={}, **transform_config):
        self.data_folder = data_folder
        self.dataset_name = dataset_name
        self.type = type
        self.partition = partition
        self.kwargs = kwargs
        self.transform_config = transform_config

        # We select the sample paths
        self.img_folder_path = os.path.join(self.data_folder, self.dataset_name, type, 'img')
        self.seg_folder_path = os.path.join(self.data_folder, self.dataset_name, type, 'label')

        # Retrieve the file lists of images and corresponding labels
        self.img_list = self.get_image_list(self.img_folder_path)
        self.seg_list = self.get_image_list(self.seg_folder_path)


    def get_image_list(self, folder_path):
        image_list = [f for f in os.listdir(folder_path) if f.endswith('.png')]  # get png files
        # Sort by numerical components in filenames
        image_list = sorted(image_list, key=lambda f: int(f.split('.')[0]))
        return image_list


    def __len__(self):
        """We return the total number of samples"""
        return len(self.img_list)

    def _get_partition(self, slice_position, num_slices):
        """
        We give the relative position of the slice in the volume with regards to the requested volume partition
        """
        _partition = (self.partition * slice_position) / num_slices
        partition = np.ceil(_partition)
        return int(partition)

    def _load_edge_weights(self, idx):
        all_weights_arr = load_single_image(self.edge_weights_save_path, self.edge_weights_list, idx)
        return all_weights_arr

    def __getitem__(self, idx):
        """We generate one sample of data"""
        # We load the segmentation samples
        img = load_mg_img(self.img_folder_path, self.img_list, idx)
        img = img / 255
        target = load_mg_label(self.seg_folder_path, self.seg_list, idx)
        if not np.all(np.isin(target, [0, 255])):
            target = np.where(target > 127, 255, 0)
        target = target / 255
        
        # We add channel dimension to target
        target = target[None]

        sample = {'data': torch.from_numpy(img).float(), 
                'label': torch.from_numpy(target).float(),
                }
        
        # We apply the transformations
        transf = Preprocess(**self.transform_config)
        transf_data = transf(sample)
        sample['data'] = transf_data['data']
        sample['label'] = transf_data['label']
        if 'type' in self.kwargs.keys() and self.kwargs['type'] == 'crf':
            sample['start_edges'] = self.start_edges
            sample['end_edges'] = self.end_edges
            sample['edge_weights'] = self._load_edge_weights(idx)
        sample["idx"] = idx

        return sample
