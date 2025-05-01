
import time
import numpy as np
import cv2
import os

import torch
from pytorch_lightning import seed_everything
import torch.nn.functional as F

from Sampling.entropy_sampler import EntropySampler
from Sampling.dropout_sampler import DropoutSampler
from Sampling.learningloss_sampler import LearningLossSampler
from Sampling.TTA_sampler import TTASampler
from Sampling.coreset_sampler import CoresetsSampler
from Sampling.sds_sampler import SDSSampler
from Sampling.adversary_sampler import AdversarySampler
from Utils.stochasticbatches_utils import generate_random_groups, aggregate_group_uncertainty, select_top_positions_with_highest_uncertainty, select_top_positions_with_highest_diversity, select_top_diversity, calculate_exter_similarity
from Utils.load_utils import save_to_logger
from Utils.adaptive_thresholding_utils import RedundancyThresholdEWMA
from Models.UNet import UNet
from Utils.utils import to_onehot, normalize
from Data.dataset import MyDataset
from Models.UNet_LearningLoss import UNetLearningLoss
from Models.UNet_VAAL import UNetVAAL
# current stage identifier
cur_stage = 1
# A fixed threshold of 0.9 is adopted in the first cycle, followed by adaptive thresholding in subsequent active learning rounds.
threshold = 0.9
threshold_calculator = RedundancyThresholdEWMA(alpha=1.0, gamma=0.8)
def sample_new_indices(sampling_config, budget, unlabeled_dataloader, labeled_dataloader, test_dataloader, trainer, budget_result_path, data_dir, dataset_name, al_cycle,  **kwargs):
    """
    We implement the logic of sample selection for labeling (AL step)
    """
    global cur_stage, threshold, threshold_calculator
    strategy = sampling_config['strategy']
    sampling_params = sampling_config['sampling_params']
    if sampling_params is not None and 'DC_start_AL_cycle' in sampling_params and al_cycle >= sampling_params['DC_start_AL_cycle']:
        strategy = sampling_params['strategy']
    print('\n sampling strategy {}'.format(strategy))
    stage_change_message_path = budget_result_path
    test_result_path = os.path.join(budget_result_path, 'cur_test_results')
    os.makedirs(test_result_path, exist_ok=True)
    budget_result_path = os.path.join(budget_result_path, 'cur_budget_results')
    os.makedirs(budget_result_path, exist_ok=True)

    seed = kwargs.get('seed')
    unlabeled_indices = kwargs['unlabeled_indices']
    print('Length unlabeled dataloder: {} ({} samples)'.format(len(unlabeled_dataloader), len(unlabeled_indices)))

    position_list = list(range(len(unlabeled_indices)))

    # We save sampling parameters in trainer
    save_to_logger(trainer.logger, 'hyperparameter', strategy, 'sampling_strategy')
    save_to_logger(trainer.logger, 'hyperparameter', sampling_params, 'sampling_config')

    seed_everything(seed)

    uncertainty_based_strategies = ['Entropy', 'Dropout', 'LearningLoss', 'TTA', 'EdgeEntropy']

    additional_time = 0
    # progressive sampling process
    if strategy == 'progressive_hybrid':
        if cur_stage == 1:
            strategy = sampling_params['strategy1']
        elif cur_stage == 2:
            strategy = sampling_params['strategy2']

    ### For random sampling
    if strategy == 'RS':
        sampling_start_time = time.time()
        query_indices = list(np.random.choice(unlabeled_indices, size=budget, replace=False))
        sampling_time = time.time() - sampling_start_time
        uncertainty_values = []

    ### For diversity-based sampling
    elif strategy == 'CoreSet':
        pooling_kwargs = {'kernel_size': 4, 'stride': 4, 'padding': 0}
        sampler = CoresetsSampler(budget, trainer, **kwargs)
        query_indices, sampling_time = sampler.sample(unlabeled_dataloader, labeled_dataloader, pooling_kwargs)
        uncertainty_values = []

    elif strategy == 'VAAL':
        sampling_start_time = time.time()
        sampler = AdversarySampler(budget, trainer, **kwargs)
        query_indices = sampler.sample(unlabeled_dataloader)
        sampling_time = time.time() - sampling_start_time
        uncertainty_values = []

    elif strategy == 'SDS':
        pooling_kwargs = {'kernel_size': 64, 'stride': 64, 'padding': 0}
        sampler = SDSSampler(budget, trainer, sampling_params['similarity_metric'], **kwargs)
        query_indices, sampling_time = sampler.sample(unlabeled_dataloader, labeled_dataloader, pooling_kwargs)
        uncertainty_values = []

    ### For uncertainty-based sampling 
    elif strategy in uncertainty_based_strategies:
        if strategy == 'Entropy':
            sampler = EntropySampler(budget, trainer, **kwargs)
            unlabeled_indice_list, uncertainty_list, sampling_time = sampler.get_uncertainty(unlabeled_dataloader)

        elif strategy == 'Dropout':
            sampler = DropoutSampler(budget, trainer, **kwargs)
            unlabeled_indice_list, uncertainty_list, sampling_time = sampler.get_uncertainty(unlabeled_dataloader)

        elif strategy == 'TTA':
            sampler = TTASampler(budget, trainer, **kwargs)
            unlabeled_indice_list, uncertainty_list, sampling_time = sampler.get_uncertainty(unlabeled_dataloader)

        elif strategy == 'LearningLoss':
            sampler = LearningLossSampler(budget, trainer, **kwargs)
            # We will compute the uncertainty of all data-points in the unlabeled dataloader or on a subset of it
            num_subset_indices = kwargs.get('num_subset_indices')
            if num_subset_indices == 'all':
                unlabeled_indice_list, uncertainty_list, sampling_time = sampler.get_uncertainty(unlabeled_dataloader)
            else:
                subset_indices = list(np.random.choice(unlabeled_dataloader.sampler.indices, size=num_subset_indices, replace=False))
                subset_sampler = torch.utils.data.sampler.SubsetRandomSampler(subset_indices,
                                                                            generator=torch.Generator())
                subset_dataloader = torch.utils.data.DataLoader(unlabeled_dataloader.dataset, sampler=subset_sampler,
                                                                batch_size=unlabeled_dataloader.batch_size,
                                                                drop_last=False, num_workers=1)
                unlabeled_indice_list, uncertainty_list = sampler.get_uncertainty(subset_dataloader)

        if sampling_params is not None and 'RandomPacks' in sampling_params.keys():
            print('uncertainty_list {}'.format(sorted(np.unique(uncertainty_list))))
            additional_start_time = time.time()

            # We extract parameters
            num_groups = sampling_params['RandomPacks']['num_groups']
            resampling = sampling_params['RandomPacks']['resampling']
            aggregation = sampling_params['RandomPacks']['aggregation']
            # SB + not dc
            if 'Diversity_Constraint' not in sampling_params['RandomPacks'] or al_cycle < sampling_params['RandomPacks']['Diversity_Constraint']['start_AL_cycle']:
                position_group_list = generate_random_groups(position_list, num_groups, group_size=budget, resample=resampling) # position_group_list = [15,10]
                print("len(position_group_list): {}".format(position_group_list))
                aggregated_uncertainty_list = aggregate_group_uncertainty(position_group_list, uncertainty_list, aggregation=aggregation) # uncertainty_list = [153] aggregated_uncertainty_list = [15]
                position_with_highest_uncertainty = select_top_positions_with_highest_uncertainty(position_group_list, aggregated_uncertainty_list, num_groups=1)
                # We get the corresponding indices and uncertainty values for the selected position
                # Since we only want one group of indices, we take the first element of the position_with_highest_uncertainty
                query_indices = [[unlabeled_indice_list[i] for i in indices] for indices in position_with_highest_uncertainty][0]  # curent AL Cycle query indexs
                uncertainty_values = [[uncertainty_list[i] for i in indices] for indices in position_with_highest_uncertainty][0] # corresponding uncertaint score
                if sampling_config['strategy'] == 'progressive_hybrid' and cur_stage == 1:
                    pooling_kwargs = {'kernel_size': 64, 'stride': 64, 'padding': 0}
                    sampler = SDSSampler(budget, trainer, 'CosineSimilarity', **kwargs)
                    # Obtain the embedding vector of the last dimension of the decoder.
                    embedding_unlabeled, embedding_labeled = sampler.get_vector(unlabeled_dataloader, labeled_dataloader, pooling_kwargs)
                    # We calculate the similarity between the samples obtained through uncertainty sampling and the labeled set.
                    mean_similarity, std_similarity, adaptive_threshold= threshold_calculator.calculate_exter_similarity(position_with_highest_uncertainty[0], embedding_unlabeled, embedding_labeled, al_cycle)
                    print("The similarity score between the current candidate samples and the existing labeled set is:", mean_similarity)
                    message_path = os.path.join(stage_change_message_path, "stage_change_message.txt")
                    with open(message_path, 'w') as file:
                        file.write(
                            f"At cycle {al_cycle + 1}, The similarity score between the current candidate samples and the existing labeled set is: {mean_similarity}, current threshold = {threshold}"
                            f"std = {std_similarity}")
                    # if mean_similarity > sampling_params['progressive_threshold']:
                    if mean_similarity > threshold:
                        print(str(al_cycle + 1) +"============================Enter the second stage=====================================")
                        cur_stage = 2
                        with open(message_path, 'a') as file:
                            file.write(f"At cycle {al_cycle + 1}: Transitioning to the second stage of the sampling process.")
                        return None, None, None
                    threshold = adaptive_threshold
                additional_time = time.time() - additional_start_time
            else:
                select_groups_num = sampling_params['RandomPacks']['Diversity_Constraint']['select_groups_num']
                position_group_list = generate_random_groups(position_list, num_groups, group_size=budget, resample=resampling)
                print("len(position_group_list): {}".format(position_group_list))
                aggregated_uncertainty_list = aggregate_group_uncertainty(position_group_list, uncertainty_list, aggregation=aggregation)
                position_with_highest_uncertainty = select_top_positions_with_highest_uncertainty(position_group_list, aggregated_uncertainty_list, num_groups=select_groups_num)
                pooling_kwargs = {'kernel_size': 64, 'stride': 64, 'padding': 0}
                sampler = SDSSampler(budget, trainer, 'CosineSimilarity', **kwargs)
                # Obtain the embedding vector of the last dimension of the decoder.
                embedding_unlabeled, embedding_labeled = sampler.get_vector(unlabeled_dataloader, labeled_dataloader, pooling_kwargs)
                # We select the most diverse group from the top groups with the highest uncertainty using cosine similarity.
                position_with_highest_diversity = select_top_positions_with_highest_diversity(position_with_highest_uncertainty, embedding_unlabeled, embedding_labeled, num_groups=1)
                query_indices = [[unlabeled_indice_list[i] for i in indices] for indices in position_with_highest_diversity][0]
                uncertainty_values = [[uncertainty_list[i] for i in indices] for indices in position_with_highest_diversity][0]
                additional_time = time.time() - additional_start_time

        else:
            print('uncertainty_list {}'.format(sorted(np.unique(uncertainty_list))))
            additional_start_time = time.time()
            if sampling_params is not None and 'Diversity_Constraint' in sampling_params:
                # Index in descending order
                position_with_highest_uncertainty = select_top_positions_with_highest_uncertainty(position_list, uncertainty_list, num_groups=sampling_params['Diversity_Constraint']['select_top_uncertainty_num'])
                pooling_kwargs = {'kernel_size': 64, 'stride': 64, 'padding': 0}
                sampler = SDSSampler(budget, trainer, 'CosineSimilarity', **kwargs)
                # Obtain the embedding vector of the last dimension of the decoder.
                embedding_unlabeled, embedding_labeled = sampler.get_vector(unlabeled_dataloader, labeled_dataloader, pooling_kwargs)
                # We select the most diverse group from the top groups with the highest uncertainty using cosine similarity.
                position_with_highest_diversity_and_top_uncertainty = select_top_diversity(position_with_highest_uncertainty, embedding_unlabeled, embedding_labeled, num_groups=budget)
                query_indices = [unlabeled_indice_list[i] for i in position_with_highest_diversity_and_top_uncertainty]
                uncertainty_values = [uncertainty_list[i] for i in position_with_highest_diversity_and_top_uncertainty]
            else:
                # Index in descending order
                position_with_highest_uncertainty = select_top_positions_with_highest_uncertainty(position_list, uncertainty_list, num_groups=budget)
                query_indices = [unlabeled_indice_list[i] for i in position_with_highest_uncertainty]
                uncertainty_values = [uncertainty_list[i] for i in position_with_highest_uncertainty]
                if sampling_config['strategy'] == 'progressive_hybrid' and cur_stage == 1:
                    pooling_kwargs = {'kernel_size': 64, 'stride': 64, 'padding': 0}
                    sampler = SDSSampler(budget, trainer, 'CosineSimilarity', **kwargs)
                    # Obtain the embedding vector of the last dimension of the decoder.
                    embedding_unlabeled, embedding_labeled = sampler.get_vector(unlabeled_dataloader, labeled_dataloader, pooling_kwargs)
                    # We calculate the similarity between the query samples obtained through uncertainty sampling and the labeled set.
                    mean_similarity, std_similarity, adaptive_threshold= threshold_calculator.calculate_exter_similarity(position_with_highest_uncertainty, embedding_unlabeled, embedding_labeled, al_cycle)
                    print("The similarity score between the current candidate samples and the existing labeled set is:", mean_similarity)
                    message_path = os.path.join(stage_change_message_path, "stage_change_message.txt")
                    with open(message_path, 'w') as file:
                        file.write(f"At cycle {al_cycle + 1}, The similarity score between the current candidate samples and the existing labeled set is: {mean_similarity}, current threshold = {threshold}"
                                   f"std = {std_similarity}")
                    # if mean_similarity > sampling_params['progressive_threshold']:
                    if mean_similarity > threshold:
                        print(str(al_cycle + 1) +"============================Enter the second stage=====================================")
                        cur_stage = 2
                        with open(message_path, 'a') as file:
                            file.write(f"At cycle {al_cycle + 1}: Transitioning to the second stage of the sampling process.")
                        return None, None, None
                    threshold = adaptive_threshold
            additional_time = time.time() - additional_start_time
            
        save_to_logger(trainer.logger, 'metric', np.round(additional_time/60, 2), 'sampling additional time (min)', trainer.current_epoch)
        save_to_logger(trainer.logger, 'list', uncertainty_values, 'uncertainty_values', None)
    elif strategy == 'hybrid':
        pooling_kwargs = {'kernel_size': 64, 'stride': 64, 'padding': 0}
        sampler = SDSSampler(sampling_params['select_top_diverse_num'], trainer, 'CosineSimilarity', **kwargs)
        select_top_diverse_indices, sds_sampling_time = sampler.sample(unlabeled_dataloader, labeled_dataloader, pooling_kwargs)
        sampler = EntropySampler(budget, trainer, **kwargs)
        unlabeled_indice_list, uncertainty_list, uncertainty_sampling_time = sampler.get_uncertainty(unlabeled_dataloader)
        sampling_time = sds_sampling_time + uncertainty_sampling_time * len(select_top_diverse_indices) / len(unlabeled_dataloader)
        additional_start_time = time.time()
        top_diverse_position = [unlabeled_indice_list.index(idx) for idx in select_top_diverse_indices]
        top_uncertainty_scores = [uncertainty_list[pos] for pos in top_diverse_position]
        top_uncertainty_indices = np.argsort(top_uncertainty_scores)[-budget:][::-1]
        query_indices = [select_top_diverse_indices[idx] for idx in top_uncertainty_indices]
        additional_time = time.time() - additional_start_time
        uncertainty_values = []
    # We keep track of runtime
    save_to_logger(trainer.logger, 'metric', np.round(sampling_time/60, 2), 'sampling time (min)', trainer.current_epoch)
    save_to_logger(trainer.logger, 'metric', np.round((sampling_time + additional_time)/60), 'sampling total time (min)', trainer.current_epoch)
    save_to_logger(trainer.logger, 'list', query_indices, 'query_indices')

    # ========================Save all images, predictions and labels from the current selected samples===========================
    best_model_path = trainer.checkpoint_callback.best_model_path
    if strategy == 'LearningLoss':
        model = UNetLearningLoss.load_from_checkpoint(best_model_path)
    elif strategy == 'VAAL':
        model = UNetVAAL.load_from_checkpoint(best_model_path)
    else:
        model = UNet.load_from_checkpoint(best_model_path)
    model.to(kwargs.get('device'))
    model.eval()
    dateset = MyDataset(data_dir, dataset_name, type='train')
    # We iterate through the unlabeled dataloader
    for batch_idx, batch in enumerate(unlabeled_dataloader):
        with torch.no_grad():
            x, y, img_idx = batch['data'], batch['label'], batch['idx']
            for id, img_id in enumerate(img_idx):
                if img_id in query_indices:
                    x = x.to(kwargs.get('device'))
                    tran_img = (x[id].cpu().detach().numpy() * 255).transpose(1, 2, 0)
                    tran_img = np.clip(tran_img, 0, 255).astype(np.uint8)
                    cv2.imwrite(f'{budget_result_path}/img_{dateset.img_list[img_id][:3]}.png', tran_img)
                    cv2.imwrite(f'{budget_result_path}/label_{dateset.img_list[img_id][:3]}.png',
                                y.cpu().detach().numpy()[id][0] * 255)
                    logits, _ = model(x)
                    norm_output = normalize(normalize_fct='sigmoid', x=logits)
                    np_output = norm_output.cpu().detach().numpy()[id][1]
                    pred = (np_output > 0.5).astype(np.uint8)  # thresholding
                    cv2.imwrite(f'{budget_result_path}/pred_{dateset.img_list[img_id][:3]}.png', pred * 255)
    # ========================Perform qualitative comparison for each active learning cycle by saving  Test images Model predictions   Ground truth labels IoU metrics===========================
    dateset = MyDataset(data_dir, dataset_name, type='test')
    # We iterate through the unlabeled dataloader
    for batch_idx, batch in enumerate(test_dataloader):
        with torch.no_grad():
            x, y, img_id = batch['data'], batch['label'], batch['idx']
            x = x.to(kwargs.get('device'))
            tran_img = (x[0].cpu().detach().numpy() * 255).transpose(1, 2, 0)
            tran_img = np.clip(tran_img, 0, 255).astype(np.uint8)
            cv2.imwrite(f'{test_result_path}/img_{dateset.img_list[img_id][:3]}.png', tran_img)
            cv2.imwrite(f'{test_result_path}/label_{dateset.img_list[img_id][:3]}.png',
                        y.cpu().detach().numpy()[0][0] * 255)
            logits, _ = model(x)
            norm_output = normalize(normalize_fct='sigmoid', x=logits)
            np_output = norm_output.cpu().detach().numpy()[0][1]
            pred = (np_output > 0.5).astype(np.uint8)
            cv2.imwrite(f'{test_result_path}/pred_{dateset.img_list[img_id][:3]}.png', pred * 255)
            y_true = y.cpu().detach().numpy()[0][0] > 0.5  # binary
            iou = calculate_iou(y_true, pred)
            # add to metric.txt
            with open(f'{test_result_path}/metric.txt', 'a') as f:
                f.write(f"{dateset.img_list[img_id][:3]}: {iou:.4f}\n")

    return query_indices, sampling_time + additional_time, uncertainty_values


def SR_based_initial_set_indices(data_dir, dataset_name, init_budget):
    # SR-ILS for initial labeled set
    img_folder_path = os.path.join(data_dir, dataset_name, 'train', 'img')
    image_list = [f for f in os.listdir(img_folder_path) if f.endswith('.png')]
    image_list = sorted(image_list, key=lambda f: int(f.split('.')[0]))
    specular_reflection_areas = []
    if dataset_name == 'mg-203':
        dataset_len = 163
        for id in range(dataset_len):
            image_path = os.path.join(img_folder_path, image_list[id])
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = image.astype(np.float32) / 255.0
            # High-Pixel-Value Threshold Filtering
            threshold_value = 0.8  # Select an appropriate threshold
            _, binary_mask = cv2.threshold(image, threshold_value, 1, cv2.THRESH_BINARY)
            # Post-processing: Remove small noise points
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # Acquire specular reflection mask
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            specular_reflection_area = np.sum(binary_mask)
            specular_reflection_areas.append((id, specular_reflection_area))
        # Sort by descending area size and select top-k samples to form the initial labeled set
        sorted_areas = sorted(specular_reflection_areas, key=lambda x: x[1], reverse=True)[:init_budget]
        initial_indices = [id for id, area in sorted_areas]
        return initial_indices
    elif dataset_name == 'MGD-1K':
        train_index_path = os.path.join(data_dir, dataset_name, 'train.txt')
        upper_lid_areas = []
        lower_lid_areas = []
        with open(train_index_path, 'r') as f:
            train_id_list = [line.strip() for line in f.readlines()]
        for id in train_id_list:
            id = int(id)
            image_path = os.path.join(img_folder_path, image_list[id])
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # High-Pixel-Value Threshold Filtering
            image = image.astype(np.float32) / 255.0
            threshold_value = 0.60  # Select an appropriate threshold
            _, binary_mask = cv2.threshold(image, threshold_value, 1, cv2.THRESH_BINARY)
            # Post-processing: Remove small noise points
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            specular_reflection_area = np.sum(binary_mask)
            specular_reflection_areas.append((id, specular_reflection_area))
            if 0 <= id <= 275 or 534 <= id <= 790:
                lower_lid_areas.append((id, specular_reflection_area))
            else:
                upper_lid_areas.append((id, specular_reflection_area))
        # Sort by descending area size and select top-k samples to form the initial labeled set
        res_lower_lid_sorted = sorted(lower_lid_areas, key=lambda x: x[1], reverse=True)[:int(init_budget * 0.25)]
        res_upper_lid_sorted = sorted(upper_lid_areas, key=lambda x: x[1], reverse=True)[:int(init_budget * 0.25)]
        lower_lid_sorted = sorted(lower_lid_areas, key=lambda x: x[1])[:int(init_budget * 0.25)]
        upper_lid_sorted = sorted(upper_lid_areas, key=lambda x: x[1])[:int(init_budget * 0.25)]
        res_lower_lid_indices = [id for id, area in res_lower_lid_sorted]
        res_upper_lid_indices = [id for id, area in res_upper_lid_sorted]
        lower_lid_indices = [id for id, area in lower_lid_sorted]
        upper_lid_indices = [id for id, area in upper_lid_sorted]
        initial_indices = res_lower_lid_indices + res_upper_lid_indices + lower_lid_indices + upper_lid_indices
        return initial_indices


def RS_initial_set_indices(data_dir, dataset_name, init_budget, seed, step=5):
    # Randomly initialize the labeled set
    index_path = os.path.join(data_dir, dataset_name, 'train.txt')
    with open(index_path, 'r') as f:
        lines = f.readlines()
    selected_indices = [int(lines[i].strip()) for i in range(seed, seed + init_budget * step, step)]
    return selected_indices

def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    iou = intersection / (union + 1e-7)
    return iou
