import torch

import numpy as np

class AdversarySampler:
    def __init__(self, budget, trainer, **kwargs):
        self.budget = budget
        self.trainer = trainer
        self.vae = self.trainer.model.vae
        self.discriminator = self.trainer.model.discriminator
        self.device = kwargs.get('device')
        self.vae.to(self.device)
        self.discriminator.to(self.device)

    def sample(self, data):
        all_preds = []
        all_indices = []
        for sample in data:
            images = sample['data']
            indices = sample['idx']
            images = images.to(self.device)
            with torch.no_grad():
                _, _, mu, _ = self.vae(images)
                preds = self.discriminator(mu)
                preds = torch.sigmoid(preds)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1
        print(all_preds)
        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]

        return querry_pool_indices.tolist()
        
