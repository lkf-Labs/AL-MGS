import torch.nn as nn

import torch.nn.init as init

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch
import cv2
import torch.optim as optim
from itertools import cycle
from torchvision import datasets, transforms
from pytorch_lightning_spells.lr_schedulers import MultiStageScheduler
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

from Models.unet_modules import decoderUNet, encoderbottleneckUNet
from Utils.loss_utils import WCE_DiceLoss
from Utils.plot_utils import plot_data_pred_volume
from Utils.training_utils import (GradualWarmupScheduler, assd_metric,
                                  dice_metric, hausdorff95_metric,
                                  hausdorff_metric, iou_metric, accuracy_ignore_background)
from Utils.utils import to_onehot, normalize

"""
CODE BORROWED:
The following lines are based on code from the github repository:
https://github.com/sinhasam/vaal
Note class was renamed
"""
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class VAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, z_dim=32, nc=3, gpu_id=0):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.gpu_id = gpu_id
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024 * 22 * 46)),
        )

        # Latent space
        self.fc_mu = nn.Linear(1024 * 22 * 46, z_dim)
        self.fc_logvar = nn.Linear(1024 * 22 * 46, z_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024 * 22 * 46),
            View((-1, 1024, 22, 46)),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 4, 2, 1),
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x):
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            # nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean=0, std=0.02):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

"""
END OF BORROWED CODE
"""


class UNetVAAL(pl.LightningModule):
    def __init__(self,
                 per_device_batch_size: int = 1,
                 num_devices: int = 1,
                 lr: float = 1e-6,  # 1e-3,
                 weight_decay: float = 5e-4,  # 1e-4,
                 model_config: dict = {},
                 sched_config: dict = {},
                 loss_config: dict = {},
                 val_plot_slice_interval: int = 1,
                 seed=42,
                 labeled_dataloader = None,
                 unlabeled_dataloader = None,
                 devices = 0,
                 **kwargs
                 ):
        """
        This modified UNet differs from the original one with the use of
        leaky relu (instead of ReLU) and the addition of residual connections.
        The idea is to help deal with fine-grained details
        :param in_channels: # of input channels (ie: 3 if  image in RGB)
        :param out_channels: # of output channels (# segmentation classes)
        """
        super().__init__()
        self.save_hyperparameters()
        self.per_device_batch_size = per_device_batch_size
        self.num_devices = num_devices
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.sched_config = sched_config
        self.tarin_iteration = 0

        self.in_channels = model_config["in_channels"]
        self.out_channels = model_config["out_channels"]
        self.channel_list = model_config["channel_list"]  # channel_list=[64, 128, 256, 512]

        self.encoderbottleneck = encoderbottleneckUNet(model_config)
        self.decoder = decoderUNet(model_config)

        if loss_config != {}:
            self.model_loss = WCE_DiceLoss(**loss_config)

        # vaal models
        self.vaal_params = kwargs
        self.vae = VAE(self.vaal_params['latent_dim'], gpu_id=devices[0])
        self.discriminator = Discriminator(self.vaal_params['latent_dim'])
        self.optim_vae = optim.Adam(self.vae.parameters(), lr=1e-6)
        self.optim_discriminator = optim.Adam(self.discriminator.parameters(), lr=1e-6)
        self.labeled_dataloader = labeled_dataloader
        self.unlabeled_dataloader = unlabeled_dataloader
        self.gpu_id = devices[0]
        self.vae.cuda(self.gpu_id)
        self.discriminator.cuda(self.gpu_id)
        self.num_vae_steps = self.vaal_params['num_vae_steps']
        self.num_adv_steps = self.vaal_params['num_adv_steps']
        self.beta = self.vaal_params['beta']
        self.adversary_param = self.vaal_params['adversary_param']
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.labeled_data = cycle(self.labeled_dataloader)
        self.unlabeled_data = cycle(self.unlabeled_dataloader)

        # Plot params
        self.log_metric_freq = 5
        self.log_img_freq = 1800  # Must be multiple of self.log_metric_freq
        self.val_plot_slice_interval = val_plot_slice_interval
        self.plot_type = 'contour' if self.out_channels == 2 else 'image'

    def forward(self, x):
        # Encoding
        center, [enc_1, enc_2, enc_3] = self.encoderbottleneck(x)
        out, [dec_1, dec_2, dec_3] = self.decoder(center, enc_1, enc_2, enc_3)
        return out, [enc_1, enc_2, enc_3, center, dec_1, dec_2, dec_3]

    def _compute_loss(self, logits, y):
        onehot_target = to_onehot(y.squeeze(1), self.out_channels)
        loss = self.model_loss(logits, onehot_target)
        return loss

    def _compute_metrics(self, logits, y, pred_dim=1):
        pred = torch.argmax(logits, dim=pred_dim)
        onehot_pred = to_onehot(pred.squeeze(1), self.out_channels)
        onehot_target = to_onehot(y.squeeze(1), self.out_channels)
        _dice = dice_metric(onehot_pred, onehot_target)
        dice = torch.mean(_dice)
        hausdorff95 = torch.mean(hausdorff95_metric(onehot_pred, onehot_target))
        _iou = iou_metric(onehot_pred, onehot_target)
        iou = torch.mean(_iou)
        acc = accuracy_ignore_background(onehot_pred, onehot_target)
        hausdorff = torch.mean(hausdorff_metric(onehot_pred, onehot_target))
        assd = torch.mean(assd_metric(onehot_pred, onehot_target))
        return 100 * dice, 100 * iou, acc * 100, hausdorff95, hausdorff, assd

    def _training_step(self, batch, batch_idx):
        """ Contains all computations necessary to produce a loss value to train the model"""
        x, y, img_idx = batch['data'], batch['label'], batch['idx']
        logits, _ = self(x)
        loss = self._compute_loss(logits, y)
        dice, iou, _, _, _, _ = self._compute_metrics(logits, y)

        if self.current_epoch % self.log_metric_freq == 0:
            self.log('train/loss', loss)
            self.log('train/dice', dice)
            self.log('train/iou', iou)

        sample = next(self.labeled_data)
        labeled_imgs, labels = sample['data'], sample['label']
        sample = next(self.unlabeled_data)
        unlabeled_imgs = sample['data']
        labeled_imgs = labeled_imgs.cuda(self.gpu_id)
        unlabeled_imgs = unlabeled_imgs.cuda(self.gpu_id)
        labels = labels.cuda(self.gpu_id)
        # VAE step
        for count in range(self.num_vae_steps):
            recon, z, mu, logvar = self.vae(labeled_imgs)
            unsup_loss = self.vae_loss(labeled_imgs, recon, mu, logvar, self.beta)
            unlab_recon, unlab_z, unlab_mu, unlab_logvar = self.vae(unlabeled_imgs)
            transductive_loss = self.vae_loss(unlabeled_imgs,
                                              unlab_recon, unlab_mu, unlab_logvar, self.beta)

            labeled_preds = self.discriminator(mu)
            unlabeled_preds = self.discriminator(unlab_mu)

            lab_real_preds = torch.ones_like(labeled_preds)
            unlab_real_preds = torch.ones_like(unlabeled_preds)

            lab_real_preds = lab_real_preds.cuda(self.gpu_id)
            unlab_real_preds = unlab_real_preds.cuda(self.gpu_id)

            dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
                       self.bce_loss(unlabeled_preds, unlab_real_preds)
            total_vae_loss = unsup_loss + transductive_loss + self.adversary_param * dsc_loss
            self.optim_vae.zero_grad()
            total_vae_loss.backward()
            self.optim_vae.step()

            # sample new batch if needed to train the adversarial network
            if count < (self.num_vae_steps - 1):
                sample = next(self.labeled_data)
                labeled_imgs, labels = sample['data'], sample['label']
                sample = next(self.unlabeled_data)
                unlabeled_imgs = sample['data']

                labeled_imgs = labeled_imgs.cuda(self.gpu_id)
                unlabeled_imgs = unlabeled_imgs.cuda(self.gpu_id)
                labels = labels.cuda(self.gpu_id)

        # Discriminator step
        for count in range(self.num_adv_steps):
            with torch.no_grad():
                _, _, mu, _ = self.vae(labeled_imgs)
                _, _, unlab_mu, _ = self.vae(unlabeled_imgs)

            labeled_preds = self.discriminator(mu)
            unlabeled_preds = self.discriminator(unlab_mu)

            lab_real_preds = torch.ones_like(labeled_preds)
            unlab_fake_preds = torch.zeros_like(unlabeled_preds)

            lab_real_preds = lab_real_preds.cuda(self.gpu_id)
            unlab_fake_preds = unlab_fake_preds.cuda(self.gpu_id)

            dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
                       self.bce_loss(unlabeled_preds, unlab_fake_preds)

            self.optim_discriminator.zero_grad()
            dsc_loss.backward()
            self.optim_discriminator.step()

            # sample new batch if needed to train the adversarial network
            if count < (self.num_adv_steps - 1):
                sample = next(self.labeled_data)
                labeled_imgs, labels = sample['data'], sample['label']
                sample = next(self.unlabeled_data)
                unlabeled_imgs = sample['data']

                labeled_imgs = labeled_imgs.cuda(self.gpu_id)
                unlabeled_imgs = unlabeled_imgs.cuda(self.gpu_id)
                labels = labels.cuda(self.gpu_id)


        return loss

    def _validation_step(self, batch, batch_idx):
        x, y, img_idx = batch['data'], batch['label'], batch['idx']
        logits, _ = self(x)
        loss = self._compute_loss(logits, y)
        dice, iou, _, _, _, _ = self._compute_metrics(logits, y)
        print(f"*******************************************val_loss = {loss}  and   val_iou = {iou}")
        self.log('val/loss', loss)
        self.log('val/dice', dice)
        self.log('val/iou', iou)


        return loss

    def _test_step(self, batch, batch_idx):
        x, y, img_idx = batch['data'], batch['label'], batch['idx']
        logits, _ = self(x)
        loss = self._compute_loss(logits, y)
        dice, iou, acc, hd95, hd, assd = self._compute_metrics(logits, y)
        self.log('test/loss', loss)
        self.log('test/dice', dice)
        self.log('test/iou', iou)
        self.log('test/acc', acc)
        self.log('test/hd95', hd95)
        self.log('test/hd', hd)
        self.log('test/assd', assd)
        print('******************************test_step_iou******************************** = ', iou)

        return loss

    def training_step(self, batch, batch_idx):
        # If logger is comet_ml
        if isinstance(self.logger, pl.loggers.CometLogger):
            with self.logger.experiment.train():
                return self._training_step(batch, batch_idx)
        # If logger is the default TensorBoardLogger or another logger
        else:
            return self._training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        # If logger is comet_ml
        if isinstance(self.logger, pl.loggers.CometLogger):
            with self.logger.experiment.validate():
                return self._validation_step(batch, batch_idx)
        # If logger is the default TensorBoardLogger or another logger
        else:
            return self._validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        # If logger is comet_ml
        if isinstance(self.logger, pl.loggers.CometLogger):
            with self.logger.experiment.test():
                return self._test_step(batch, batch_idx)
        # If logger is the default TensorBoardLogger or another logger
        else:
            return self._test_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        base_sched = CosineAnnealingLR(optimizer,
                                       T_max=self.sched_config["max_epoch"] - self.sched_config["warmup_max"],
                                       eta_min=1e-7)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=self.sched_config['multiplier'],
                                           total_epoch=self.sched_config["warmup_max"],
                                           after_scheduler=base_sched)
        scheduler = MultiStageScheduler(schedulers=[scheduler],
                                        start_at_epochs=[0])

        scheduler = {
            "scheduler": scheduler,
            "interval": self.sched_config["update_interval"],
            "frequency": self.sched_config["update_freq"],
        }

        return [optimizer], [scheduler]


    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, _ in dataloader:
                    yield img, label
        else:
            while True:
                for img, _, _ in dataloader:
                    yield img


    def vae_loss(self, x, recon, mu, logvar, beta):
        MSE = self.mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD


