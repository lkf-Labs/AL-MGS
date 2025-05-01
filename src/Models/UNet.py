
import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch
import cv2
import torch.optim as optim
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


class UNet(pl.LightningModule):
    def __init__(self, 
                 per_device_batch_size: int = 1,
                 num_devices: int = 1,
                 lr: float = 1e-6, #1e-3,
                 weight_decay: float = 5e-4, #1e-4,
                 model_config: dict = {},
                 sched_config: dict = {},
                 loss_config: dict = {},
                 val_plot_slice_interval: int = 1,
                 seed = 42,
                 checkpoint_path = None
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
        self.tarin_iteration = 3600

        self.in_channels = model_config["in_channels"]
        self.out_channels = model_config["out_channels"]
        self.channel_list = model_config["channel_list"] # channel_list=[64, 128, 256, 512]
        
        self.encoderbottleneck = encoderbottleneckUNet(model_config)
        self.decoder = decoderUNet(model_config)
        # self.save_img_result_path = os.path.join(checkpoint_path, 'result')
        # os.makedirs(self.save_img_result_path, exist_ok=True)
        if loss_config != {}:
            self.model_loss = WCE_DiceLoss(**loss_config)

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
        return 100*dice, 100*iou, acc*100, hausdorff95, hausdorff, assd

    def _training_step(self, batch, batch_idx):
        """ Contains all computations necessary to produce a loss value to train the model"""
        x, y, img_idx = batch['data'], batch['label'], batch['idx']
        logits, _ = self(x)
        loss = self._compute_loss(logits, y)
        dice, iou, _, _, _, _  = self._compute_metrics(logits, y)

        if self.current_epoch % self.log_metric_freq == 0:
            self.log('train/loss', loss)
            self.log('train/dice', dice)
            self.log('train/iou', iou)

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
                return  self._training_step(batch, batch_idx)
        # If logger is the default TensorBoardLogger or another logger
        else:
            return self._training_step(batch, batch_idx)
        
    def validation_step(self, batch, batch_idx):
        # If logger is comet_ml
        if isinstance(self.logger, pl.loggers.CometLogger):
            with self.logger.experiment.validate():
                return  self._validation_step(batch, batch_idx)
        # If logger is the default TensorBoardLogger or another logger
        else:
            return self._validation_step(batch, batch_idx)
        
    def test_step(self, batch, batch_idx):
        # If logger is comet_ml
        if isinstance(self.logger, pl.loggers.CometLogger):
            with self.logger.experiment.test():
                return  self._test_step(batch, batch_idx)
        # If logger is the default TensorBoardLogger or another logger
        else:
            return self._test_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        base_sched = CosineAnnealingLR(optimizer, T_max=self.sched_config["max_epoch"] - self.sched_config["warmup_max"], 
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
    
    

