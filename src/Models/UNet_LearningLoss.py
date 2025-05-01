import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_lightning_spells.lr_schedulers import MultiStageScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from Models.unet_modules import decoderUNet, encoderbottleneckUNet
from Utils.loss_utils import WCE_DiceLoss
from Utils.plot_utils import plot_data_pred_volume
from Utils.training_utils import (GradualWarmupScheduler, dice_metric, hausdorff95_metric, hausdorff_metric, iou_metric, accuracy_ignore_background, assd_metric)
from Utils.utils import to_onehot


"""
CODE BORROWED:
The following lines are based on code from the github repository:
https://github.com/Mephisto405/Learning-Loss-for-Active-Learning (as of May 31st 2021)
Excerpts are taken from the resnet.py file of the model folder
Note class was renamed
"""
class LossPredictionModule(nn.Module):
    def __init__(self, feature_sizes=[128, 64, 32, 16], num_channels=[64, 128, 256, 512],
                 interm_dim=128):
        super(LossPredictionModule, self).__init__()

        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)

        self.linear = nn.Linear(4 * interm_dim, 1)

    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out
"""
END OF BORROWED CODE
"""


"""
CODE BORROWED:
The following lines are based on code from the github repository:
https://github.com/Mephisto405/Learning-Loss-for-Active-Learning (as of May 31st 2021)
Excerpts are taken from the main.py file.
"""
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape

    # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    input = (input - input.flip(0))[:len(input)//2]
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    # This is an operation which is defined by the authors. 
    # Yields +1 if diff. between 2 targets is positive, -1 if it is negative or 0
    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        # Note that the size of input is already halved
        loss = loss / input.size(0)
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss
"""
END OF BORROWED CODE
"""


class UNetLearningLoss(pl.LightningModule):
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
                 **kwargs
                 ):
        """
        This modified UNet differs from the original one with the use of
        leaky relu (instead of ReLU).
        The idea is to help deal with fine-grained details
        """
        super().__init__()
        
        # Important: This property activates manual optimization.
        self.automatic_optimization = False
        
        self.save_hyperparameters()
        self.per_device_batch_size = per_device_batch_size
        self.num_devices = num_devices
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.sched_config = sched_config
        self.module_params = kwargs

        self.in_channels = model_config["in_channels"]
        self.out_channels = model_config["out_channels"]
        self.channel_list = model_config["channel_list"] # channel_list=[64, 128, 256, 512]
        
        self.encoderbottleneck = encoderbottleneckUNet(model_config)
        self.decoder = decoderUNet(model_config)
        # self.loss_module = LossPredictionModule(feature_sizes=self.module_params['lossmodule_feature_size'])
        self.loss_module = LossPredictionModule(feature_sizes=[(352, 736), (176, 368), (88, 184), (44, 92)])

        if loss_config != {}:
            self.model_loss = WCE_DiceLoss(**loss_config)

        # Additional parameters for learning loss
        self.module_weight = self.module_params['loss_module_weight']
        self.module_margin = self.module_params['loss_module_margin']
        self.epoch_stop_module_gradient = self.module_params['loss_module_epoch_stop_gradient']

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
        opt_model, opt_module = self.optimizers()
        opt_model.zero_grad()
        opt_module.zero_grad()
        
        x, y, img_idx = batch['data'], batch['label'], batch['idx']
        
        logits, [enc1, enc2, enc3, center, _, _, _]  = self(x)

        # We compute the model loss
        model_loss = self._compute_loss(logits, y)
        avg_model_loss = torch.mean(model_loss)   # average loss over all samples in batch
        dice, iou, _, _, _, _ = self._compute_metrics(logits, y)

        # We compute the loss of the loss prediction module
        features = [enc1, enc2, enc3, center]
        if self.current_epoch > self.epoch_stop_module_gradient:
            # After 120 epochs, the gradient from the loss prediction module stops being propagated to the target model.
            for i in range(len(features)):
                features[i] = features[i].detach()
        pred_loss = self.loss_module(features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        module_loss = LossPredLoss(pred_loss, model_loss, self.module_margin)
        loss = avg_model_loss + self.module_weight * module_loss
        
        # We do back propagation
        self.manual_backward(loss)
        opt_model.step()
        opt_module.step()
        
        # We do a scheduler step every epoch
        sch_model, sch_module = self.lr_schedulers()
        # step every N epochs
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0:
            sch_model.step()
            sch_module.step()

        if self.current_epoch % self.log_metric_freq == 0:
            self.log('train/loss', loss)
            self.log('train/dice', dice)
            self.log('train/iou', iou)
   
        return loss

    def _validation_step(self, batch, batch_idx):
        x, y, img_idx = batch['data'], batch['label'], batch['idx']
        
        logits, [enc1, enc2, enc3, center, _, _, _]= self(x)
        
        # We compute the model loss
        model_loss = self._compute_loss(logits, y)
        loss = torch.mean(model_loss)   # average loss over all samples in batch
        dice, iou, _, _, _, _ = self._compute_metrics(logits, y)

        # We compute the loss of the loss prediction module
        features = [enc1, enc2, enc3, center]
        pred_loss = self.loss_module(features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        # Because Learning loss loss computation requires batch size divisible by 2, we will not take into account last batch if not full sized
        if x.shape[0] == self.per_device_batch_size:
            module_loss = LossPredLoss(pred_loss, model_loss, self.module_margin)
            loss = loss + self.module_weight * module_loss
            if self.current_epoch % self.log_metric_freq == 0:
                self.log('val/total_loss', loss)
                self.log('val/module_loss', module_loss)

        print(f"*******************************************val_loss = {loss}  and   val_iou = {iou}")
        self.log('val/loss', loss)
        self.log('val/dice', dice)
        self.log('val/iou', iou)

        return loss

    def _test_step(self, batch, batch_idx):
        x, y, img_idx = batch['data'], batch['label'], batch['idx']
        logits, _ = self(x)
        model_loss = self._compute_loss(logits, y)
        loss = torch.mean(model_loss)
        dice, iou, acc, hd95, hd, assd = self._compute_metrics(logits, y)
        self.log('test/loss', loss)
        self.log('test/dice', dice)
        self.log('test/iou', iou)
        self.log('test/acc', acc)
        self.log('test/hd95', hd95)
        self.log('test/hd', hd)
        self.log('test/assd', assd)
                
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

    def lr_scheduler_step(self, scheduler, optimizer, metric):
        """
        Manually step the LR scheduler after every epoch
        """
        scheduler.step()

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
        
        # We define optimizer and scheduler for the learning loss module
        optimizer_module = optim.SGD(self.loss_module.parameters(),
                                     lr=self.module_params['loss_module_init_lr'],
                                     momentum=self.module_params['loss_module_sched_momentum'],
                                     weight_decay=self.module_params['loss_module_sched_wdecay'])
        scheduler_module = MultiStepLR(optimizer_module,
                                                    milestones=self.module_params['loss_module_sched_freq'])

        return [optimizer, optimizer_module], [scheduler, scheduler_module]
    
    

