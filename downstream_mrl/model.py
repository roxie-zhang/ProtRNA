import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR

import lightning.pytorch as pl

from torchmetrics.regression import R2Score

from downstream_mrl.network.scaler import StandardScaler
from downstream_mrl.network.downstream import RibosomeLoadingPredictionHead


class RibosomeLoadingPredictionWrapper(pl.LightningModule):
    def __init__(
        self,
        lm_config: str = "giga",
        head_embed_dim: int = 32,
        head_num_blocks: int = 6,
        lr: float = 1e-3,
        lm_type: str = 'rinalmo',
    ):
        super().__init__()
        self.save_hyperparameters()

        self.scaler = StandardScaler()
        
        self.lm_type = lm_type

        if lm_type == 'rinalmo' or lm_type == 'ProtRNA':
            pred_head_c_in = 1280
        elif lm_type == 'RNA_FM':
            pred_head_c_in = 640
        else:
            pred_head_c_in = 768

        self.pred_head = RibosomeLoadingPredictionHead(
            c_in=pred_head_c_in,
            embed_dim=head_embed_dim,
            num_blocks=head_num_blocks
        )

        self.loss = nn.MSELoss()
        self.r2_metric = R2Score()

        self.lr = lr
        
        # self.pad_idx = 1
        
    def forward(self, pad_mask, vector):
        x = vector
        x[pad_mask, :] = 0.0

        pred = self.pred_head(x, pad_mask)
        return pred

    def fit_scaler(self, batch):
        _, _, rl = batch
        self.scaler.partial_fit(rl)

    def _common_step(self, batch, batch_idx, log_prefix: str):

        pad_mask, vector, rl_target = batch
        preds = self(pad_mask, vector)

        scaled_rl_target = self.scaler.transform(rl_target)
        loss = self.loss(preds, scaled_rl_target)

        preds = self.scaler.inverse_transform(preds).clamp(min=0.0) # "Unscale" predictions
        mse = F.mse_loss(preds, rl_target)
        mae = F.l1_loss(preds, rl_target)
        self.r2_metric.update(preds, rl_target)

        log = {
            f'{log_prefix}/loss': loss,
            f'{log_prefix}/mse': mse,
            f'{log_prefix}/mae': mae,
        }
        self.log_dict(log, sync_dist=True)

        return loss

    def _eval_step(self, batch, batch_idx, log_prefix):
        return self._common_step(batch, batch_idx, log_prefix=log_prefix)

    def _on_eval_epoch_start(self):
        # Reset metric calculator
        self.r2_metric.reset()

    def _on_eval_epoch_end(self, log_prefix: str):
        # Log and reset metric calculator
        if not self.trainer.sanity_checking:
            self.log(f"{log_prefix}/r2", self.r2_metric.compute(), sync_dist=True)
            self.r2_metric.reset()

    def training_step(self, batch, batch_idx):
        if self.current_epoch == 0:
            return self.fit_scaler(batch)

        return self._common_step(batch, batch_idx, log_prefix="train")

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, log_prefix="val")
    
    def on_validation_epoch_start(self):
        return self._on_eval_epoch_start()

    def on_validation_epoch_end(self):
        return self._on_eval_epoch_end("val")
    
    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, log_prefix="test")
    
    def on_test_epoch_start(self):
        return self._on_eval_epoch_start()

    def on_test_epoch_end(self):
        return self._on_eval_epoch_end("test")

    def configure_optimizers(self):
        optimizer = Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=5000) # TODO: Currently hardcoded!

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }