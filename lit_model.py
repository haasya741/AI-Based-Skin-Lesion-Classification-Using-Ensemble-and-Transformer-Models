import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from sklearn.metrics import f1_score, balanced_accuracy_score
import numpy as np


class LitEnsembleModel(pl.LightningModule):
    def __init__(self, ensemble_model, loss_fn, lr, weight_decay):
        super().__init__()
        self.model = ensemble_model
        self.loss_fn = loss_fn
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Store predictions and targets for validation metrics
        self.val_preds = []
        self.val_targets = []

    def forward(self, x_img, x_meta=None):
        return self.model(x_img, x_meta)

    def training_step(self, batch, batch_idx):
        # Unpack batch - handle both with and without metadata
        if len(batch) == 3:
            x_img, x_meta, y = batch
        else:
            x_img, y = batch
            x_meta = None
            
        y_hat = self(x_img, x_meta)
        loss = self.loss_fn(y_hat, y)
        
        # Log training loss
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch - handle both with and without metadata
        if len(batch) == 3:
            x_img, x_meta, y = batch
        else:
            x_img, y = batch
            x_meta = None
            
        y_hat = self(x_img, x_meta)
        loss = self.loss_fn(y_hat, y)
        
        # Get predictions
        preds = torch.argmax(y_hat, dim=1)
        
        # Store predictions and targets for epoch-end metrics
        self.val_preds.append(preds.cpu())
        self.val_targets.append(y.cpu())
        
        # Log validation loss
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        if len(self.val_preds) > 0 and len(self.val_targets) > 0:
            # Concatenate all predictions and targets
            preds = torch.cat(self.val_preds).numpy()
            targets = torch.cat(self.val_targets).numpy()

            # Calculate metrics
            f1 = f1_score(targets, preds, average='macro')
            bal_acc = balanced_accuracy_score(targets, preds)

            # Log metrics
            self.log('val_macro_f1', f1, prog_bar=True, on_epoch=True)
            self.log('val_bal_acc', bal_acc, prog_bar=True, on_epoch=True)
            
            print(f"Validation F1: {f1:.4f}, Balanced Accuracy: {bal_acc:.4f}")

        # Clear stored outputs for next epoch
        self.val_preds.clear()
        self.val_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        return optimizer