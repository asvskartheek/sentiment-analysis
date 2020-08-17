import pytorch_lightning as pl
import torch
from torch import optim

from utils import bce_loss_with_logits, binary_accuracy


class Bare(pl.LightningModule):
    """
    The bare bone lightning model, any sub-class forward should return output: [b * o]
    """

    def __init__(self, hparams, *args, **kwargs):
        super().__init__()

        self.hparams = hparams

    def forward(self, x):
        return torch.Tensor()  # dummy

    def configure_optimizers(self):
        learning_rate = self.hparams.lr
        optim_name = self.hparams.optimizer

        if optim_name=='adam':
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optim_name == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        return optimizer

    def training_step(self, batch, batch_idx):
        examples = batch.text
        labels = batch.label
        logits = self.forward(examples).squeeze(1)
        loss = bce_loss_with_logits(logits, labels)
        acc = binary_accuracy(logits, labels)

        logs = {'train_loss': loss, 'train_acc': acc}
        return {'loss': loss, 'acc': acc, 'log': logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        loss = results['loss']
        acc = results['acc']
        logs = {'valid_loss': loss, 'valid_acc': acc}
        return {'loss': loss, 'acc': acc, 'progress_bar': logs}

    def test_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        loss = results['loss']
        acc = results['acc']
        logs = {'test_loss': loss, 'test_acc': acc}
        return {'loss': loss, 'acc': acc, 'progress_bar': logs}

    def validation_epoch_end(self, outputs):
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'avg_val_loss': avg_loss, 'avg_val_acc': avg_acc, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': avg_acc}
        return {'avg_test_loss': avg_loss, 'avg_test_acc': avg_acc, 'log': tensorboard_logs}

    def transfer_batch_to_device(self, batch, device):
        text = batch.text[0].to(device)
        text_lens = batch.text[1].to(device)
        batch.text = (text, text_lens)
        batch.label = batch.label.to(device)
        return batch
