

from enum import IntFlag
import types

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn as nn

from astconfig import DictObject

from .utils import TorchDataset


class Scope(IntFlag):
    TRAIN   = 1
    EVAL    = 2
    PREDICT = 4
    ALL     = -1


class PyConfigModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = DictObject(config)
        self.operations = nn.ModuleList(self.config.operations) # Modules should be initialized in LightningModule __init__
        self.training_step   = self.step_factory('train', Scope.TRAIN)
        self.validation_step = self.step_factory('val')
        self.testing_step    = self.step_factory('test')
        config = {k:v for k,v in config.items() if not isinstance(v, types.ModuleType)}
        self.save_hyperparameters(config)

    def __gestate__(self):
        return (self.config, )

    def __setstate__(self, state):
        config, = state
        self.__init__(config)

    def step_factory(self, prefix, scope=Scope.EVAL):
        cycle_names = list(self.config.subsets[prefix].keys())
        def step_function(batch, batch_idx, dataloader_idx=0):
            batch = self(batch, scope=scope)
            name = cycle_names[dataloader_idx]
            batch_size = batch['input'].shape[0]
            for metric in self.metrics:
                if metric in batch:
                    self.log(f'{name}_{metric}', batch[metric], add_dataloader_idx=False, on_epoch=True, on_step=False, batch_size=batch_size)
            return batch
        return step_function

    @property
    def metrics(self):
        return ['loss']

    def forward(self, batch, scope):
        for operation in self.operations:
            if operation.scope & scope:
                operation(batch)
        return batch

    def predict_step(self, batch, batch_idx):
        return self(batch, scope=Scope.PREDICT)

    def configure_optimizers(self):
        return self.config.optimizer(self.parameters(), **self.config.optimizer_kwargs)


class PyConfigDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def __gestate__(self):
        return (self.config, )

    def __setstate__(self, state):
        config, = state
        self.__init__(config)

    def train_dataloader(self):
        return DataLoader(TorchDataset(self.config.subsets["train"]["training"]), batch_size=self.config.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self, sampler=None):
        return {
            name: DataLoader(TorchDataset(subset), sampler=sampler) for name, subset in self.config.subsets['val'].items()
        }

