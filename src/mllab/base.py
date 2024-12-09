
from enum import IntFlag
import os

from lightning.pytorch import LightningModule
from torch.utils.data import DataLoader
import torch.nn as nn

from astconfig import DictObject

from mllab.utils import TorchDataset
from torch.profiler import record_function


MAX_WORKERS = 8

class Scope(IntFlag):
    TRAIN   = 1
    VAL     = 2
    TEST    = 4
    PREDICT = 8
    ALL     = -1


class PyConfigModule(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = DictObject(config)
        self.operations = nn.ModuleList(self.config.operations) # Modules should be initialized in LightningModule __init__

        self.training_step   = StepFunctionFactory(self, Scope.TRAIN, list(self.config.subsets['train'].keys()))
        self.validation_step = StepFunctionFactory(self, Scope.VAL,   list(self.config.subsets['val'].keys()))
        self.test_step       = StepFunctionFactory(self, Scope.TEST,  list(self.config.subsets['test'].keys()))

        self.train_dataloader   = DataLoadersFactory(self, Scope.TRAIN, self.config.subsets['train'])
        self.val_dataloader     = DataLoadersFactory(self, Scope.VAL,   self.config.subsets['val'])
        self.test_dataloader    = DataLoadersFactory(self, Scope.TEST,  self.config.subsets['test'])
        self.predict_dataloader = DataLoadersFactory(self, Scope.PREDICT,  self.config.subsets['test'])

        #config = {k:v for k,v in config.items() if k not in ['operations']} # operations fails when using a profiler
        self.save_hyperparameters(config)

    @property
    def metrics(self):
        return ['loss']

    def forward(self, batch, scope):
        for operation in self.operations:
            with record_function(operation.__class__.__name__):
                if operation is not None and operation.scope & scope:
                    operation(batch)
        return batch

    def predict_step(self, batch, batch_idx):
        return self(batch, scope=Scope.PREDICT)

    def configure_optimizers(self):
        return self.config.optimizer(self.parameters(), **self.config.optimizer_kwargs)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return LazyBatch(batch, device=device)

    # def on_load_checkpoint(self, checkpoint): (called after __init__ on load_from_checkpoint)
    #    return super().on_load_checkpoint(checkpoint)
    # def on_save_checkpoint(self, checkpoint):
    #     return super().on_save_checkpoint(checkpoint)

class DataLoadersFactory():
    __code__ = None
    def __init__(self, module, scope, subsets):
        self.module = module
        self.subsets = subsets
        self.num_workers = self.module.config.get('num_workers', min((os.cpu_count() or 1) - 1, MAX_WORKERS))
        self.batch_size = self.module.config.get('batch_size') if scope == Scope.TRAIN else 1
        self.is_training = scope == Scope.TRAIN
    def __call__(self, sampler=None, batch_size=None, **kwargs):
        batch_size = batch_size or self.batch_size
        dataloaders = [
            DataLoader(
                TorchDataset(subset),
                sampler=sampler,
                batch_size=batch_size,
                shuffle=self.is_training,
                drop_last=self.is_training,
                num_workers=self.num_workers,
                persistent_workers=True if self.num_workers else False,
            ) for _, subset in self.subsets.items()
        ]
        return dataloaders[0] if len(dataloaders) == 1 else dataloaders

class StepFunctionFactory:
    __code__ = None
    def __init__(self, module, scope, cycle_names):
        self.module = module
        self.scope = scope
        self.cycle_names = cycle_names
    def __call__(self, batch, batch_idx, dataloader_idx=0):
        batch = self.module(batch, scope=self.scope)
        name = self.cycle_names[dataloader_idx]
        batch_size = batch['input'].shape[0]
        self.module.log_dict({
            f'{name}_{metric}': batch.get(metric) for metric in self.module.metrics
        }, add_dataloader_idx=False, on_epoch=True, on_step=False, batch_size=batch_size)
        return batch


class LazyBatch(dict):
    def __init__(self, batch, device=None):
        super().__init__(batch)
        self.device = device
    def __getitem__(self, name):
        return super().__getitem__(name).to(self.device)
