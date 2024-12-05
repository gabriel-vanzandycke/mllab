import abc
import torch
import torch.nn as nn
import torchvision.models as models

from mllab.base import Scope


class Operation(nn.Module, abc.ABC):
    scope = Scope.ALL
    @abc.abstractmethod
    def forward(self, batch):
        raise NotImplementedError
    def __repr__(self):
        return self.__class__.__name__

class Lambda(Operation):
    def __init__(self, l):
        super().__init__()
        self.l = l
    def forward(self, batch):
        self.l(batch)


class Permute(Operation):
    def __init__(self, name='input'):
        super().__init__()
        self.name = name
    def forward(self, batch):
        if len(batch[self.name].shape) == 4:
            batch[self.name] = batch[self.name].permute(0, 3, 1, 2).contiguous()


class MaskLoss(Operation):
    def forward(self, batch):
        batch['loss'] = batch['loss'] * ~batch['mask']


class NormalizeInput(Operation):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()  # Initialize nn.Module
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    def forward(self, batch):
        if batch['input'].dtype == torch.uint8:
             batch['input'] = batch['input'].float() / 255.0
        batch['input'] = (batch['input'] - self.mean) / self.std


class MobilenetV3Backbone(Operation):
    def __init__(self, num_classes):
        super().__init__()  # Initialize nn.Module
        self.model = models.segmentation.deeplabv3_mobilenet_v3_large(num_classes=num_classes)
    def forward(self, batch):
        batch["logits"] = self.model(batch["input"])['out']


class SumLosses(Operation):
    def __init__(self, loss_weights):
        super().__init__()  # Initialize nn.Module
        self.loss_weights = loss_weights
    def forward(self, batch):
        batch['loss'] = sum([batch[name]*w for name, w in self.loss_weights.items() if name in batch])


class ComputeLoss(Operation):
    def __init__(self, criterion):
        super().__init__()  # Initialize nn.Module
        self.criterion = criterion
    def forward(self, batch):
        batch["loss"] = self.criterion(batch["logits"], batch["target"])

class ReduceLoss(Operation):
    def forward(self, batch):
        batch["loss_map"] = batch["loss"]
        batch["loss"] = batch["loss"].mean()

class TargetToLongOperation(Operation):
    def forward(self, batch):
        if batch['logits'].shape[1] == 2:
            batch['target'] = torch.any(batch['target'], dim=1).long()
        else:
            batch["target"] = batch["target"].float()

class TorchVisionDataAugmentation(Operation):
    scope = Scope.TRAIN
    def __init__(self, *transforms):
        super().__init__()  # Initialize nn.Module
        self.transforms = transforms
    def forward(self, batch):
        for transform in self.transforms:
            batch['input'] = transform(batch['input'])
