import abc
import torch
import torch.nn as nn
import torchvision.models as models

from mllab.base import Scope


class Operation(nn.Module, abc.ABC):
    scope = Scope.ALL
    def __init__(self):
        super().__init__()  # Initialize nn.Module
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    @abc.abstractmethod
    def forward(self, batch):
        raise NotImplementedError

class Lambda(Operation):
    def __init__(self, l):
        super().__init__()
        self.l = l
    def forward(self, batch):
        self.l(batch)

class PermuteOperation(Operation):
    def forward(self, batch):
        for name in batch:
            if len(batch[name].shape) == 4:
                batch[name] = batch[name].permute(0, 3, 1, 2).contiguous()


class MaskLoss(Operation):
    def forward(self, batch):
        batch['loss'] = batch['loss'] * ~batch['mask']


class NormalizeOperation(Operation):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()  # Initialize nn.Module
        self.mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)
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


class LossOperation(Operation):
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

class CropBlockDividable(Operation):
    def __init__(self, block_size=16):
        super().__init__()  # Initialize nn.Module
        self.block_size = block_size
    def forward(self, batch):
        for name in batch:
            H, W = batch[name].shape[-2:]
            w = W//self.block_size*self.block_size
            h = H//self.block_size*self.block_size
            batch[name] = batch[name][..., 0:h, 0:w]
