#!/bin/env/python

import argparse
import itertools
import os
import sys

import dotenv
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger

from astconfig import Config
from mllab.utils import find
from mllab import PyConfigModule, PyConfigDataModule


if __name__ == "__main__":
    if os.getcwd() not in os.sys.path:
        os.sys.path.insert(0, os.getcwd())

    dotenv.load_dotenv()
    print(sys.executable)

    parser = argparse.ArgumentParser(description="MLLab launches machine learning experiments with pytorch lightning", prog="mllab")
    parser.add_argument("filename")
    parser.add_argument("--epochs", type=int)
    parser.add_argument('--kwargs', nargs="*", action='append', default=[[]])
    args = parser.parse_args()

    config = Config(find(args.filename))
    config.update(";".join(itertools.chain(*args.kwargs)))

    module = PyConfigModule(config)
    datamodule = PyConfigDataModule(config)

    project_name = os.path.splitext(os.path.basename(args.filename))[0]

    wandb_logger = WandbLogger(project=project_name)
    trainer = pl.Trainer(max_epochs=args.epochs, callbacks=config['callbacks'], logger=wandb_logger)
    trainer.fit(module, datamodule=datamodule)

