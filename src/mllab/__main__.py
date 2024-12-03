#!/bin/env/python

import argparse
import datetime
import itertools
import os
import sys

import dotenv
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler
import torch

from astconfig import Config
from mllab.utils import find
from mllab import PyConfigModule


if __name__ == "__main__":
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

    dotenv.load_dotenv(override=True)
    print(sys.executable)

    parser = argparse.ArgumentParser(description="MLLab launches machine learning experiments with pytorch lightning", prog="mllab")
    parser.add_argument("filename")
    parser.add_argument("--epochs", type=int)
    parser.add_argument('--kwargs', nargs="*", action='append', default=[[]])
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()

    config = Config(find(args.filename))
    config.update(";".join(itertools.chain(*args.kwargs)))

    processkey = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    project_name = os.path.splitext(os.path.basename(args.filename))[0]

    profiler = None
    if args.profile:
        profile_folder = f"results/tensorboard/{processkey}"
        profiler = PyTorchProfiler(
            dirpath=profile_folder,
            filename=os.path.basename(args.filename),
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=4),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name=profile_folder)
        )

    trainer = Trainer(
        max_epochs=args.epochs,
        logger=config.get('logger', WandbLogger(project=project_name)),
    )
    trainer.fit(PyConfigModule(config))

