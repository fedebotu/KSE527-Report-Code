import torch
import pytorch_lightning as pl
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import Trainer
import numpy as np
import os
import math
from box import Box
import argparse 
from collections import OrderedDict

from kse527.systems.classic_control import Pendulum
from kse527.control.controllers import RandConstController
from kse527.utils.datamodules.trajectory import TrajectoryDataModule
from kse527.tasks.default import DynamicSystemLearner
from kse527.callbacks.wandb_callbacks import setup_logging


class SurrogateDynamicModel(nn.Module):
    """
    Surrogate (="learned") model that takes as input current state and control input
    Predicts state update
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.residual = True # add previous state
    
    def forward(self, x, u):
        xu = torch.cat([x, u], dim=-1)
        if self.residual: 
            next_state = self.model(xu) + x
        else:
            next_state = self.model(xu)
        return next_state


def build_model(layer_dims, activations='relu', model_name = ['mlp-']):
    """
    Build sequential model given layers mapping
    layers of the type: [[in, hdim], [hdim, out]]
    """
    act_name = activations.lower()
    if act_name == 'relu':
        act = nn.ReLU()
    elif act_name == 'gelu':
        act = nn.GELU()
    else:
        raise NotImplementedError("TODO")

    num_layers = len(layer_dims)
    layers = []
    for i, ldim in enumerate(layer_dims):
        layers.append(nn.Linear(ldim[0], ldim[1]))
        model_name.append('{}-{}'.format(ldim[0], ldim[1]))
        if i < num_layers-1:
            model_name.append('-{}/'.format(act_name))
            layers.append(act)

    model = nn.Sequential(*layers)
    model_name = ''.join(model_name)

    # Declare the model structure with the neural network
    model = SurrogateDynamicModel(model)
    return model, model_name


def main(model, cfg, model_name='',  use_wandb=True):

    # Instantiate Datamodule
    datamodule = TrajectoryDataModule(cfg.datamodule.data_path, batch_size=cfg.datamodule.batch_size, num_workers=cfg.datamodule.num_workers) # call from the main directory

    # Declare model with the Lighting Module, which will take care of a shitload of stuff
    model = DynamicSystemLearner(model, lr=float(cfg.train.lr))

    # Logging
    logger, callbacks = setup_logging(name=model_name, project_name=cfg.project_name, use_wandb=use_wandb) # insert your name here

    # Instantiate Trainer
    trainer = pl.Trainer(max_epochs=cfg.train.epochs,
                logger=logger,
                callbacks=callbacks,
                devices=cfg.train.device,
                accelerator=cfg.train.accelerator)

    # Fit the model
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default='notebooks/architectures/config')
    parser.add_argument('--config', default='default')
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction,
                        help='use --no-wandb to disable wandb logging')
    parser.set_defaults(wandb=True)
    args = parser.parse_args()

    config = Box.from_yaml(filename='{}.yaml'.format(os.path.join(args.folder, args.config)))

    # model, model_name = build_model(config.model.layers, activations=config.model.activation)
    # main(model, config, model_name=model_name, use_wandb=args.wandb)

    # # Manual override for layers - architecture sweep
    layers_list = [] 
    for hdim in 4, 8, 16, 32, 64, 128:
        layers_list.append([[5, hdim], [hdim,4]])
        layers_list.append([[5, hdim],[hdim,hdim],[hdim,4]])

    for layers in layers_list:
        config.model.layers = layers
        model, model_name = build_model(config.model.layers, activations=config.model.activation)
        main(model, config, model_name=model_name, use_wandb=args.wandb)

    # import multiprocessing as mp
    # def _mp_train(layers):
    #     model, model_name = build_model(layers, activations=config.model.activation)
    #     main(model, config, model_name=model_name, use_wandb=args.wandb)

    # with mp.Pool() as p:
    #     p.map(_mp_train, layers_list)
