# import sys; sys.path.append(2*'../') # go n dirs back
import matplotlib.pyplot as plt


import wandb

import torch
#get_ipython().system('pip install pytorch_lightning')
import pytorch_lightning as pl
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F


from pytorch_lightning import Trainer
import numpy as np
import os
import math

from kse527.systems.classic_control import Pendulum
from kse527.control.controllers import RandConstController
from kse527.utils.datamodules.trajectory import TrajectoryDataModule
from kse527.tasks.default import DynamicSystemLearner
from kse527.callbacks.wandb_callbacks import setup_logging

# Change device according to your configuration
##NOTE: DO NOT USE THIS HERE, USE IN pl.Trainer
#device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('gpu') # feel free to change :)
#device = torch.device('gpu') # override

#from src.models ... import Snake
from kse527.models.activations.snake import Snake
Snake(64)

#from src.models ... import Snake
from kse527.models.activations.siren import Siren



#import tensorflow.keras as tf
#from tensorflow.keras import layers
#from tensorflow.keras import activations
#model softsign.add(layers.Dense(64, activation=activations.softsign))

#model.add(layers.Dense(64))
#model.add(layers.Activation(activations.relu))


#from keras.layers import LeakyReLU
#LeakyReLU(64)

#from keras.layers import Softmax
#Softmax(64)





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


def train_loop(model, name):

    # Declare the model structure with the neural network
    model_nn = SurrogateDynamicModel(model)

    # Instantiate Datamodule
    ROOT_PATH = '../../data/cartpole'
    datamodule = TrajectoryDataModule(ROOT_PATH)

    # Declare model with the Lighting Module, which will take care of a shitload of stuff
    model = DynamicSystemLearner(model_nn, lr=3e-4)

    logger, callbacks = setup_logging(name=name,project_name='KS527-cartpole') # insert your name here

    # Instantiate Trainer
    trainer = pl.Trainer(max_epochs=500,
                logger=logger,
                callbacks=callbacks,
		accelerator='gpu',
		devices=[0])

    # Fit the model
    trainer.fit(model, datamodule)


    # ## Plot Results
    #
    # We first create some data distributions

    # In[4]:


    trainer.test(model, datamodule)

    
# SIREN Example
#model_siren = Siren(in_features=5, out_features=4, hidden_features=64, hidden_layers=2)


#model_leakyrelu = nn.Sequential(nn.Linear(3, 64),  nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2))
#model_softmax = nn.Sequential(nn.Linear(5, 64),  nn.Softmax(), nn.Linear(64, 64), nn.Softmax(), nn.Linear(64, 4))
#model_softplus = nn.Sequential(nn.Linear(5, 64),  nn.Softplus(), nn.Linear(64, 64), nn.Softplus(), nn.Linear(64, 4))
model_snake = nn.Sequential(nn.Linear(5, 64),  Snake(64), nn.Linear(64, 64), Snake(64), nn.Linear(64, 4))
#model_relu = nn.Sequential(nn.Linear(5, 64),  nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4))
#model_tanh = nn.Sequential(nn.Linear(5, 64),  nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 4))
#model_gelu = nn.Sequential(nn.Linear(5, 64),  nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 4))
#model_sigmoid = nn.Sequential(nn.Linear(5, 64),  nn.Sigmoid(), nn.Linear(64, 64), nn.Sigmoid(), nn.Linear(64, 4))
#model_elu = nn.Sequential(nn.Linear(5, 64),  nn.ELU(), nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 4))
#model_selu = nn.Sequential(nn.Linear(5, 64),  nn.SELU(), nn.Linear(64, 64), nn.SELU(), nn.Linear(64, 4))

model_reluselu = nn.Sequential(nn.Linear(5, 64),  nn.ReLU(), nn.Linear(64, 64), nn.SELU(), nn.Linear(64, 4))
model_gelurelu = nn.Sequential(nn.Linear(5, 64),  nn.GELU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 4))

# Declare the model structure with the neural network
#model_siren = SurrogateDynamicModel(model_siren)

#model_snakegelu = nn.Sequential(nn.Linear(5, 64),  Snake(64), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 4))
#model_snakeleakyrelu = nn.Sequential(nn.Linear(3, 64),  Snake(64), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 2))
#model_selugelu = nn.Sequential(nn.Linear(5, 64),  nn.SELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 4))

#names = ['model siren', 'model softmax', 'model softplus', 'model snake', 'model relu', 'model tanh', 'model gelu', 'model sigmoid', 'model elu', 'model selu']
#models = [model_siren, model_softmax, model_softplus, model_snake, model_relu, model_tanh, model_gelu, model_sigmoid, model_elu, model_selu]

#names = ['model siren', 'model leakyrelu', 'model snake', 'model relu', 'model gelu', 'model selu', 'model snakegelu', 'model snakeleakyrelu', 'model selugelu']
#models =  [model_siren, model_leakyrelu, model_snake, model_relu, model_gelu, model_selu, model_selugelu, model_snakeleakyrelu, model_selugelu]

#names = ['model siren', 'model snake', 'model relu', 'model gelu', 'model selu', 'model snakegelu', 'model selugelu']
#models =  [model_siren, model_snake, model_relu, model_gelu, model_selu, model_snakegelu, model_selugelu]

names = ['model snake', 'model reluselu', 'model gelurelu']
models = [model_snake, model_reluselu, model_gelurelu]

for model, name in zip(models, names):
    print("Training {}".format(name))
    train_loop(model, name)

#for model, name in range(5)
#    print()
