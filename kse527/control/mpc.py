from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torchdyn.numerics import odeint
from notebooks.federico.utils import SurrogateDynamicModel

class Action(nn.Module):
    """
    A nn.Module wrapper for action
    """

    def __init__(self,
                 H,
                 action_dim,
                 action_min,
                 action_max):
        super(Action, self).__init__()
        us = np.random.uniform(low=action_min, high=action_max, size=(1, H, action_dim))
        self.us = torch.nn.Parameter(torch.from_numpy(us).float())
        self.action_min = action_min
        self.action_max = action_max

    def forward(self):
        return self.us

    def clamp_action(self):
        self.us.data = self.us.data.clamp(min=self.action_min, max=self.action_max)


class MPC(nn.Module):
    """
    Minimal MPC implementation utilizing arbitrary torch.nn.Module as the dynamic model
    """

    def __init__(self,
                 model: nn.Module,
                 state_dim: int,
                 action_dim: int,
                 H: int,  # receding horizon
                 action_min: float = -3.0,
                 action_max: float = 3.0,
                 gamma: float = 1.0,
                 lr: float = 1e-1,
                 rollout_fn: Callable = None, 
                 use_default_rollout_fn: bool = True,
                 loss_fn=torch.nn.MSELoss(reduction='none')):
        super(MPC, self).__init__()

        self.model = model
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.H = H
        self.action_min = action_min
        self.action_max = action_max
        self.gamma = gamma
        self._gamma = torch.tensor([self.gamma ** t for t in range(H)]).view(1, H, 1)


        if rollout_fn is None:
            if use_default_rollout_fn:
                self.rollout_fn = self._default_rollout_fn
            else:
                self.rollout_fn = self._modified_rollout_fn
        else:
            self.rollout_fn = rollout_fn
        self.loss_fn = loss_fn
        self.lr = lr

    def solve(self, x0, target, max_iter: int, tol=1e-5):

        us = Action(self.H, self.action_dim, self.action_min, self.action_max).to(target.device)
        opt = torch.optim.Adam(us.parameters(), lr=self.lr)  # Large LR (step size) start heuristics
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)

        info = dict()
        solve = False
        for i in range(max_iter):
            opt.zero_grad()
            prediction = self.rollout_fn(x0, us())
            loss = (self.loss_fn(prediction, target) * self._gamma).mean()

            loss.backward()
            opt.step()
            scheduler.step(loss)

            # Projection heuristics
            us.clamp_action()
            if loss <= tol:
                solve = True
                break
        info['loss'] = loss.item()
        info['solve'] = solve
        info['iters'] = i
        return us.us.data, info

    def _modified_rollout_fn(self, x0, us):
        """
        :param x0: initial state. expected to get 'torch.tensor' with dimension of [batch x state_dim]
        :param us: action sequences [batch x time stamps x  action_dim]
        """
        xs = []
        x = x0
        # iterating over time stamps
        for u_ in us.unbind(dim=1): 
            u_ = u_.squeeze(0) # compatibility
            if isinstance(self.model, SurrogateDynamicModel):
                # learned model
                x = self.model.forward(x, u_)
            else:
                # nominal model = differential equation
                self.model.u.u0 = u_ # substitute controller
                x = odeint(self.model.dynamics, x, torch.linspace(0, 0.02, 2), solver=self.model.solver)[1][-1]
            xs.append(x)
        return torch.stack(xs, dim=0) 
        
    def _default_rollout_fn(self, x0, us):
        """
        :param x0: initial state. expected to get 'torch.tensor' with dimension of [batch x state_dim]
        :param us: action sequences [batch x time stamps x  action_dim]
        """
        xs = []
        x = x0

        for u in us.unbind(dim=1):  # iterating over time stamps
            x = self.model(x, u)
            xs.append(x)
        return torch.stack(xs, dim=1)  # [batch x time stamps x state_dim]