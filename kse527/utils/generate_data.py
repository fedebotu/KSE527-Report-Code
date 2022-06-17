import argparse
from math import pi as π
import numpy as np
import os
from pyrfc3339 import generate
import torch
import torch.nn as nn

from kse527.control.controllers import RandConstController
from kse527.systems.classic_control import Pendulum, CartPole

def generate_data_pendulum(n_traj: int,
                            dt: float = 0.02,
                            T: float = 0.02,
                            x_bounds = 3*π,
                            u_bounds = 20):
        # Distributions
        t_span = torch.linspace(0, T, int(T/dt)+1)
        dist_x = torch.distributions.Uniform(torch.Tensor([-x_bounds, -x_bounds]), torch.Tensor([x_bounds, x_bounds]))
        dist_u = torch.distributions.Uniform(torch.Tensor([-u_bounds]), torch.Tensor([u_bounds]))
        x0 = dist_x.sample((n_traj,))
        u0 = dist_u.sample((n_traj,))

        # System
        u = RandConstController()
        u.u0 = u0
        pendulum = Pendulum(u=u, solver='rk4')

        # Simulate
        x_target = pendulum.forward(x0, t_span)[1:].swapaxes(0,1) # swap time and batch axes to [batch x time x dim]

        data = {'x0': x0,
                'u0': u.u0,
                'target': x_target,
                't_span': t_span}
        return data


def generate_data_cartpole(n_traj: int,
                            dt: float = 0.02,
                            T: float = 0.02,
                            x_bounds = 3*π,
                            u_bounds = 20):
        # Distributions
        t_span = torch.linspace(0, T, int(T/dt)+1)
        dist_x = torch.distributions.Uniform(torch.Tensor([-x_bounds, -x_bounds, -x_bounds, -x_bounds]), torch.Tensor([x_bounds, x_bounds, x_bounds, x_bounds]))
        dist_u = torch.distributions.Uniform(torch.Tensor([-u_bounds]), torch.Tensor([u_bounds]))
        x0 = dist_x.sample((n_traj,))
        u0 = dist_u.sample((n_traj,))

        # System
        u = RandConstController()
        u.u0 = u0
        pendulum = CartPole(u=u, solver='rk4')

        # Simulate
        x_target = pendulum.forward(x0, t_span)[1:].swapaxes(0,1) # swap time and batch axes to [batch x time x dim]

        data = {'x0': x0,
                'u0': u.u0,
                'target': x_target,
                't_span': t_span}
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', default='pendulum')
    parser.add_argument('--folder', default='data')
    parser.add_argument('--size', default=100000)
    parser.add_argument('--Ttrain', default=0.02)
    parser.add_argument('--Ttest', default=0.5)


    args = parser.parse_args()

    assert args.system in ['pendulum', 'cartpole'], f"{args.system} is not (yet) implemented"
    if args.system == 'pendulum':
        generate_data = generate_data_pendulum
    elif args.system == 'cartpole':
        generate_data = generate_data_cartpole

    root_folder = os.path.join(args.folder, args.system)
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    TRAIN_SIZE = args.size
    data = generate_data(TRAIN_SIZE, T=args.Ttrain)
    np.savez(os.path.join(root_folder, 'train.npz'), **data)

    VAL_SIZE = int(TRAIN_SIZE*0.1)
    data = generate_data(VAL_SIZE, T=args.Ttest)
    np.savez(os.path.join(root_folder, 'val.npz'), **data)

    TEST_SIZE = int(TRAIN_SIZE*0.1)
    data = generate_data(TEST_SIZE, T=args.Ttest)
    np.savez(os.path.join(root_folder, 'test.npz'), **data)