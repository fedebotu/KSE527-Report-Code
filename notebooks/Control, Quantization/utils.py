import torch
import torch.nn as nn

class NeuralModelWrapper(nn.Module):
    """
    Wrapper for the learned model
    """
    def __init__(self, model, u, 
                 retain_u=False,
                 residual=True):
        super().__init__()
        self.model = model # neural model
        self.u = u
        self.retain_u = retain_u # use for retaining control input (e.g. MPC simulation)
        self.residual = residual # use x(t+1) = x(t) + model(x(t)). If false: x(t+1) = model(x(t))
        self.nfe = 0 # count number of function evaluations of the vector field
        self.cur_u = None # current controller value
        self._retain_flag = False # temporary flag for evaluating the controller only the first time

    def forward(self, x0, t_span):
        """Execute rollout over time span"""
        x = [x0[None]]
        xt = x0
        if self.retain_u:
            '''Iterate over the t_span: evaluate the controller the first time only and then retain it'''
            for i in range(len(t_span)-1):
                self._retain_flag = False
                xt = self.step(t_span[i], xt)
                x.append(xt[None])
            traj = torch.cat(x)
        else:
            '''Compute trajectory with odeint and base solvers'''
            for i in range(len(t_span)-1):
                xt = self.step(t_span[i], xt)
                x.append(xt[None])
            traj = torch.cat(x)       
        return traj

    def step(self, t, x):
        u = self._evaluate_controller(t, x)
        xu = torch.cat([x, u], dim=-1)
        if self.residual: 
            next_state = self.model(xu) + x
        else:
            next_state = self.model(xu)
        return next_state

    def reset_nfe(self):
        """Return number of function evaluation and reset"""
        cur_nfe = self.nfe; self.nfe = 0
        return cur_nfe

    def _evaluate_controller(self, t, x):
        '''
        If we wish not to re-evaluate the control input, we set the retain
        flag to True so we do not re-evaluate next time
        '''
        if self.retain_u:
            if not self._retain_flag:
                self.cur_u = self.u(t, x)
                self._retain_flag = True
            else: 
                pass # We do not re-evaluate the control input
        else:
            self.cur_u = self.u(t, x)
        return self.cur_u