import torch
import torch.nn as nn
import pytorch_lightning as pl

class DynamicSystemLearner(pl.LightningModule):
    def __init__(self, 
                    model: nn.Module = None, 
                    loss_fn: nn.Module = nn.MSELoss(),
                    lr: float = 5e-4):

        """
        The model predicts the state update given current state and control:
        x(t+1) = x(t) + model(x(t), u(t))

        Args:
            model: (nn.Module) dynamic model
            loss_fn: (nn.Module) loss function
            lr: (float) learning rate
        """

        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr

    def forward(self, x, u):
        return self.model(x, u)

    def rollout(self, x0, u, n_rollout=10):
        """
        Perform rollout: we iterate n times to obtain future states
        """
        xs = []
        x = x0
        for i in range(n_rollout):  # iterating over time stamps
            x = self.forward(x, u) # we keep the same control u here
            xs.append(x)
        return torch.stack(xs, dim=1)

    def training_step(self, batch, batch_idx):
        x0, u0, target, _ = batch
        x_pred = self.forward(x0, u0) # forward propagate model
        loss = self.loss_fn(target.squeeze(), x_pred) # loss function
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.999)
        return [optimizer] #, [scheduler]

    def validation_step(self, batch, batch_idx):
        return self.__val_test_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.__val_test_step(batch, batch_idx, 'test')

    def __val_test_step(self, batch, batch_idx, step_name):
        x0, u0, targets, t_span = batch
        with torch.no_grad():
            preds = self.rollout(x0, u0, n_rollout = t_span.shape[-1]-1)
            loss = torch.mean((preds - targets)**2, dim=(0,2)) # manual MSE for validation
            metrics = {
                    '{}/one_step_loss'.format(step_name): loss[:1].mean(),
                    '{}/rollout_loss'.format(step_name): loss.mean()
                    }
            self.log_dict(metrics)
        return metrics