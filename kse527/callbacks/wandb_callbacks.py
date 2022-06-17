import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

def setup_logging(name='', project_name='KS527', early_stop=False, use_wandb=True, reinit=True, config={}):
    if use_wandb is True:
        name = '{}'.format(name)
        wandb_run = wandb.init(project=project_name,
                               name=name, 
                               reinit=reinit,
                               config=config.to_dict())
        logger = WandbLogger(experiment=wandb_run, log_model="all")
        latest_checkpoint = ModelCheckpoint(dirpath=wandb.run.dir,
                                            filename="latest-{epoch}",
                                            monitor='epoch',
                                            mode='max',
                                            every_n_epochs=1)
        val_checkpoint = ModelCheckpoint(dirpath=wandb.run.dir,
                                         monitor='val/rollout_loss',)
        # callbacks = [latest_checkpoint, val_checkpoint]
        callbacks = [val_checkpoint]
        if early_stop:
            callbacks.append(EarlyStopping(monitor='val/rollout_loss', patience=10, mode='min'))
        return logger, callbacks
    else:
        return None, None