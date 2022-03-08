import os
import logging
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from torch.utils.data import DataLoader
from tqdm import tqdm
from glob import glob
from omegaconf import DictConfig

from model import Model
from prepare_data import prepare_data, birdDataset

@hydra.main(config_path='../config', config_name='config')
def main(conf: DictConfig):
    csv = [
        conf.path.bv_dir,
        conf.path.ff_dir,
        conf.path.warblr_dir
    ]
    dataset = birdDataset(csv, conf.path.data_dir, conf.features.sample_rate)
    dataset_train, dataset_val, dataset_test = torch.utils.data.random_split(
        dataset,
        [int(len(dataset)*0.7), int(len(dataset)*0.2), int(len(dataset)*0.1)],
        generator=torch.Generator().manual_seed(42)
    )

    if conf.set.train:
        if not os.path.isdir(conf.path.model):
            os.makedirs(conf.path.model)

        # Make dataloaders
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=conf.training.batch_size,
            shuffle=True,
            num_workers=conf.set.num_workers
        )
        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=conf.training.batch_size,
            shuffle=False,
            num_workers=conf.set.num_workers
        )
    
        fast_run = True if conf.set.debug else False
        
        model = Model(conf)
        checkpoint_cb = pl.callbacks.ModelCheckpoint(
            dirpath=conf.path.root_dir + '/models',
            filename=f"resnest50_{conf.features.frontend}",
            monitor="val_loss"
        )
        trainer = pl.Trainer(
            gpus=conf.set.gpus,
            max_epochs=conf.training.epochs,
            callbacks=[checkpoint_cb],
            logger=tb_logger,
            fast_dev_run=fast_run
        )
        trainer.fit(
            model, 
            train_dataloaders=train_loader, 
            val_dataloaders=val_loader
        )

        logger.info("Training Complete")
    
    if conf.set.eval:
        pass


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(modules)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    tb_logger = TensorBoardLogger("logs", name="cnn_attn")
    main()
