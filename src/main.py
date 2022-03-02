import os
import logging
import hydra
import h5py
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from torch.utils.data import DataLoader
from tqdm import tqdm
from glob import glob
from omegaconf import DictConfig

from model import CNN_Att
from prepare_data import prepare_data

@hydra.main(config_path='../config', config_name='config')
def main(conf: DictConfig):
    if conf.set.train:
        if not os.path.isdir(conf.path.model):
            os.makedirs(conf.path.model)

        data_path = conf.path.data_dir
        metadata_path = conf.path.meta_dir

        df = pd.read_csv(metadata_path)
        # TODO train/val split
        df = df.sample(frac=0.1)
        df_train = df.sample(frac=0.8, random_state=42)
        df_val = df.drop(df_train.index)

        # load train data
        dataset_train, dataset_val = prepare_data(
            csv=conf.path.meta_dir,
            data_path=conf.path.data_dir,
            class_num=conf.training.class_num,
            sample_rate=conf.features.sample_rate,
            num_workers=conf.set.num_workers
        )

        # Convert preloaded data to torch tensor to use as input in the pytorch Dataloader
        dataset_train.set_format(type='torch', columns=['audio', 'target', 'hot_target'])
        dataset_val.set_format(type='torch', columns=['audio', 'target', 'hot_target'])

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
        
        model = CNN_Att(conf)
        trainer = pl.Trainer(
            gpus=conf.set.gpus,
            max_epochs=conf.training.epochs,
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
