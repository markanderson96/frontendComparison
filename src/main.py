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
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split

from model import Model, CNN_Att
from prepare_data import birdDataset

@hydra.main(config_path='../config', config_name='config')
def main(conf: DictConfig):
    csv = [
        #conf.path.bv_dir,
        conf.path.ff_dir,
        conf.path.warblr_dir
    ]
    df = pd.concat((pd.read_csv(f) for f in csv))
    df = df.sample(frac=1, random_state=42)
    df_train, df_test = train_test_split(df, test_size=0.3, train_size=0.7, random_state=42)
    df_val = df_test.sample(frac=0.5, random_state=42)
    df_test = df_test.drop(df_val.index)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    dataset_train = birdDataset(df_train, conf.path.data_dir, conf.features.sample_rate)
    dataset_val   = birdDataset(df_val, conf.path.data_dir, conf.features.sample_rate)
    dataset_test  = birdDataset(df_test, conf.path.data_dir, conf.features.sample_rate)

    model = Model(conf)

    fast_run = True if conf.set.debug else False

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
        
        checkpoint_cb = pl.callbacks.ModelCheckpoint(
            dirpath=conf.path.model,
            filename=f"efficient_{conf.features.frontend}",
            monitor="val_loss"
        )
        early_stopping_cb = pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10
        )
        trainer = pl.Trainer(
            gpus=conf.set.gpus,
            max_epochs=conf.training.epochs,
            callbacks=[checkpoint_cb, early_stopping_cb],
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
        test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=8,
            shuffle=False,
            num_workers=conf.set.num_workers
        )
        tester = pl.Trainer(
            gpus=conf.set.gpus,
            fast_dev_run=fast_run
        )
        ckpt_path = os.path.join(
            conf.path.model,
            f"efficient_{conf.features.frontend}.ckpt"
        )

        tester.test(model, ckpt_path=ckpt_path, dataloaders=test_loader)
        input("Press any key to continue")

        splits = 25
        samples_per_split = len(dataset_test)//splits
        df = pd.DataFrame()
        for i in range(splits):
            print(f"Split: {i}")
            subset = np.arange(i*samples_per_split, (i+1)*samples_per_split)
            dataset_pred = torch.utils.data.Subset(dataset_test, subset)
            pred_loader = torch.utils.data.DataLoader(
                dataset_pred,
                batch_size=8,
                shuffle=False,
                num_workers=conf.set.num_workers
            )
            x = tester.predict(model, ckpt_path=ckpt_path, dataloaders=pred_loader)
            x = torch.cat(x)
            x = x.view(-1)
            df2 = dataset_test.meta.iloc[dataset_pred.indices]
            df2['pred'] = x.cpu()
            df2['split'] = i
            df2['frontend'] = conf.features.frontend
            df = pd.concat([df, df2])

        df.sort_values(by=['split', 'datasetid'])
        df.to_csv(f"efficient_{conf.features.frontend}_pred.csv", index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(modules)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    tb_logger = TensorBoardLogger("logs", name="cnn_attn")
    main()
