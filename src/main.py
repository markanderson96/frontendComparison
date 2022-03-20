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
from sklearn.model_selection import KFold

from model import Model, CNN_Att
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

    model = CNN_Att(conf)

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
            filename=f"simple_{conf.features.frontend}",
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
        test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=64,
            shuffle=False,
            num_workers=conf.set.num_workers
        )
        tester = pl.Trainer(
            gpus=conf.set.gpus,
            fast_dev_run=fast_run
        )
        ckpt_path = os.path.join(
            conf.path.model,
            f"resnest50_{conf.features.frontend}"
        )
        tester.test(model, ckpt_path=ckpt_path, dataloaders=test_loader)

        splits = 10
        samples_per_split = len(dataset_test)//10
        df = pd.DataFrame()
        for i in range(splits):
            print(f"Split: {i}")
            subset = np.arange(i*samples_per_split, (i+1)*samples_per_split)
            dataset_pred = torch.utils.data.Subset(dataset_test, subset)
            pred_loader = torch.utils.data.DataLoader(
                dataset_pred,
                batch_size=32,
                shuffle=False,
                num_workers=conf.set.num_workers
            )
            x = tester.predict(model, ckpt_path=ckpt_path, dataloaders=pred_loader)
            x = torch.cat(x)
            x = x.view(-1)
            df2 = dataset_test.dataset.meta.iloc[dataset_pred.indices]
            df2['pred'] = x.cpu()
            df2['split'] = i
            df2['frontend'] = conf.features.frontend
            df = pd.concat([df, df2])

        df.sort_values(by=['split', 'datasetid'])
        df.to_csv(f"resnest50_{conf.features.frontend}_pred.csv", index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(modules)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    tb_logger = TensorBoardLogger("logs", name="cnn_attn")
    main()
