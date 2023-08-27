import os 
import pandas as pd
import torch 
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset 
import pytorch_lightning as pl


def split_dataset(ds, train_fraction, seed):
    n_full = len(ds)
    n_train = int(train_fraction*n_full)
    n_test = n_full - n_train 
    ds_train, ds_test = random_split(ds, [n_train, n_test], generator=torch.Generator().manual_seed(seed))
    return ds_train, ds_test 


def get_dataset(data_dir, sequence_len):

    filenames = [f"U{n}.csv" for n in range(sequence_len+1)]
    data = []
    for fname in filenames: 
        u = pd.read_csv(os.path.join(data_dir, fname)).to_numpy()
        data.append(torch.tensor(u))
    ds = TensorDataset(*data)
    return ds


class DataModule(pl.LightningDataModule):
    def __init__(self, train_dir, test_dir, sequence_len, batch_size, num_workers=4):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir 
        self.sequence_len = sequence_len 
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        self.ds_train = get_dataset(self.train_dir, self.sequence_len)
        self.ds_test = get_dataset(self.test_dir, self.sequence_len)
    
        # ds_train = Subset(ds_train, range(len(ds_train)//5))

        print("U_n (n=0,1,...,{0}) train: {1}".format(len(self.ds_train[:])-1, self.ds_train[:][0].shape))
        print("U_n (n=0,1,...,{0}) test: {1}".format(len(self.ds_test[:])-1, self.ds_test[:][0].shape))

    def train_dataloader(self):
        if self.trainer.current_epoch < 400:
            batch_size = 100
        elif self.trainer.current_epoch < 1000:
            batch_size = 200
        elif self.trainer.current_epoch < 2200:
            batch_size = 400
        else:
            batch_size = self.batch_size
        return DataLoader(self.ds_train, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False)

    def predict_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False)
    
