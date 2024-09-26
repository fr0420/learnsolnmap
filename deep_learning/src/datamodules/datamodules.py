import logging
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities import CombinedLoader
from datamodules.datasets import get_sequence_data


logger = logging.getLogger(__name__)


class DefaultDataModule(pl.LightningDataModule):
    def __init__(self, train: dict, test: dict, batch_size: int = 100, num_workers: int = 4, pin_memory: bool = True):
        super().__init__()
        self.train_config = train
        self.test_config = test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        try:
            assert self.train_config["dtype"] == self.test_config["dtype"], "Mismatch in data types."
            self.ds_train = get_sequence_data(self.train_config)
            self.ds_test = get_sequence_data(self.test_config)
            logger.info(f"Datasets successfully loaded.")
        except AssertionError as e:
            logger.error(f"Data type consistency error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            raise

        logger.info(f"Train dataset directory: {self.train_config['data_dir']}")
        logger.info(f"Test dataset directory: {self.test_config['data_dir']}")
        logger.info(f"U_n (n=0,1,...,{self.train_config['sequence_len']-1}) train: {self.ds_train[:]['input'].shape}")
        logger.info(f"U_n (n=0,1,...,{self.test_config['sequence_len']-1}) test: {self.ds_test[:]['input'].shape}")

    def get_dtype(self):
        return self.train_config["dtype"]
    
    def _create_loader(self, train: bool = True) -> DataLoader:
        dataset = self.ds_train if train else self.ds_test
        try:
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=train, num_workers=self.num_workers, pin_memory=self.pin_memory)
            return loader
        except Exception as e:
            logger.error(f"Error creating DataLoader: {e}")
            raise

    def train_dataloader(self):
        return self._create_loader(train=True)
    
    def val_dataloader(self):
        return self._create_loader(train=False)

    def test_dataloader(self):
        return self._create_loader(train=False)

    def predict_dataloader(self):
        return self._create_loader(train=False)


class CombinedDataModule(pl.LightningDataModule):
    def __init__(self, supervised: dict, unsupervised: dict, num_workers: int = 4, pin_memory: bool = True):
        super().__init__()
        assert "train" in supervised and "test" in supervised, "Supervised data configuration must contain 'train' and 'test' keys"
        assert "train" in unsupervised and "test" in unsupervised, "Unsupervised data configuration must contain 'train' and 'test' keys"
        self.sup_train_config = supervised["train"]
        self.sup_test_config = supervised["test"]
        self.unsup_train_config = unsupervised["train"]
        self.unsup_test_config = unsupervised["test"]
        self.sup_batch_size = supervised.get("batch_size", 100)
        self.unsup_batch_size = unsupervised.get("batch_size", 100)

        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        try:
            assert self.sup_train_config["dtype"] == self.sup_test_config["dtype"], "Mismatch in supervised data types."
            assert self.unsup_train_config["dtype"] == self.unsup_test_config["dtype"], "Mismatch in unsupervised data types."
            assert self.sup_train_config["dtype"] == self.unsup_train_config["dtype"], "Mismatch between supervised and unsupervised data types."
            self.ds_train_sup = get_sequence_data(self.sup_train_config)
            self.ds_test_sup = get_sequence_data(self.sup_test_config)
            self.ds_train_unsup = get_sequence_data(self.unsup_train_config)
            self.ds_test_unsup = get_sequence_data(self.unsup_test_config)
            logger.info(f"Datasets successfully loaded.")
        except AssertionError as e:
            logger.error(f"Data type consistency error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            raise

        logger.info(f"Supervised train dataset directory: {self.sup_train_config['data_dir']}")
        logger.info(f"Supervised test dataset directory: {self.sup_test_config['data_dir']}")
        logger.info(f"U_n (n=0,1,...,{self.ds_train_sup.sequence_len-1}) train: {self.ds_train_sup.inputs.shape}")
        logger.info(f"U_n (n=0,1,...,{self.ds_test_sup.sequence_len-1}) test: {self.ds_test_sup.inputs.shape}")
        logger.info(f"Unsupervised train dataset directory: {self.unsup_train_config['data_dir']}")
        logger.info(f"Unsupervised test dataset directory: {self.unsup_test_config['data_dir']}")
        logger.info(f"U_0 train: {self.ds_train_unsup.inputs.shape}")
        logger.info(f"U_0 test: {self.ds_test_unsup.inputs.shape}")

    def get_dtype(self):
        return self.sup_train_config["dtype"]

    def _create_combined_loader(self, train: bool = True) -> CombinedLoader:
        supervised_dataset = self.ds_train_sup if train else self.ds_test_sup
        unsupervised_dataset = self.ds_train_unsup if train else self.ds_test_unsup

        try:
            loader_sup = DataLoader(supervised_dataset, batch_size=self.sup_batch_size, shuffle=train, num_workers=self.num_workers, pin_memory=self.pin_memory)
            loader_unsup = DataLoader(unsupervised_dataset, batch_size=self.unsup_batch_size, shuffle=train, num_workers=self.num_workers, pin_memory=self.pin_memory)
            combined_loader = CombinedLoader({"supervised": loader_sup, "unsupervised": loader_unsup}, mode="max_size_cycle")
            return combined_loader
        except Exception as e:
            logger.error(f"Error creating combined DataLoader: {e}")
            raise

    def train_dataloader(self):
        return self._create_combined_loader(train=True)

    def val_dataloader(self):
        return self._create_combined_loader(train=False)
        
    def test_dataloader(self):
        return self._create_combined_loader(train=False)
        
    def predict_dataloader(self):
        return self._create_combined_loader(train=False)
