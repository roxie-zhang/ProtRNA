from torch.utils.data import DataLoader

import lightning.pytorch as pl

from typing import Union, Optional
from pathlib import Path

from downstream_mrl.utils.dataset import RibosomeLoadingDataset
# from downstream_mrl.utils.download import download_ribosome_loading_data

VARYING_LEN_25_TO_100_CSV = "GSM4084997_varying_length_25to100.csv.gz"

class RibosomeLoadingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: Union[Path, str],
        feature_path: str,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        skip_data_preparation: bool = True,
        lm_type: str = 'rinalmo',
        test_set: str = 'random'
    ):
        super().__init__()

        self.data_root = Path(data_root)

        self.feature_path = feature_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.skip_data_preparation = skip_data_preparation
        self._data_prepared = skip_data_preparation

        self.lm_type = lm_type
        self.test_set = test_set
        
    # def prepare_data(self):
    #     if not self.skip_data_preparation and not self._data_prepared:
    #         download_ribosome_loading_data(self.data_root)
    #         self._data_prepared = True

    def setup(self, stage: Optional[str] = None):
        dataset = RibosomeLoadingDataset(self.data_root / VARYING_LEN_25_TO_100_CSV, feature_path=self.feature_path, lm_type=self.lm_type)
        if self.test_set == 'random7600': # train, random7600, human7600
            self.train_dataset, self.test_dataset, self.val_dataset = dataset.train_eval_split(num_eval_samples_per_len=100)
        elif self.test_set == 'human7600': # train, random7600, human7600
            self.train_dataset, self.val_dataset, self.test_dataset = dataset.train_eval_split(num_eval_samples_per_len=100)
        else:
            raise NotImplementedError("Test dataset not implemented")
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # shuffle=True,
            shuffle=False
        )

    def val_dataloader(self):
        return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False
        )
