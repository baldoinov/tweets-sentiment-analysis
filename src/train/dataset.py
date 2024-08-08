import torch
import datasets
import lightning as pl
import fsspec as fs

from glob import glob
from torch.utils.data import DataLoader
from datasets import ClassLabel
from src.utils import clean_text, split_data, tokenize_function


class TweetsDataModule(pl.LightningDataModule):
    def __init__(
        self, raw_data_dir: str, processed_data_dir: str, batch_size: int, num_workers: int
    ) -> None:

        super().__init__()

        self.raw_data_dir = raw_data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.processed_data_dir = processed_data_dir

    def prepare_data(self) -> None:

        files = [str(file) for file in self.raw_data_dir.glob("*.csv")]
        dataset = datasets.load_dataset("csv", data_files=files)
        dataset = dataset["train"]

        dataset = dataset.rename_columns({"tweet_text": "text", "sentiment": "labels"})
        dataset = dataset.remove_columns(column_names=["id", "tweet_date", "query_used"])
        dataset = dataset.cast_column(
            column="labels", feature=ClassLabel(names=["Neutro", "Positivo", "Negativo"])
        )
        dataset = dataset.map(clean_text, batched=True)
        dataset = dataset.map(tokenize_function, batched=True)
        dataset = split_data(dataset)
        dataset.set_format("torch")

        dataset.save_to_disk(self.processed_data_dir)

    def setup(self, stage: str) -> None:
        self.train_ds = datasets.load_from_disk(self.processed_data_dir / "train")
        self.val_ds = datasets.load_from_disk(self.processed_data_dir / "dev")
        self.test_ds = datasets.load_from_disk(self.processed_data_dir / "test")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False
        )
