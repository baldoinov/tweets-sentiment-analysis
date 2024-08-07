import torch
import datasets
import lightning as pl


from glob import glob
from torch.utils.data import DataLoader
from datasets import ClassLabel
from src.utils import clean_text, split_data, tokenize_function
from src.config import PROCESSED_DATA_DIR
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS


class TweetsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int) -> None:

        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:

        files = glob(self.data_dir + "*.csv")
        dataset = datasets.load_from_disk("csv", files)
        dataset = dataset["train"]

        dataset.rename_columns({"tweet_text": "text", "sentiment": "labels"})

        dataset = dataset.remove_columns(column_names=["id", "tweet_date", "query_used"])
        dataset = dataset.cast_column(
            column="labels", feature=ClassLabel(names=["Neutro", "Positivo", "Negativo"])
        )
        dataset = dataset.map(clean_text, batched=True)
        dataset = dataset.map(tokenize_function, batched=True)
        dataset = split_data(dataset)
        dataset.set_format("torch")

        dataset.save_to_disk(PROCESSED_DATA_DIR)

    def setup(self, stage: str) -> None:

        if stage == "fit":
            self.train_ds = datasets.load_from_disk(PROCESSED_DATA_DIR / "train")

        elif stage == "validate":
            self.val_ds = datasets.load_from_disk(PROCESSED_DATA_DIR / "dev")

        elif stage == "test":
            self.test_ds = datasets.load_from_disk(PROCESSED_DATA_DIR / "test")

    def train_dataloader(self) -> torch.Any:
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.test_ds, batch_size=self.batch_size)
