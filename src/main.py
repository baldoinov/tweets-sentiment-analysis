# TODO: Realizar teste posterior com o dataset do cardniffnlp


import lightning as pl
from lightning.pytorch.cli import LightningCLI

from src.model import SentimentAnalysisModel
from src.dataset import TweetsDataModule


def main():
    cli = LightningCLI(model_class=SentimentAnalysisModel, datamodule_class=TweetsDataModule)


if __name__ == "__main__":
    main()
