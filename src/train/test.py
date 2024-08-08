# TODO: Realizar teste posterior com o dataset do cardniffnlp


import lightning as pl

from src import config
from src.train.model import SentimentAnalysisModel
from src.train.dataset import TweetsDataModule


if __name__ == "__main__":

    model = SentimentAnalysisModel(
        n_classes=config.NUM_LABELS,
        learning_rate=config.LEARNING_RATE,
        model_checkpoint=config.MODEL_CHECKPOINT,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
        class_weights=config.CLASS_WEIGHTS,
    )

    dm = TweetsDataModule(
        raw_data_dir=config.RAW_DATA_DIR,
        processed_data_dir=config.PROCESSED_DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    trainer = pl.Trainer(min_epochs=1, max_epochs=config.NUM_EPOCHS)

    trainer.test(model, dm)
