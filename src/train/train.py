# TODO: Realizar teste posterior com o dataset do cardniffnlp


import lightning as pl


from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateFinder,
    BatchSizeFinder,
    EarlyStopping,
)
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger


from src import config
from src.train.model import SentimentAnalysisModel
from src.train.dataset import TweetsDataModule


seed_everything(42)


if __name__ == "__main__":

    logger = CSVLogger(
        save_dir=config.MODELS_DIR,
        name=config.MODEL_NAME,
        )

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
    trainer = pl.Trainer(
        logger=logger,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        callbacks=[
            ModelCheckpoint(
                dirpath=config.MODELS_DIR / config.MODEL_NAME, 
                monitor="val_f1", 
                auto_insert_metric_name=True
            )
        ],
    )

    trainer.fit(model, dm)
