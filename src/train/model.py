import torch
import lightning as pl

from torch import nn
from torch.optim import AdamW
from torchmetrics import Accuracy, Precision, Recall, F1Score
from transformers import AutoModelForSequenceClassification
from src.config import MODEL_CHECKPOINT


class SentimentAnalysisModel(pl.LightningModule):
    def __init__(self, n_classes, learning_rate) -> None:
        super().__init__()

        self.bert = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

        self.training_step_outputs = []
        self.training_step_labels = []

        self.accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.f1_score = F1Score(task="multiclass", num_classes=n_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_ouput)
        loss = 0

        if labels is not None:
            loss = self.criterion(output, labels)

        return loss, output

    def __common_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        loss, outputs = self.forward(input_ids, attention_mask, labels)

        return loss, outputs, labels

    def training_step(self, batch, batch_idx):

        loss, outputs, labels = self.__common_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, logger=True)

        self.training_step_outputs.append(outputs)
        self.training_step_labels.append(labels)

        return {"loss": loss, "predctions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):

        loss, outputs, labels = self.__common_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):

        loss, outputs, labels = self.__common_step(batch, batch_idx)
        self.log("test_loss", loss, prog_bar=True, logger=True)

        return loss

    def predict_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        loss, outputs = self.forward(input_ids, attention_mask, labels)
        outputs = torch.argmax(outputs, dim=1)

        return outputs

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def on_train_epoch_end(self) -> None:
        epoch_outputs = torch.cat(self.training_step_outputs)
        epoch_labels = torch.cat(self.training_step_labels)
 
        self.log_dict(
            {
                "train_acc": self.accuracy(epoch_outputs, epoch_labels),
                "train_f1": self.f1_score(epoch_outputs, epoch_labels),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self) -> None:
        return super().on_validation_epoch_end()
