# TODO: Identificar UMA implementação que te agrade e segui-lá até o fim.
# TODO: Realizar teste posterior com o dataset do cardniffnlp
# TODO: Como no @pysientimiento, realizar ajuste retirando o emoticon utilizado
# para query do texto
# TODO: Fazer under e oversampling do dataset.
# TODO: Ajustar função de custo para penalizar erros na classe menos representada.

import os
import torch
import numpy as np
import lightning as L

from tqdm.auto import tqdm

from torch import nn
from torch import utils
from torch import optim
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from datasets import load_from_disk
from datasets import DatasetDict

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainingArguments
from transformers import get_scheduler


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NUM_LABELS = 3
MAX_LENGTH = 128
TASK = "sentiment-analysis"
MODEL_NAME = "bertimbau"

ID2LABEL = {0: "Neutro", 1: "Positivo", 2: "Negativo"}
LABEL2ID = {"Neutro": 0, "Positivo": 1, "Negativo": 2}
MODEL_CHECKPOINT = "neuralmind/bert-base-portuguese-cased"
OUTPUT_DIR = f"models/{MODEL_NAME}-finetuned-{TASK}"

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
MODEL = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT, num_labels=NUM_LABELS
)


def tokenize_function(examples: DatasetDict):
    return TOKENIZER(
        examples["text"], padding="max_length", max_length=MAX_LENGTH, truncation=True
    )


def compute_metrics(predictions, references):

    predictions = predictions.detach().cpu().numpy().tolist()
    references = references.detach().cpu().numpy().tolist()

    accuracy = accuracy_score(references, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        references, predictions, average="macro", zero_division=0
    )

    return {
        f"accuracy": accuracy,
        f"f1": f1,
        f"precision": precision,
        f"recall": recall,
    }


class NN(L.LightningModule):
    def __init__(self, model):
        self.model = model

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3)
        return optimizer


trainer = L.Trainer()

def main(ds):

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="steps",
        eval_steps=50,
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=MODEL,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["dev"],
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":

    ds_type = "with-emoticons"

    ds = load_from_disk(f"data/interim/{ds_type}")
    ds = ds.map(tokenize_function, batched=True)
    ds.save_to_disk(f"data/processed/{ds_type}/")

    main(ds)
