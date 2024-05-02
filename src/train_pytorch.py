import os
import torch
import numpy as np
import pandas as pd

from datasets import load_from_disk
from datasets import DatasetDict

from pathlib import Path

from ray import cloudpickle
from ray import tune
from ray import train
from ray.train import Checkpoint
from ray.train import get_checkpoint
from ray.tune.schedulers import ASHAScheduler

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from tqdm.auto import tqdm

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

DATA_4_HYPERPARAM_SEARCH_PATH = ""
DATA_PATH = ""
OUTPUT_DIR = f"models/{MODEL_NAME}-finetuned-{TASK}"

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)
MODEL = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID
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


def load_data(data_path):

    ds = load_from_disk(data_path)
    ds = ds.map(tokenize_function, batched=True)
    ds = ds.remove_columns(["text"])
    ds.set_format("torch")

    train_dataloader = DataLoader(ds["train"], shuffle=True, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(ds["dev"], shuffle=False, batch_size=BATCH_SIZE)

    return train_dataloader, eval_dataloader


def training_step(
    train_dataloader, optimizer, lr_scheduler, progress_bar, epoch
) -> dict:

    prefix = "train"
    train_metrics = {
        "mode": [],
        "step": [],
        "epoch": [],
        "batch": [],
        "loss": [],
        "accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
    }

    for idx, batch in enumerate(train_dataloader):

        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = MODEL(**batch)
        loss = outputs.loss
        loss.backward()

        # "Logging" training metrics
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        references = batch["labels"]

        metrics = compute_metrics(predictions, references)

        train_metrics["mode"].append(prefix)
        train_metrics["step"].append(idx)
        train_metrics["epoch"].append(epoch)
        train_metrics["loss"].append(loss.detach().cpu().numpy().tolist())
        train_metrics["accuracy"].append(metrics["accuracy"])
        train_metrics["f1"].append(metrics["f1"])
        train_metrics["precision"].append(metrics["precision"])
        train_metrics["recall"].append(metrics["recall"])

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    return train_metrics


def evaluation_step(eval_dataloader: DataLoader, epoch: int) -> dict:

    prefix = "eval"
    eval_metrics = {
        "mode": [],
        "step": [],
        "epoch": [],
        "batch": [],
        "loss": [],
        "accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
    }

    for idx, batch in enumerate(eval_dataloader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = MODEL(**batch)

        loss = outputs.loss
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        references = batch["labels"]

        # "Logging" evaluation metrics
        metrics = compute_metrics(predictions, references)

        eval_metrics["mode"].append(prefix)
        eval_metrics["step"].append(idx)
        eval_metrics["epoch"].append(epoch)
        eval_metrics["loss"].append(loss.detach().cpu().numpy().tolist())
        eval_metrics["accuracy"].append(metrics["accuracy"])
        eval_metrics["f1"].append(metrics["f1"])
        eval_metrics["precision"].append(metrics["precision"])
        eval_metrics["recall"].append(metrics["recall"])

    return eval_metrics


def train(lr, wd, num_epochs, data_path):

    training_set, evaluate_set = load_data(data_path)
    optimizer = AdamW(MODEL.parameters(), lr=lr, weight_decay=wd)
    num_training_steps = num_epochs * len(training_set)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))
    best_model_metric = 0
    logging_df = None

    MODEL.to(DEVICE)
    for epoch in range(num_epochs):
        MODEL.train()
        train_metrics = training_step(
            training_set, optimizer, lr_scheduler, progress_bar, epoch
        )
        train_metrics = pd.DataFrame(train_metrics)
        print(train_metrics)

        MODEL.eval()
        eval_metrics = evaluation_step(evaluate_set, epoch)
        eval_metrics = pd.DataFrame(eval_metrics)
        print(eval_metrics)

        logging_df = pd.concat(
            [logging_df, train_metrics, eval_metrics], axis=0, ignore_index=True
        )
        logging_df.to_csv(f"{OUTPUT_DIR}/training-log.csv", index=False)

        eval_f1_score = sum(eval_metrics["f1"]) / len(eval_metrics["f1"])
        if eval_f1_score > best_model_metric:
            best_model_metric = eval_f1_score

            MODEL.save_pretrained(f"{OUTPUT_DIR}/hugging-face-save")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": MODEL.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": train_metrics["loss"],
                },
                f"{OUTPUT_DIR}/pytorch-save",
            )


# Abstrair o training step em um função
# Encapsular o código para conseguir fazer busca de hiperparametros
# criar um dataset menor para a busca de hiperparametros
# adicionar codigo de busca de hiperparametros
# Escrever função para realizar teste no conjunto de teste e no outro dataset
if __name__ == "__main__":

    train(lr=5e-5,
          wd=0.01,
          num_epochs=5,
          data_path=DATA_PATH)
