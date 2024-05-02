import os
import torch
import logging
import evaluate
import numpy as np

from datasets import load_from_disk
from datasets import DatasetDict

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

OUTPUT_DIR = f"models/{MODEL_NAME}-finetuned-{TASK}"

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)
MODEL = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID
)

def tokenize_function(examples: DatasetDict):
    return TOKENIZER(
        examples["text"],
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True
    )


def compute_metrics(eval_pred):

    print(eval_pred)
    print(type(eval_pred))

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro"
    )

    return {"accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall}


ds = load_from_disk("")
ds = ds.map(tokenize_function, batched=True)
ds = ds.remove_columns(["text"])
ds.set_format("torch")

# small_train_dataset = ds["train"].shuffle(seed=42).select(range(1000))
# small_eval_dataset = ds["test"].shuffle(seed=42).select(range(1000))


train_dataloader = DataLoader(ds["train"], shuffle=True, batch_size=BATCH_SIZE)
eval_dataloader = DataLoader(ds["dev"], batch_size=BATCH_SIZE)
optimizer = AdamW(MODEL.parameters(), lr=5e-5)

num_epochs = 5
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

MODEL.to(DEVICE)

progress_bar = tqdm(range(num_training_steps))

MODEL.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = MODEL(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

import evaluate

metric = evaluate.load("accuracy")
MODEL.eval()
for batch in eval_dataloader:
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    with torch.no_grad():
        outputs = MODEL(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
