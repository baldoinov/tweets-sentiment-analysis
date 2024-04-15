# Apresentar modelo com n-gram e depois apresentar o finetuning do modelo da hugging face
# Ajustar função de custo para penalizar erros na classe menos representada
# Under - oversmapling
# Realizar teste posterior com o dataset do cardniffnlp
# Como no @pysientimiento, realizar ajuste retirando o emoticon utilizado para query do texto

import os
import torch
import logging
import evaluate

import numpy as np

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from datasets import load_from_disk


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_LABELS = 3
DATA_DIR = "data/interim/with-emoticon"

id2label = {0: "Neutro", 1: "Positivo", 2: "Negativo"}
label2id = {"Neutro": 0, "Positivo": 1, "Negativo": 2}
model_checkpoint = "neuralmind/bert-base-portuguese-cased"
model_output_dir = f"{model_checkpoint}-finetuned-sentiment-analysis"


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
metric = evaluate.load("f1")
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=NUM_LABELS
)

def tokenize_function(examples):
    return tokenizer(examples["tweet_text"], padding="max_length", truncation=True, max_length=512)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels, average="macro")

def main(ds):

    args = TrainingArguments(
        output_dir=model_output_dir,
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
        model=model,
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
