# -*- coding: utf-8 -*-
import re
import click


from glob import glob
from unidecode import unidecode
from datasets import ClassLabel, DatasetDict, load_dataset
from datasets import Dataset


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--emoticons", default=True)
def main(input_filepath, output_filepath, emoticons):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    dataset = load_raw_dataset(input_filepath)
    dataset = dataset.map(clean_text, batched=True)

    if not emoticons:
        dataset = dataset.map(remove_emoticons, batched=True)

    train_dataset, test_dataset = dataset.train_test_split(
        test_size=0.3, stratify_by_column="sentiment", shuffle=True, seed=42
    ).values()

    dev_dataset, test_dataset = test_dataset.train_test_split(
        test_size=0.5, stratify_by_column="sentiment", shuffle=True, seed=42
    ).values()

    dataset = DatasetDict(
        {"train": train_dataset, "dev": dev_dataset, "test": test_dataset}
    )

    dataset = dataset.rename_columns({"tweet_text": "text", "sentiment": "label"})

    dataset.save_to_disk(output_filepath)


def load_raw_dataset(input_filepath: str) -> Dataset:

    files = glob(input_filepath + "*.csv")

    dataset = load_dataset("csv", data_files=files)
    dataset = dataset["train"]
    dataset = dataset.remove_columns(column_names=["id", "tweet_date", "query_used"])
    dataset = dataset.cast_column(
        column="sentiment", feature=ClassLabel(names=["Neutro", "Positivo", "Negativo"])
    )
    
    return dataset

def clean_text(example: dict) -> dict:
    """
    Cleans the text field with a series of regex patterns.
    """

    processed_batch = []
    tweets = example["tweet_text"]

    for text in tweets:
        # Removes user from tweet
        text = re.sub("@\w+", "", text.lower())

        # Unicode characters to ascii
        text = unidecode(text)

        # Removes URLs
        text = re.sub(
            "((?:(?<=[^a-zA-Z0-9]){0,}(?:(?:https?\:\/\/){0,1}(?:[a-zA-Z0-9\%]{1,}\:[a-zA-Z0-9\%]{1,}[@]){,1})(?:(?:\w{1,}\.{1}){1,5}(?:(?:[a-zA-Z]){1,})|(?:[a-zA-Z]{1,}\/[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\:[0-9]{1,4}){1})){1}(?:(?:(?:\/{0,1}(?:[a-zA-Z0-9\-\_\=\-]){1,})*)(?:[?][a-zA-Z0-9\=\%\&\_\-]{1,}){0,1})(?:\.(?:[a-zA-Z0-9]){0,}){0,1})",
            "",
            text,
        )

        # Stores emoticons and remove non-word chars
        emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
        text = re.sub("[\W]+", " ", text)

        # Removes more than three repeated chars
        text = re.sub(r"(.)\1{2,3}", r"\1", text)

        # Restores emoticons
        text = text + " ".join(emoticons).replace("-", "")

        # Removes trailing whitespace
        text = text.strip()

        processed_batch.append(text)

    example["tweet_text"] = processed_batch

    return example


def remove_emoticons(example: dict) -> dict:
    """
    Removes all non-word chars, including emoticons.
    """

    processed_batch = []
    tweets = example["tweet_text"]

    for text in tweets:

        text = re.sub("[\W]+", " ", text)
        processed_batch.append(text)

    example["tweet_text"] = processed_batch
    return example


if __name__ == "__main__":

    main()
