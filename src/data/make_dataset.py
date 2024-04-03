# -*- coding: utf-8 -*-
import click
import logging


from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from datasets import ClassLabel, DatasetDict, load_dataset


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    files = Path(input_filepath).rglob("*.csv")

    dataset = load_dataset("csv", data_files=files)
    dataset = dataset.remove_columns(column_names=["id", "tweet_date", "query_used"])
    dataset = dataset.cast_column(
        column="sentiment", feature=ClassLabel(names=["Neutro", "Positivo", "Negativo"])
    )

    train_dataset, test_dataset = dataset.train_test_split(
        test_size=0.3, stratify_by_column="sentiment", shuffle=True, seed=42
    ).values()

    dev_dataset, test_dataset = test_dataset.train_test_split(
        test_size=0.5, stratify_by_column="sentiment", shuffle=True, seed=42
    ).values()

    dataset = DatasetDict(
        {"train": train_dataset, "dev": dev_dataset, "test": test_dataset}
    )

    dataset.save_to_disk(output_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
