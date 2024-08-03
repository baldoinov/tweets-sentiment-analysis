import os
import re
import torch
import streamlit as st

from unidecode import unidecode
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Only for type annotation
from transformers.tokenization_utils_base import BatchEncoding

NUM_LABELS = 3
MAX_LENGTH = 128
ID2LABEL = {0: "Neutro", 1: "Positivo", 2: "Negativo"}
LABEL2ID = {"Neutro": 0, "Positivo": 1, "Negativo": 2}
MODEL_CHECKPOINT = "neuralmind/bert-base-portuguese-cased"
MODEL_PATH = "models/bertimbau-finetuned-sentiment-analysis/checkpoint-hugging-face/"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)
MODEL = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID
)


# Streamlit app
def main():

    st.set_page_config(
        page_title="Análise de Sentimentos em Tweets",
        page_icon="assets/CIBERDEMIcon.png",
    )
    st.title("Análise de Sentimentos em Tweets")

    # Input text box
    text = st.text_area("Insira um tweet para realizar a análise de sentimento:")

    # Button to perform sentiment analysis
    if st.button("Processar"):
        cols = st.columns(spec=3)
        if text:
            text = text_cleaning(text)
            text = tokenize_function(TOKENIZER, text)
            output = model_inference(MODEL, text)

            for idx, (col, (sentiment, score)) in enumerate(zip(cols, output)):
                score = f"{score * 100:.4f}%"

                if idx == 0:
                    col.metric(
                        label="Classificação",
                        value=sentiment,
                        delta=score,
                        delta_color="normal",
                    )
                else:
                    col.metric(
                        label="Classificação",
                        value=sentiment,
                        delta=score,
                        delta_color="off",
                    )


def model_inference(model, tokens: BatchEncoding) -> list[tuple]:
    """
    This functions performs the sentiment classification using a given model.
    """

    outputs = model(**tokens)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()[0]

    scores = []

    for i, p in zip(range(len(probabilities)), probabilities):
        scores.append((ID2LABEL[i], p))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    return scores


def tokenize_function(tokenizer: AutoTokenizer, text: str) -> BatchEncoding:
    """
    Function to handle tokenization.
    """
    return tokenizer(
        text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True, padding=True
    )


def text_cleaning(text: str) -> str:
    """
    Text pre-processing function that used to build the training dataset.
    More details of its working can be found at `build_dataset.py`.
    In sum, it works in the following way:
        1. Removes user from tweet;
        2. Converts unicode characters to ascii;
        3. Removes URLs;
        4. Removes non-word chars and emoticons;
        5. Removes more than three repeated chars;
        6. Restores emoticons removed in step 4;
        7. Removes trailing whitespaces.
    """

    text = re.sub("@\w+", "", text.lower())
    text = unidecode(text)
    text = re.sub(
        "((?:(?<=[^a-zA-Z0-9]){0,}(?:(?:https?\:\/\/){0,1}(?:[a-zA-Z0-9\%]{1,}\:[a-zA-Z0-9\%]{1,}[@]){,1})(?:(?:\w{1,}\.{1}){1,5}(?:(?:[a-zA-Z]){1,})|(?:[a-zA-Z]{1,}\/[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\:[0-9]{1,4}){1})){1}(?:(?:(?:\/{0,1}(?:[a-zA-Z0-9\-\_\=\-]){1,})*)(?:[?][a-zA-Z0-9\=\%\&\_\-]{1,}){0,1})(?:\.(?:[a-zA-Z0-9]){0,}){0,1})",
        "",
        text,
    )

    # Stores emoticons and remove non-word chars
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
    text = re.sub("[\W]+", " ", text)

    text = re.sub(r"(.)\1{2,3}", r"\1", text)
    text = text + " ".join(emoticons).replace("-", "")
    text = text.strip()

    return text


if __name__ == "__main__":

    print(os.getcwd())
    main()
