# This script was made to be run from inside notebooks/0.2-visualizing-the-dataset.ipynb


import seaborn as sns
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from nltk.corpus import stopwords

SW = stopwords.words("portuguese")


def classes_distributions_pie_plot(ds):

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    for key, ax in zip(ds.keys(), axes):
        df = ds[key].to_pandas()

        class_counts = df["labels"].value_counts()
        class_percentages = (class_counts / class_counts.sum()) * 100

        ax.pie(
            class_percentages,
            labels=class_percentages.index.map(
                {0: "Neutro", 1: "Positivo", 2: "Negativo"}
            ),
            autopct="%1.1f%%",
        )

        ax.set_title(f"{key.capitalize()} - {ds[key].shape[0]}")

    fig.suptitle("Distribuição de Classes")

    plt.savefig(fname="../figures/class-distribution-for-dataset-splits.png")
    plt.show()


def number_of_words_per_tweet(df):

    fig, ax = plt.subplots(figsize=(10, 8))

    x = df["text"].str.split().map(lambda x: len(x))
    max_words_in_tweet = x.max()

    sns.histplot(x=x, binwidth=3, stat="count", ax=ax)
    plt.axvline(x=max_words_in_tweet, color="b")

    ax.set_xlabel("")
    ax.set_ylabel("# Palavras por Tweet")

    plt.savefig(fname="../figures/number-of-words-per-tweet-in-train.png")
    plt.show()


def metrics_from_training(df):

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    epoch_aggregate = (
        df.pivot_table(
            index=["mode", "epoch"], values=["loss", "accuracy", "f1"], aggfunc="mean"
        )
        .reset_index()
        .replace({"train": "Treino", "eval": "Validação"})
        .rename({"epoch": "Epoch", "loss": "Loss", "f1": "F1-Score"}, axis=1)
    )

    metrics = ["Loss", "F1-Score"]

    for idx, m in enumerate(metrics):
        g = sns.lineplot(data=epoch_aggregate, x="Epoch", y=m, hue="mode", ax=axes[idx])
        g.get_legend().set_title(None)

    fig.suptitle("Métricas no Conjunto de Treino/Validação")
    sns.despine()

    plt.savefig(fname="../figures/metrics-in-train-and-eval.png")
    plt.show()

def tweets_wordcloud(df):

    text = " ".join(tweet for tweet in df["text"])
    wcloud = WordCloud(
        stopwords=SW, background_color="white", max_font_size=50, max_words=5000
    ).generate(text)

    plt.figure(figsize=(14, 12))
    plt.imshow(wcloud, interpolation="bilinear")
    plt.axis("off")

    plt.savefig(fname="../figures/worldcloud-in-train.png")
    plt.show()
