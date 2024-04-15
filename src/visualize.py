import seaborn as sns
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from nltk.corpus import stopwords

SW = stopwords.words("portuguese")


def classes_distributions_pie_plot(ds):

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    for key, ax in zip(ds.keys(), axes):
        df = ds[key].to_pandas()

        class_counts = df["sentiment"].value_counts()
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

    x = df["tweet_text"].str.split().map(lambda x: len(x))
    sns.histplot(x=x, binwidth=3, stat="count", ax=ax)

    ax.set_xlabel("")
    ax.set_ylabel("# Palavras por Tweet")

    plt.savefig(fname="../figures/number-of-words-per-tweet-in-train.png")
    plt.show()


def tweets_wordcloud(df):

    text = " ".join(tweet for tweet in df["tweet_text"])
    wcloud = WordCloud(
        stopwords=SW, background_color="white", max_font_size=50, max_words=5000
    ).generate(text)

    plt.figure(figsize=(14, 12))
    plt.imshow(wcloud, interpolation="bilinear")
    plt.axis("off")

    plt.savefig(fname="../figures/worldcloud-in-train.png")
    plt.show()
