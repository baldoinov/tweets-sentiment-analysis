import matplotlib.pyplot as plt


def sentiment_pie_plot(dataset, ax):
    df = dataset.to_pandas()
    class_counts = df["sentiment"].value_counts()
    class_percentages = (class_counts / class_counts.sum()) * 100

    out = ax.pie(
        class_percentages,
        labels=class_percentages.index.map({0: "Neutro", 1: "Positivo", 2: "Negativo"}),
        autopct="%1.1f%%",
    )

    return out
