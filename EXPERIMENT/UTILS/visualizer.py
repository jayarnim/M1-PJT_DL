import matplotlib.pyplot as plt


def score_plot(
    history: list,
    title: str,
    metric: str,
    figsize: tuple=(8,5),
):
    plt.figure(figsize=figsize)
    plt.plot(history)
    plt.xlabel('EPOCH')
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def versus_plot(
    history: dict,
    title: str,
    metric: str,
    figsize: tuple=(8,5),
):
    plt.figure(figsize=figsize)
    plt.plot(history['trn'], label='TRN')
    plt.plot(history['val'], label='VAL')
    plt.xlabel('EPOCH')
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()