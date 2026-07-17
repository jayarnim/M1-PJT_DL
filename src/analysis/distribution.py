from pathlib import Path
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path):
    path = Path(path)
    files = list(path.iterdir())

    dfs = {
        file.stem: pd.read_csv(file)
        for file in files
    }

    scores = defaultdict(dict)

    for key, val in dfs.items():
        model, dim = key.split("_")
        scores[model][int(dim)] = val

    return scores


def draw_ax(ax, df, threshold, model, dim):
    colors = {
        0: "#1f77b4",   # 파랑
        1: "#d62728",   # 빨강
    }

    labels = {
        0: "Non-click",
        1: "Click",
    }

    for label in [0,1]:
        sns.kdeplot(
            data=df[df["true"]==label],
            x="prob",
            fill=True,
            common_norm=False,
            alpha=0.3,
            color=colors[label],
            label=labels[label],
            ax=ax,
        )

    ax.axvline(
        x=threshold,
        color="black",
        linestyle="-",
        linewidth=2,
        label="Threshold",
    )
    ax.set_title(
        f"{model} ({dim} dimension)", 
        fontsize=12, 
        fontweight="bold",
    )
    ax.set_xlabel(
        "Click Through-Rate", 
        fontsize=10,
    )
    ax.set_ylabel(
        "Density", 
        fontsize=10,
    )
    ax.grid(
        True, 
        linestyle="--", 
        alpha=0.5,
    )
    ax.legend(
        fontsize=9,
    )


def main(path, thresholds, figsize, models, dims):
    scores = load_data(path)

    NROWS = len(models)
    NCOLS = len(dims)
    WEIGHTS = figsize[0]
    HEIGHTS = figsize[1]

    fig, axes = plt.subplots(
        nrows=NROWS, 
        ncols=NCOLS, 
        figsize=(WEIGHTS*NCOLS, HEIGHTS*NROWS), 
        sharex=True, 
        sharey=True,
    )

    for i, model in enumerate(models):
        for j, dim in enumerate(dims):
            kwargs = dict(
                ax=axes[i,j],
                df=scores[model][dim],
                threshold=thresholds[model][dim],
                model=model,
                dim=dim,
            )
            draw_ax(**kwargs)

    plt.suptitle(
        t="Click Through-Rate Distribution",
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.show()