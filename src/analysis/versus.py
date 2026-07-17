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


def draw_ax(ax, dfs, models, label, dim):
    PALLETE = ["#228B22", "#1f77b4", "#d62728"]
    
    colors = dict(zip(models, PALLETE))

    labels = {
        1: "Click",
        0: "Non-Click",
    }

    for model in models:
        df = dfs[model][dim]
        
        sns.kdeplot(
            data=df[df["true"]==label],
            x="prob",
            fill=True,
            common_norm=False,
            alpha=0.3,
            color=colors[model],
            label=model,
            ax=ax,
        )

    ax.set_title(
        f"{dim} dimension ({labels[label]})", 
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


def main(path, figsize, models, dims, suptitle):
    scores = load_data(path)

    LABELS = [1,0]
    NROWS = len(LABELS)
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

    for i, label in enumerate(LABELS):
        for j, dim in enumerate(dims):
            kwargs = dict(
                ax=axes[i,j],
                dfs=scores,
                models=models,
                label=label,
                dim=dim,
            )
            draw_ax(**kwargs)

    plt.suptitle(
        t=f"Click Through-Rate Distribution ({suptitle})",
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.show()