from pathlib import Path
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt


def load_data(path):
    path = Path(path)
    files = list(path.iterdir())

    dfs = {
        file.stem: pd.read_csv(file)
        for file in files
    }

    auroc = defaultdict(dict)
    logloss = defaultdict(dict)

    for key, val in dfs.items():
        model, dim = key.split("_")
        auroc[model][int(dim)] = val["auroc"].item()
        logloss[model][int(dim)] = val["logloss"].item()

    return dict(
        AUROC=auroc, 
        Logloss=logloss,
    )


def draw_ax(ax, scores, models, metric, title):
    for model in models:
        dims = sorted(scores[model].keys())
        x = range(len(dims))
        y = [scores[model][dim] for dim in dims]

        ax.plot(
            *(x, y),
            marker="o",
            label=model,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(dims)
    
    ax.set_title(
        f"{title} ({metric})", 
        fontsize=12, 
        fontweight="bold",
    )
    ax.set_xlabel("Dimension")
    ax.set_ylabel(metric)
    ax.grid(
        True, 
        linestyle="--", 
        alpha=0.5,
    )
    ax.legend(
        fontsize=9,
    )


def main(path, figsize, modelset, titles):
    metrics = load_data(path)

    NROWS = len(metrics)
    NCOLS = len(titles)
    WEIGHTS = figsize[0]
    HEIGHTS = figsize[1]

    fig, axes = plt.subplots(
        nrows=NROWS, 
        ncols=NCOLS, 
        figsize=(WEIGHTS*NCOLS, HEIGHTS*NROWS), 
        sharex=True, 
        sharey="row",
    )

    for i, (metric, vals) in enumerate(metrics.items()):
        for j, (models, title) in enumerate(zip(modelset, titles)):
            kwargs = dict(
                ax=axes[i,j],
                scores=vals,
                models=models, 
                metric=metric, 
                title=title,
            )
            draw_ax(**kwargs)

    plt.suptitle(
        t="Metric per Dimension",
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.show()