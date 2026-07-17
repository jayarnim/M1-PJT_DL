from pathlib import Path
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


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


def draw_ax(ax, df, y_prob, y_true, model, dim):
    prob_true, prob_pred = calibration_curve(
        y_true=df[y_true],
        y_prob=df[y_prob],
        n_bins=10,
    )
    ax.plot(
        *(prob_pred, prob_true), 
        marker="o",
    )
    ax.plot(
        *([0,1],[0,1]),
        linestyle="-",
        color="red",
    )

    ax.set_title(
        f"{model} ({dim} dimension)", 
        fontsize=12, 
        fontweight="bold",
    )
    ax.set_xlabel(
        "Mean Predicted Probability", 
        fontsize=10,
    )
    ax.set_ylabel(
        "Observed Frequency", 
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


def main(path, y_prob, y_true, figsize, models, dims):
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
                df=scores[model][dim],
                y_prob=y_prob, 
                y_true=y_true, 
                ax=axes[i,j],
                model=model,
                dim=dim,
            )
            draw_ax(**kwargs)

    plt.suptitle(
        t="Reliability Diagram",
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.show()