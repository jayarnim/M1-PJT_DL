from torchmetrics.regression import (
    MeanSquaredError,
    NormalizedRootMeanSquaredError,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    R2Score,
)


class MetricsComputer:
    def __init__(self):
        self._set_up_components()

    def __call__(self, result):
        self.mse.reset()
        self.rmse.reset()
        self.mae.reset()
        self.mape.reset()
        self.r2.reset()

        args = (result["pred"], result["true"])

        self._metric_printer(*args)

    def _metric_printer(self, y_pred, y_true):
        print(
            f"MSE:\t\t{self.mse(y_pred, y_true).item():.4f}",
            f"RMSE(norm):\t{self.rmse(y_pred, y_true).item():.4f}",
            f"MAE:\t\t{self.mae(y_pred, y_true).item():.4f}",
            f"MAPE:\t\t{self.mape(y_pred, y_true).item():.4f}",
            f"R2:\t\t{self.r2(y_pred, y_true).item():.4f}",
            sep="\n",
        )

    def _set_up_components(self):
        self._init_metrics()

    def _init_metrics(self):
        self.mse = MeanSquaredError()
        self.rmse = NormalizedRootMeanSquaredError()
        self.mae = MeanAbsoluteError()
        self.mape = MeanAbsolutePercentageError()
        self.r2 = R2Score()