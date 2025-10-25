from IPython.display import clear_output
import torch
import torch.nn as nn
from .predictor import Predictor
from .metrics import MetricsComputer


class PerformanceEvaluator:
    def __init__(
        self, 
        model: nn.Module, 
    ):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)
        self.model = model.to(self.device)
        self._set_up_components()

    def evaluate(self, tst_loader):
        result = self.predictor(tst_loader)
        clear_output(wait=False)
        performance = self.metrics(result)
        return performance

    def _set_up_components(self):
        self._init_predictor()
        self._init_metrics()

    def _init_predictor(self):
        kwargs = dict(
            model=self.model,
        )
        self.predictor = Predictor(**kwargs)

    def _init_metrics(self):
        self.metrics = MetricsComputer()