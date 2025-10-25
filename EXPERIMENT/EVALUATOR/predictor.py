from tqdm import tqdm
import torch


class Predictor:
    def __init__(self, model):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)
        self.model = model.to(self.device)

    @torch.no_grad()
    def __call__(self, dataloader):
        self.model.eval()

        # to save result
        pred_list = []
        true_list = []

        iter_obj = tqdm(
            iterable=dataloader, 
            desc=f"TST",
        )

        for X, y in iter_obj:
            # to gpu
            kwargs = dict(
                X=self._dict_to_device(X),
            )

            # predict
            pred = self.model.predict(**kwargs)

            # to cpu & save
            pred_list.extend(pred.cpu().tolist())
            true_list.extend(y.cpu().tolist())

        return dict(
            pred=torch.tensor(pred_list, dtype=torch.float32).squeeze(-1),
            true=torch.tensor(true_list, dtype=torch.float32).squeeze(-1),
        )

    def _dict_to_device(self, obj):
        return {
            k: v.to(self.device) 
            for k, v in obj.items()
        }