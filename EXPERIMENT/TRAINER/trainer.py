from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast


class CustomizedTrainer:
    def __init__(
        self,
        model: nn.Module,
        lr: float=1e-4, 
        weight_decay: float=1e-3, 
    ):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)
        self.model = model.to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self._set_up_components()

    def __call__(self, trn_loader, val_loader, epoch, n_epochs):
        kwargs = dict(
            dataloader=trn_loader,
            epoch=epoch,
            n_epochs=n_epochs,
        )
        trn_loss = self.trn(**kwargs)

        kwargs = dict(
            dataloader=val_loader,
            epoch=epoch,
            n_epochs=n_epochs,
        )
        val_loss = self.val(**kwargs)

        return trn_loss, val_loss

    def _set_up_components(self):
        self._init_trn()
        self._init_val()

    def _init_trn(self):
        kwargs = dict(
            model=self.model,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.trn = STEP_TRN(**kwargs)

    def _init_val(self):
        kwargs = dict(
            model=self.model,
        )
        self.val = STEP_VAL(**kwargs)


class STEP_TRN:
    def __init__(self, model, lr, weight_decay):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)
        self.model = model.to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self._set_up_components()

    def __call__(self, dataloader, epoch, n_epochs):
        self.model.train()

        epoch_loss = 0.0

        iter_obj = tqdm(
            iterable=dataloader, 
            desc=f"Epoch {epoch+1}/{n_epochs} TRN"
        )

        for X, y in iter_obj:
            # to gpu
            kwargs = dict(
                X=self._dict_to_device(X),
                y=y.to(self.device),
            )

            # forward pass
            with autocast(self.device.type):
                batch_loss = self._batch_step(**kwargs)

            # backward pass
            self._run_fn_opt(batch_loss)

            # accumulate loss
            epoch_loss += batch_loss.item()

        return epoch_loss / len(dataloader)

    def _batch_step(self, X, y):
        logit = self.model(X)
        loss = self.loss_fn(logit, y)
        return loss

    def _dict_to_device(self, obj):
        return {
            k: v.to(self.device) 
            for k, v in obj.items()
        }

    def _run_fn_opt(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _set_up_components(self):
        self._init_loss_fn()
        self._init_optimizer()
        self._init_scaler()

    def _init_loss_fn(self):
        self.loss_fn = nn.MSELoss()

    def _init_optimizer(self):
        kwargs = dict(
            params=self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay,
        )
        self.optimizer = optim.Adam(**kwargs)

    def _init_scaler(self):
        kwargs = dict(
            device=self.device,
        )
        self.scaler = GradScaler(**kwargs)


class STEP_VAL:
    def __init__(self, model):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)
        self.model = model.to(self.device)
        self._set_up_components()

    @torch.no_grad()
    def __call__(self, dataloader, epoch, n_epochs):
        self.model.eval()

        epoch_loss = 0.0

        iter_obj = tqdm(
            iterable=dataloader, 
            desc=f"Epoch {epoch+1}/{n_epochs} VAL"
        )

        for X, y in iter_obj:
            # to gpu
            kwargs = dict(
                X=self._dict_to_device(X),
                y=y.to(self.device),
            )

            # forward pass
            with autocast(self.device.type):
                batch_loss = self._batch_step(**kwargs)

            # accumulate loss
            epoch_loss += batch_loss.item()

        return epoch_loss / len(dataloader)

    def _batch_step(self, X, y):
        logit = self.model(X)
        loss = self.loss_fn(logit, y)
        return loss

    def _dict_to_device(self, obj):
        return {
            k: v.to(self.device) 
            for k, v in obj.items()
        }

    def _set_up_components(self):
        self._init_loss_fn()

    def _init_loss_fn(self):
        self.loss_fn = nn.MSELoss()