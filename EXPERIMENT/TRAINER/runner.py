from IPython.display import clear_output
import torch


class Runner:
    def __init__(self, model, trainer):
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)
        self.model = model.to(self.device)
        self.trainer = trainer

    def fit(self, trn_loader, val_loader, n_epochs):
        trn_loss_list = []
        val_loss_list = []

        for epoch in range(n_epochs):
            if epoch % 10 == 0:
                print(f"EPOCH {epoch+1} START ---->>>>")

            # trn, val
            kwargs = dict(
                trn_loader=trn_loader, 
                val_loader=val_loader, 
                epoch=epoch,
                n_epochs=n_epochs,
            )
            trn_loss, val_loss = self._run_trainer(**kwargs)

            # accumulate
            trn_loss_list.append(trn_loss)
            val_loss_list.append(val_loss)

            # log reset
            if (epoch + 1) % 50 == 0:
                clear_output(wait=False)

        # log reset
        clear_output(wait=False)

        print("TRN IS FINISHED!")

        return dict(
            trn=trn_loss_list,
            val=val_loss_list,
        )

    def _run_trainer(self, trn_loader, val_loader, epoch, n_epochs):
        kwargs = dict(
            trn_loader=trn_loader, 
            val_loader=val_loader, 
            epoch=epoch,
            n_epochs=n_epochs,
        )
        trn_loss, val_loss = self.trainer(**kwargs)

        print(
            f"TRN LOSS: {trn_loss:.4f}",
            f"VAL LOSS: {val_loss:.4f}",
            sep='\n',
        )

        return trn_loss, val_loss