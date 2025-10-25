import torch
from torch.utils.data import DataLoader, Dataset
from ..UTILS.constants import (
    DEFAULT_RATING_COL,
    SEED,
)


class FMDataset(Dataset):
    def __init__(self, X, y, field_map):
        self.X = {
            field: (
                torch.tensor(X[features].values, dtype=torch.long)
                .squeeze(-1)
            )
            for field, features in field_map.items()
        }
        self.y = torch.tensor(y.values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X_idx = {
            field: tensor[idx]
            for field, tensor in self.X.items()
        }
        return X_idx, self.y[idx]


class FMDataLoader:
    def __init__(
        self, 
        origin, 
        field_map,
        col_X,
        col_y=DEFAULT_RATING_COL, 
        seed=SEED,
    ):
        self.origin = origin
        self.field_map = field_map
        self.col_X = col_X
        self.col_y = col_y
        self.seed = seed
    
    def __call__(
        self, 
        split, 
        batch_size, 
        shuffle,
    ):
        dataset = self._dataset_generator(split)
        return self._dataloader_generator(dataset, batch_size, shuffle)

    def _dataset_generator(self, split):
        kwargs = dict(
            X=split.loc[:, self.col_X],
            y=split.loc[:, self.col_y],
            field_map=self.field_map,
        )
        return FMDataset(**kwargs)

    def _dataloader_generator(self, dataset, batch_size, shuffle):
        kwargs = dict(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=self._collate_fn,
        )
        return DataLoader(**kwargs)

    def _collate_fn(self, batch):
        # unpacking
        X_dict, y_batch = zip(*batch)
        # fields
        fields = self.field_map.keys()
        # dict of field → list[tensor]
        X_batch = {
            field: torch.stack([sample[field] for sample in X_dict])
            for field in fields
        }
        # y: (B,)
        y_batch = torch.stack(y_batch)
        return X_batch, y_batch