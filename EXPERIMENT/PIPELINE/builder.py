from collections import defaultdict
import pandas as pd
from ..UTILS.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    SEED,
)
from .msr.python_splitters import python_stratified_split
from .dataloader import FMDataLoader


class Builder:
    def __init__(
        self, 
        origin: pd.DataFrame,
        col_y: str=DEFAULT_RATING_COL,
        col_user: str=DEFAULT_USER_COL, 
        col_item: str=DEFAULT_ITEM_COL,
        seed: int=SEED,
    ):
        self.origin = origin
        self.col_X = origin.columns.difference([col_y])
        self.col_y = col_y
        self.col_user = col_user
        self.col_item = col_item
        self.seed = seed
        self._set_up_components()

    def __call__(
        self, 
        trn_val_tst_ratio: dict=dict(trn=0.6, val=0.2, tst=0.2),
        batch_size: dict=dict(trn=128, val=128, tst=128),
        shuffle: bool=True,
    ):
        # split original data
        kwargs =dict(
            trn_val_tst_ratio=trn_val_tst_ratio,
        )
        split_dict = self._data_splitter(**kwargs)

        # generate data loaders
        kwargs = dict(
            split_dict=split_dict, 
            batch_size=batch_size, 
            shuffle=shuffle,
        )
        return self._dataloader_generator(**kwargs), self._field_dim_generator()

    def _field_dim_generator(self):
        field_dim = dict()
        for field, features in self.field_map.items():
            if len(features)==1:
                field_dim |= self.origin[features].nunique().to_dict()
            else:
                field_dim[field] = len(features)
        return field_dim

    def _data_splitter(self, trn_val_tst_ratio):
        split_type = list(trn_val_tst_ratio.keys())
        split_ratio = list(trn_val_tst_ratio.values())

        kwargs = dict(
            data=self.origin,
            ratio=split_ratio,
            col_user=self.col_user,
            col_item=self.col_item,
            seed=self.seed,
        )
        split_list = python_stratified_split(**kwargs)

        return dict(zip(split_type, split_list))

    def _dataloader_generator(self, split_dict, batch_size, shuffle):
        loader_dict = {}
        for k, v in split_dict.items():
            kwargs = dict(
                split=v, 
                batch_size=batch_size[k], 
                shuffle=shuffle,
            )
            loader_dict[k] = self.dataloader(**kwargs)

        return loader_dict

    def _set_up_components(self):
        self._init_field_map()
        self._init_dataloader()

    def _init_field_map(self):
        self.field_map = defaultdict(list)
        for col in self.col_X:
            prefix = col.split("_")[0]
            self.field_map[prefix].append(col)

    def _init_dataloader(self):
        kwargs = dict(
            origin=self.origin,
            field_map=self.field_map,
            col_X=self.col_X,
            col_y=self.col_y,
            seed=self.seed,
        )
        self.dataloader = FMDataLoader(**kwargs)