import torch
import torch.nn as nn


class DeepFactorizationMachine(nn.Module):
    def __init__(
        self, 
        embed_dim: int,
        hidden: list,
        dropout: float,
        n_fields: int,
        field_dim: dict, 
        field_type: dict,
    ):
        super().__init__()
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        self.embed_dim = embed_dim
        self.hidden = hidden
        self.dropout = dropout
        self.n_fields = n_fields
        self.field_dim = field_dim
        self.field_type = field_type
        self._set_up_components()

    def forward(self, X):
        return self.score(X).squeeze(-1)

    @torch.no_grad()
    def predict(self, X):
        return self.score(X).squeeze(-1)

    def score(self, X):
        score_main = self.main_effect(X)
        score_interaction = self.interaction_effect(X)
        return score_main + score_interaction + self.bias

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        kwargs = dict(
            field_dim=self.field_dim,
            field_type=self.field_type,
        )
        self.main_effect = MainEffect(**kwargs)

        kwargs = dict(
            embed_dim=self.embed_dim,
            hidden=self.hidden,
            dropout=self.dropout,
            n_fields=self.n_fields,
            field_dim=self.field_dim,
            field_type=self.field_type,
        )
        self.interaction_effect = InteractionEffect(**kwargs)

        self.bias = nn.Parameter(torch.zeros(1))


class MainEffect(nn.Module):
    def __init__(self, field_dim, field_type):
        super().__init__()
        self.field_dim = field_dim
        self.field_type = field_type
        self._set_up_components()

    def forward(self, X):
        embed_list = self.embed_list_generator(X)
        stacked = torch.cat(embed_list, dim=1)
        return torch.sum(stacked, dim=1, keepdim=False)

    def embed_list_generator(self, X):
        emb_list = []

        if self.field_type["num"]:
            num_emb_list = [
                (X[field].float() @ emb.weight)
                for field, emb in self.embeddings.items()
                if field in self.field_type["num"]
            ]
            emb_list.extend(num_emb_list)

        if self.field_type["oht"]:
            oht_emb_list = [
                emb(X[field])
                for field, emb in self.embeddings.items()
                if field in self.field_type["oht"]
            ]
            emb_list.extend(oht_emb_list)

        if self.field_type["mht"]:
            mht_emb_list = [
                (
                    (X[field].float() @ emb.weight) 
                    / (torch.sum(X[field], dim=1, keepdim=True) + 1e-8)
                )
                for field, emb in self.embeddings.items()
                if field in self.field_type["mht"]
            ]
            emb_list.extend(mht_emb_list)

        return emb_list

    def _set_up_components(self):
        self._create_embeddings()
        self._init_embeddings()

    def _create_embeddings(self):
        components = {
            field: nn.Embedding(n_features, 1) 
            for field, n_features in self.field_dim.items()
        }
        self.embeddings = nn.ModuleDict(components)

    def _init_embeddings(self):
        for emb in self.embeddings.values():
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)


class InteractionEffect(nn.Module):
    def __init__(self, embed_dim, hidden, dropout, n_fields, field_dim, field_type):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.dropout = dropout
        self.n_fields = n_fields
        self.field_dim = field_dim
        self.field_type = field_type
        self._set_up_components()

    def forward(self, X):
        emb_list = self.emb_list_generator(X)
        score_fm = self.fm(emb_list)
        score_dnn = self.dnn(emb_list)
        return score_fm + score_dnn

    def emb_list_generator(self, X):
        emb_list = []

        if self.field_type["num"]:
            num_emb_list = [
                (X[field].float() @ emb.weight)
                for field, emb in self.embeddings.items()
                if field in self.field_type["num"]
            ]
            emb_list.extend(num_emb_list)

        if self.field_type["oht"]:
            oht_emb_list = [
                emb(X[field])
                for field, emb in self.embeddings.items()
                if field in self.field_type["oht"]
            ]
            emb_list.extend(oht_emb_list)

        if self.field_type["mht"]:
            mht_emb_list = [
                (
                    (X[field].float() @ emb.weight) 
                    / (torch.sum(X[field], dim=1, keepdim=True) + 1e-8)
                )
                for field, emb in self.embeddings.items()
                if field in self.field_type["mht"]
            ]
            emb_list.extend(mht_emb_list)

        return emb_list

    def _set_up_components(self):
        self._create_embeddings()
        self._init_embeddings()
        self._create_layers()

    def _create_embeddings(self):
        components = {
            field: nn.Embedding(n_features, self.embed_dim) 
            for field, n_features in self.field_dim.items()
        }
        self.embeddings = nn.ModuleDict(components)

    def _init_embeddings(self):
        for emb in self.embeddings.values():
            nn.init.normal_(emb.weight, mean=0.0, std=0.01)

    def _create_layers(self):
        self.fm = FM()

        kwargs = dict(
            embed_dim=self.embed_dim,
            hidden=self.hidden,
            dropout=self.dropout,
            n_fields=self.n_fields,
        )
        self.dnn = DNN(**kwargs)


class FM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, emb_list):
        stacked = torch.stack(emb_list, dim=1)
        summed_square = torch.sum(stacked, dim=1) ** 2
        squared_sum = torch.sum(stacked ** 2, dim=1)
        pred_vec = 0.5 * (summed_square - squared_sum)
        return torch.sum(pred_vec, dim=1)


class DNN(nn.Module):
    def __init__(self, embed_dim, hidden, dropout, n_fields):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.dropout = dropout
        self.n_fields = n_fields
        self._assert_arg_error()
        self._set_up_components()

    def forward(self, emb_list):
        concat = torch.cat(emb_list, dim=1)
        pred_vec = self.fc(concat)
        return self.pred_layer(pred_vec).squeeze(-1)

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        components = list(self._yield_linear_block(self.hidden))
        self.fc = nn.Sequential(*components)

        kwargs = dict(
            in_features=self.hidden[-1],
            out_features=1,
        )
        self.pred_layer = nn.Linear(**kwargs)

    def _yield_linear_block(self, hidden):
        idx = 1
        while idx < len(hidden):
            yield nn.Sequential(
                nn.Linear(hidden[idx-1], hidden[idx]),
                nn.LayerNorm(hidden[idx]),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            )
            idx += 1

    def _assert_arg_error(self):
        CONDITION = (self.hidden[0]==self.embed_dim*self.n_fields)
        ERROR_MESSAGE = f"First hidden units must match input size: {self.embed_dim*self.n_fields}"
        assert CONDITION, ERROR_MESSAGE