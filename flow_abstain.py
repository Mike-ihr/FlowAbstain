import math
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.as_tensor(mask, dtype=self.weight.dtype))

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    def __init__(self, dim, hidden_features=512, hidden_layers=2):
        super().__init__()
        self.dim = dim
        layers = []
        features = [dim] + [hidden_features] * hidden_layers + [2 * dim]
        for in_f, out_f in zip(features[:-1], features[1:]):
            layers.append(MaskedLinear(in_f, out_f))
            if out_f != 2 * dim:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self._create_masks(hidden_features, hidden_layers)

    def _create_masks(self, hidden_features, hidden_layers):
        degrees = [np.arange(1, self.dim + 1)]
        for _ in range(hidden_layers):
            degrees.append(np.arange(hidden_features) % max(1, self.dim - 1) + 1)
        degrees.append(np.tile(np.arange(1, self.dim + 1), 2))

        linear_layers = [m for m in self.net if isinstance(m, MaskedLinear)]
        for i, layer in enumerate(linear_layers):
            if i == len(linear_layers) - 1:
                mask = degrees[i][:, None] < degrees[i + 1][None, :]
            else:
                mask = degrees[i][:, None] <= degrees[i + 1][None, :]
            layer.set_mask(mask.T.astype("float32"))

    def forward(self, x):
        mu, log_scale = self.net(x).chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale).clamp(min=-5.0, max=5.0)
        z = (x - mu) * torch.exp(-log_scale)
        log_det = -log_scale.sum(dim=-1)
        return z, log_det


class FlowBlock(nn.Module):
    def __init__(self, dim, hidden_features=512, hidden_layers=2):
        super().__init__()
        self.made = MADE(dim, hidden_features, hidden_layers)
        self.register_buffer("perm", torch.arange(dim - 1, -1, -1))

    def forward(self, x):
        z, log_det = self.made(x)
        return z[:, self.perm], log_det


class MAF(nn.Module):
    def __init__(self, dim, blocks=8, hidden_features=512, hidden_layers=2):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList(
            [FlowBlock(dim, hidden_features, hidden_layers) for _ in range(blocks)]
        )

    def log_prob(self, x):
        log_det_total = x.new_zeros(x.shape[0])
        z = x
        for block in self.blocks:
            z, log_det = block(z)
            log_det_total = log_det_total + log_det
        base = -0.5 * (z.pow(2).sum(dim=-1) + self.dim * math.log(2 * math.pi))
        return base + log_det_total

    def forward(self, x):
        return self.log_prob(x)


@dataclass
class FlowAbstainConfig:
    pca_dim: int = 128
    maf_blocks: int = 8
    hidden_features: int = 512
    hidden_layers: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 200
    patience: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class FlowAbstain:
    def __init__(self, config=None):
        self.config = config or FlowAbstainConfig()
        self.pca = PCA(n_components=self.config.pca_dim, whiten=True, random_state=0)
        self.flow = None

    def fit(self, hidden_states):
        x = np.asarray(hidden_states, dtype=np.float32)
        z = self.pca.fit_transform(x).astype("float32")

        rng = np.random.default_rng(0)
        indices = rng.permutation(len(z))
        split = max(1, int(0.9 * len(indices)))
        train_z = torch.from_numpy(z[indices[:split]])
        val_z = torch.from_numpy(z[indices[split:]]) if split < len(indices) else train_z

        self.flow = MAF(
            dim=z.shape[1],
            blocks=self.config.maf_blocks,
            hidden_features=self.config.hidden_features,
            hidden_layers=self.config.hidden_layers,
        ).to(self.config.device)

        optimizer = torch.optim.AdamW(
            self.flow.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        loader = DataLoader(
            TensorDataset(train_z),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        best_state = None
        best_val = -float("inf")
        bad_epochs = 0
        for _ in range(self.config.epochs):
            self.flow.train()
            for (batch,) in loader:
                batch = batch.to(self.config.device)
                loss = -self.flow.log_prob(batch).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            val_score = self._mean_log_prob_tensor(val_z)
            if val_score > best_val:
                best_val = val_score
                best_state = {k: v.detach().cpu().clone() for k, v in self.flow.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.config.patience:
                    break

        if best_state is not None:
            self.flow.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _mean_log_prob_tensor(self, z):
        self.flow.eval()
        scores = []
        loader = DataLoader(TensorDataset(z), batch_size=self.config.batch_size)
        for (batch,) in loader:
            scores.append(self.flow.log_prob(batch.to(self.config.device)).cpu())
        return torch.cat(scores).mean().item()

    @torch.no_grad()
    def score(self, hidden_states):
        x = np.asarray(hidden_states, dtype=np.float32)
        z = self.pca.transform(x).astype("float32")
        tensor = torch.from_numpy(z)
        self.flow.eval()
        scores = []
        loader = DataLoader(TensorDataset(tensor), batch_size=self.config.batch_size)
        for (batch,) in loader:
            scores.append(self.flow.log_prob(batch.to(self.config.device)).cpu().numpy())
        return np.concatenate(scores)

    def save(self, path):
        checkpoint = {
            "config": self.config.__dict__,
            "pca": self.pca,
            "flow_state": self.flow.state_dict(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path, map_location=None):
        checkpoint = torch.load(path, map_location=map_location or "cpu", weights_only=False)
        config = FlowAbstainConfig(**checkpoint["config"])
        model = cls(config)
        model.pca = checkpoint["pca"]
        model.flow = MAF(
            dim=model.pca.n_components_,
            blocks=config.maf_blocks,
            hidden_features=config.hidden_features,
            hidden_layers=config.hidden_layers,
        ).to(config.device)
        model.flow.load_state_dict(checkpoint["flow_state"])
        return model
