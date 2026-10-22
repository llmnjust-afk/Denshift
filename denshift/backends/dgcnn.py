"""DGCNN classifier matching the PointCloud-C / ModelNet-C reference weights.

The architecture follows ``dgcnn_cls`` from the PointCloud-C repository so that
the released checkpoints load without a conversion step: four EdgeConv blocks
with 64/64/128/256 channels, a 1024-dimensional embedding, and max+average
pooling into a two-layer head.

Input is ``(B, 3, N)``; the wrapper in :mod:`denshift.blackbox` transposes for
you.
"""

from __future__ import annotations

import numpy as np

try:  # torch is an optional dependency of the core package
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "denshift.backends.dgcnn needs PyTorch; install it with `pip install torch`"
    ) from exc


def knn(x: "torch.Tensor", k: int) -> "torch.Tensor":
    """Indices of the ``k`` nearest neighbours of every point in ``x``.

    ``x`` is ``(B, C, N)`` and the result is ``(B, N, k)``.
    """
    inner = -2.0 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise = -xx - inner - xx.transpose(2, 1)
    return pairwise.topk(k=k, dim=-1)[1]


def get_graph_feature(
    x: "torch.Tensor", k: int = 20, idx: "torch.Tensor | None" = None
) -> "torch.Tensor":
    """Edge features ``[x_j - x_i, x_i]`` over the kNN graph, shape (B, 2C, N, k)."""
    batch_size, _, num_points = x.size()
    if idx is None:
        idx = knn(x, k=k)
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)

    num_dims = x.size(1)
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    centre = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    return torch.cat((feature - centre, centre), dim=3).permute(0, 3, 1, 2).contiguous()


class DGCNN(nn.Module):
    """Point-cloud classifier with the reference DGCNN parameter names."""

    def __init__(self, n_classes: int = 40, k: int = 20, dropout: float = 0.5) -> None:
        super().__init__()
        self.k = int(k)
        self.n_classes = int(n_classes)

        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(512, 1024, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

        self.linear1 = nn.Linear(2048, 512, bias=False)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, self.n_classes)
        self.dp1 = nn.Dropout(p=dropout)
        self.dp2 = nn.Dropout(p=dropout)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        batch_size = x.size(0)

        x = get_graph_feature(x, k=self.k)
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.2)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        return self.linear3(x)


def load_checkpoint(model: DGCNN, path: str) -> DGCNN:
    """Load a PointCloud-C style checkpoint, tolerating ``module.`` prefixes."""
    state = torch.load(path, map_location="cpu")
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a state dict")

    cleaned = {}
    for key, value in state.items():
        name = key[7:] if key.startswith("module.") else key
        cleaned[name] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if unexpected:
        raise ValueError(f"unexpected keys in checkpoint: {sorted(unexpected)[:5]}")
    if missing:
        raise ValueError(f"missing keys in checkpoint: {sorted(missing)[:5]}")
    return model


def to_black_box(
    model: DGCNN, device: str = "cpu", n_classes: int | None = None
):
    """Wrap a DGCNN for use with the density adapters."""
    from ..blackbox import BlackBoxModel

    def preprocess(points: np.ndarray) -> "torch.Tensor":
        arr = np.asarray(points, dtype=np.float32)
        return torch.from_numpy(arr.T).unsqueeze(0)

    return BlackBoxModel(
        module=model,
        preprocess=preprocess,
        device=device,
        n_classes=n_classes or model.n_classes,
    )
