from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import sqrtm


def covariance_from_features(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    features = features.double()
    if features.shape[0] < 2:
        raise ValueError("At least two features are required for covariance")
    mean = features.mean(dim=0)
    centered = features - mean
    covariance = centered.T @ centered / (features.shape[0] - 1)
    covariance = (covariance + covariance.T) * 0.5
    return mean, covariance


def symmetric_matrix_sqrt(matrix: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    matrix = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp_min(eps).sqrt()
    return (eigenvectors * eigenvalues.unsqueeze(0)) @ eigenvectors.T


def frechet_distance_loss(
    mean: torch.Tensor,
    covariance: torch.Tensor,
    reference_mean: torch.Tensor,
    reference_covariance: torch.Tensor,
    reference_sqrt: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Stable differentiable Fréchet distance between two Gaussian feature laws."""
    mean = mean.double()
    covariance = covariance.double()
    reference_mean = reference_mean.double()
    reference_covariance = reference_covariance.double()
    dimension = covariance.shape[0]
    identity = torch.eye(dimension, device=covariance.device, dtype=covariance.dtype)
    covariance = (covariance + covariance.T) * 0.5 + eps * identity
    reference_covariance = (
        (reference_covariance + reference_covariance.T) * 0.5 + eps * identity
    )
    if reference_sqrt is None:
        reference_sqrt = symmetric_matrix_sqrt(reference_covariance)
    else:
        reference_sqrt = reference_sqrt.double()
    middle = reference_sqrt @ covariance @ reference_sqrt
    middle = (middle + middle.T) * 0.5 + eps * identity
    # trace(sqrt(middle)) equals the sum of square roots of its eigenvalues.
    root_trace = torch.sqrt(torch.linalg.eigvalsh(middle).clamp_min(eps)).sum()
    mean_term = (mean - reference_mean).square().sum()
    covariance_term = torch.trace(covariance) + torch.trace(reference_covariance) - 2 * root_trace
    return (mean_term + covariance_term).clamp_min(0.0)


def normalized_fd(fd_value: torch.Tensor, offset: float = 0.01) -> torch.Tensor:
    """FD-loss normalization from the paper; denominator has no gradient."""
    return fd_value / (fd_value.detach() + float(offset))


class FeatureQueue(nn.Module):
    """Fixed-size differentiable queue statistics.

    Stored features are detached historical samples. Features from the current
    generator batch replace the oldest entries inside the differentiable mean and
    covariance, then are enqueued only after generator.backward().
    """

    def __init__(self, size: int, feature_dim: int) -> None:
        super().__init__()
        if size < 2:
            raise ValueError("queue size must be at least 2")
        self.size = int(size)
        self.feature_dim = int(feature_dim)
        self.register_buffer("features", torch.zeros(size, feature_dim, dtype=torch.float32))
        self.register_buffer("feature_sum", torch.zeros(feature_dim, dtype=torch.float64))
        self.register_buffer(
            "feature_outer", torch.zeros(feature_dim, feature_dim, dtype=torch.float64)
        )
        self.register_buffer("pointer", torch.zeros((), dtype=torch.long))
        self.register_buffer("filled", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def initialize(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected [N,{self.feature_dim}] features, got {tuple(features.shape)}"
            )
        if features.shape[0] < self.size:
            raise ValueError(f"Need at least {self.size} features to initialize queue")
        selected = features[-self.size :].detach().to(device=self.features.device, dtype=torch.float32)
        self.features.copy_(selected)
        selected64 = selected.double()
        self.feature_sum.copy_(selected64.sum(dim=0))
        self.feature_outer.copy_(selected64.T @ selected64)
        self.pointer.zero_()
        self.filled.fill_(True)

    def _positions(self, batch_size: int) -> torch.Tensor:
        if batch_size > self.size:
            raise ValueError("Current class batch cannot exceed queue size")
        start = int(self.pointer.item())
        return (torch.arange(batch_size, device=self.features.device) + start) % self.size

    def differentiable_stats(
        self, current_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(self.filled.item()):
            raise RuntimeError("Queue must be initialized before computing statistics")
        if current_features.ndim != 2 or current_features.shape[1] != self.feature_dim:
            raise ValueError("Unexpected current feature shape")
        current = current_features.double()
        positions = self._positions(current.shape[0])
        evicted = self.features.index_select(0, positions).double()
        combined_sum = self.feature_sum - evicted.sum(dim=0) + current.sum(dim=0)
        combined_outer = (
            self.feature_outer - evicted.T @ evicted + current.T @ current
        )
        mean = combined_sum / self.size
        covariance = (combined_outer - self.size * torch.outer(mean, mean)) / (self.size - 1)
        covariance = (covariance + covariance.T) * 0.5
        return mean, covariance

    @torch.no_grad()
    def enqueue(self, current_features: torch.Tensor) -> None:
        if not bool(self.filled.item()):
            raise RuntimeError("Queue must be initialized before enqueue")
        current = current_features.detach().to(device=self.features.device, dtype=torch.float32)
        positions = self._positions(current.shape[0])
        evicted = self.features.index_select(0, positions).double()
        current64 = current.double()
        self.feature_sum.add_(current64.sum(dim=0) - evicted.sum(dim=0))
        self.feature_outer.add_(current64.T @ current64 - evicted.T @ evicted)
        self.features.index_copy_(0, positions, current)
        self.pointer.fill_((int(self.pointer.item()) + current.shape[0]) % self.size)


def load_reference_stats(
    path: str | Path,
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    arrays = np.load(path)
    mean = torch.from_numpy(arrays["mean"]).to(device=device, dtype=torch.float64)
    covariance = torch.from_numpy(arrays["covariance"]).to(
        device=device, dtype=torch.float64
    )
    sqrt = symmetric_matrix_sqrt(covariance)
    return {
        "mean": mean,
        "covariance": covariance,
        "sqrt": sqrt,
        "count": int(arrays["count"]),
    }


def numpy_frechet_distance(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Evaluation-only FD in the learned CXR feature space."""
    real_features = np.asarray(real_features, dtype=np.float64)
    fake_features = np.asarray(fake_features, dtype=np.float64)
    mean_real = real_features.mean(axis=0)
    mean_fake = fake_features.mean(axis=0)
    covariance_real = np.cov(real_features, rowvar=False)
    covariance_fake = np.cov(fake_features, rowvar=False)
    covariance_real += np.eye(covariance_real.shape[0]) * eps
    covariance_fake += np.eye(covariance_fake.shape[0]) * eps
    covariance_mean, _ = sqrtm(covariance_real @ covariance_fake, disp=False)
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    value = (
        np.square(mean_real - mean_fake).sum()
        + np.trace(covariance_real)
        + np.trace(covariance_fake)
        - 2.0 * np.trace(covariance_mean)
    )
    return float(max(value, 0.0))
