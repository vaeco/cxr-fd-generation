#!/usr/bin/env python3
"""Fast CPU test for model shapes and differentiable queue FD."""
from __future__ import annotations

import torch

from src.fd_loss import FeatureQueue, frechet_distance_loss, normalized_fd
from src.models import ConditionalGenerator, ProjectionDiscriminator


def main() -> None:
    torch.manual_seed(7)
    generator = ConditionalGenerator(z_dim=16, base_channels=8)
    discriminator = ProjectionDiscriminator(base_channels=8)
    noise = torch.randn(2, 16)
    labels = torch.tensor([0, 1])
    images = generator(noise, labels)
    assert images.shape == (2, 1, 128, 128)
    scores = discriminator(images, labels)
    assert scores.shape == (2,)

    dimension = 16
    queue = FeatureQueue(size=32, feature_dim=dimension)
    queue.initialize(torch.randn(32, dimension))
    source = torch.randn(8, dimension, requires_grad=True)
    transform = torch.randn(dimension, dimension)
    current = source @ transform
    mean, covariance = queue.differentiable_stats(current)
    snapshot = queue.features.double().clone()
    snapshot[: current.shape[0]] = current.double()
    direct_mean = snapshot.mean(dim=0)
    direct_centered = snapshot - direct_mean
    direct_covariance = direct_centered.T @ direct_centered / (snapshot.shape[0] - 1)
    assert torch.allclose(mean, direct_mean, atol=1e-8, rtol=1e-6)
    assert torch.allclose(covariance, direct_covariance, atol=1e-8, rtol=1e-6)
    reference = torch.randn(64, dimension, dtype=torch.float64)
    reference_mean = reference.mean(dim=0)
    centered = reference - reference_mean
    reference_covariance = centered.T @ centered / (reference.shape[0] - 1)
    loss = normalized_fd(
        frechet_distance_loss(
            mean,
            covariance,
            reference_mean,
            reference_covariance,
        )
    )
    loss.backward()
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()
    assert float(source.grad.norm()) > 0
    queue.enqueue(current)
    assert torch.allclose(queue.feature_sum, queue.features.double().sum(0), atol=1e-8)
    assert torch.allclose(
        queue.feature_outer,
        queue.features.double().T @ queue.features.double(),
        atol=1e-8,
    )
    print(
        "[SELF_TEST_PASS] "
        f"generator_shape={tuple(images.shape)} "
        f"fd={float(loss):.6f} "
        f"fd_grad_norm={float(source.grad.norm()):.6f}"
    )


if __name__ == "__main__":
    main()
