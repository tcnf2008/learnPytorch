"""Small autograd examples used by the learning notebooks."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LinearRegressionData:
    """Synthetic one-dimensional linear regression data."""

    x: torch.Tensor
    y: torch.Tensor
    true_weight: float
    true_bias: float


@dataclass(frozen=True)
class ManualLinearRegressionResult:
    """Result from a hand-written autograd training loop."""

    weight: float
    bias: float
    losses: list[float]


def make_linear_regression_data(
    num_samples: int = 128,
    weight: float = 2.0,
    bias: float = -1.0,
    noise_std: float = 0.1,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> LinearRegressionData:
    """Create synthetic data for y = weight * x + bias + noise."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.linspace(-1.0, 1.0, num_samples, dtype=torch.float32).reshape(-1, 1)
    noise = noise_std * torch.randn(num_samples, 1, generator=generator)
    y = weight * x + bias + noise

    target_device = torch.device(device)
    return LinearRegressionData(
        x=x.to(target_device),
        y=y.to(target_device),
        true_weight=weight,
        true_bias=bias,
    )


def predict_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Compute y_hat = weight * x + bias without using nn.Linear."""
    return weight * x + bias


def mean_squared_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return mean squared error as a scalar Tensor."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have the same shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    return ((prediction - target) ** 2).mean()


def train_manual_linear_regression(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    learning_rate: float = 0.1,
    epochs: int = 200,
    seed: int = 42,
    initial_weight: float | None = None,
    initial_bias: float | None = None,
) -> ManualLinearRegressionResult:
    """Fit y = w * x + b using raw tensors, autograd, and manual SGD."""
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {tuple(x.shape)} and {tuple(y.shape)}")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    torch.manual_seed(seed)
    device = x.device
    weight = torch.randn((), device=device, requires_grad=True)
    bias = torch.zeros((), device=device, requires_grad=True)

    if initial_weight is not None:
        weight = torch.tensor(float(initial_weight), device=device, requires_grad=True)
    if initial_bias is not None:
        bias = torch.tensor(float(initial_bias), device=device, requires_grad=True)

    losses: list[float] = []
    for _ in range(epochs):
        prediction = predict_linear(x, weight, bias)
        loss = mean_squared_error(prediction, y)
        loss.backward()

        with torch.no_grad():
            weight -= learning_rate * weight.grad
            bias -= learning_rate * bias.grad
            weight.grad.zero_()
            bias.grad.zero_()

        losses.append(float(loss.detach().cpu()))

    return ManualLinearRegressionResult(
        weight=float(weight.detach().cpu()),
        bias=float(bias.detach().cpu()),
        losses=losses,
    )
