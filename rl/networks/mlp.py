from __future__ import annotations

from typing import Iterable, Sequence

import torch.nn as nn


def get_activation(name: str) -> type[nn.Module]:
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    if name == "elu":
        return nn.ELU
    if name == "gelu":
        return nn.GELU
    raise ValueError(f"Unsupported activation: {name}")


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str = "tanh",
    output_activation: str | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    activation_cls = get_activation(activation)

    prev_dim = int(input_dim)
    for hidden_dim in _as_ints(hidden_dims):
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation_cls())
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, int(output_dim)))
    if output_activation is not None:
        layers.append(get_activation(output_activation)())

    return nn.Sequential(*layers)


def _as_ints(values: Iterable[int]) -> list[int]:
    return [int(value) for value in values]
