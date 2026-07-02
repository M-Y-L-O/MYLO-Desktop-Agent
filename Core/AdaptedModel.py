import torch
import torch.nn as nn
from typing import Optional


class AdaptedModel(nn.Module):
    """Runtime adapter wrapper for input/output shape mismatches."""

    def __init__(
        self,
        base_model: nn.Module,
        input_adapter: Optional[nn.Module] = None,
        output_adapter: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_adapter is not None:
            x = self.input_adapter(x)
        out = self.base_model(x)
        if self.output_adapter is not None:
            out = self.output_adapter(out)
        return out

    @classmethod
    def from_shape_mismatch(
        cls,
        base_model: nn.Module,
        actual_input_dim: int,
        expected_input_dim: int,
        actual_output_dim: int,
        expected_output_dim: int,
    ) -> "AdaptedModel":
        input_adapter = None
        output_adapter = None

        if actual_input_dim != expected_input_dim:
            input_adapter = nn.Linear(actual_input_dim, expected_input_dim)

        if actual_output_dim != expected_output_dim:
            output_adapter = nn.Linear(expected_output_dim, actual_output_dim)

        return cls(base_model, input_adapter, output_adapter)
