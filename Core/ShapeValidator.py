from typing import List, Optional, Tuple
import torch


class ShapeValidator:
    """Runtime shape checks used only when static descriptor validation is insufficient."""

    @staticmethod
    def validate_input(x: torch.Tensor, expected_shape: List[int]) -> Tuple[bool, Optional[str]]:
        if x.dim() != len(expected_shape):
            return False, f"Expected rank {len(expected_shape)}, got {x.dim()}"

        for idx, expected in enumerate(expected_shape):
            if expected == -1:
                continue
            actual = x.shape[idx]
            if actual != expected:
                return False, f"Dim {idx}: expected {expected}, got {actual}"
        return True, None

    @staticmethod
    def infer_runtime_shape(x: torch.Tensor) -> List[int]:
        return [-1 if dim is None else dim for dim in x.shape]

    @staticmethod
    def needs_adapter(actual: List[int], expected: List[int]) -> bool:
        if len(actual) != len(expected):
            return True
        for a, e in zip(actual, expected):
            if e != -1 and a != e:
                return True
        return False
