import torch
import torch.nn as nn
from typing import Dict, Any, Type

class NodeRegistry:
    """Registry that maps textual node types to PyTorch operations and semantics."""
    
    _registry: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, node_type: str, module_cls: Type[nn.Module], param_mapping: Dict[str, str], execution_op: str = "standard_forward"):
        cls._registry[node_type] = {
            "module_cls": module_cls,
            "param_mapping": param_mapping,
            "execution_op": execution_op
        }

    @classmethod
    def get_module(cls, node_type: str, params: Dict[str, Any]) -> nn.Module:
        if node_type not in cls._registry:
            # Fallback mappings for standard types if not explicitly registered
            cls._register_defaults()
            
        if node_type not in cls._registry:
            raise ValueError(f"Unknown node type: {node_type}")

        registry_info = cls._registry[node_type]
        module_cls = registry_info["module_cls"]
        param_mapping = registry_info["param_mapping"]

        kwargs = {}
        for param_key, param_value in params.items():
            if param_key in param_mapping:
                kwargs[param_mapping[param_key]] = param_value
            else:
                kwargs[param_key] = param_value

        if node_type in ("Softmax", "LogSoftmax") and "dim" not in kwargs:
            kwargs["dim"] = -1

        return module_cls(**kwargs)

    @classmethod
    def get_execution_semantic(cls, node_type: str) -> str:
        if node_type not in cls._registry:
             cls._register_defaults()
        if node_type not in cls._registry:
            return "standard_forward"
        return cls._registry[node_type].get("execution_op", "standard_forward")

    @classmethod
    def _register_defaults(cls):
        # Prevent duplicate registrations
        if "Linear" in cls._registry:
            return
            
        cls.register("Linear", nn.Linear, {"in_features": "in_features", "out_features": "out_features", "bias": "bias"})
        cls.register("Conv1d", nn.Conv1d, {"in_channels": "in_channels", "out_channels": "out_channels", "kernel_size": "kernel_size", "stride": "stride"})
        cls.register("Conv2d", nn.Conv2d, {"in_channels": "in_channels", "out_channels": "out_channels", "kernel_size": "kernel_size", "stride": "stride"})
        cls.register("LSTM", nn.LSTM, {"input_size": "input_size", "hidden_size": "hidden_size", "num_layers": "num_layers", "batch_first": "batch_first"}, execution_op="recurrent_lstm")
        cls.register("GRU", nn.GRU, {"input_size": "input_size", "hidden_size": "hidden_size", "num_layers": "num_layers", "batch_first": "batch_first"}, execution_op="recurrent_gru")
        cls.register("ReLU", nn.ReLU, {})
        cls.register("Tanh", nn.Tanh, {})
        cls.register("Sigmoid", nn.Sigmoid, {})
        cls.register("Softmax", nn.Softmax, {})
        cls.register("LogSoftmax", nn.LogSoftmax, {})
        cls.register("Dropout", nn.Dropout, {"p": "p"})
        cls.register("Flatten", nn.Flatten, {"start_dim": "start_dim", "end_dim": "end_dim"})
        cls.register("BatchNorm1d", nn.BatchNorm1d, {"num_features": "num_features"})
        cls.register("BatchNorm2d", nn.BatchNorm2d, {"num_features": "num_features"})

        class Add(nn.Module):
            def forward(self, x):
                return x

        class Concat(nn.Module):
            def __init__(self, dim: int = -1):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                return x

        cls.register("Add", Add, {}, execution_op="merge_add")
        cls.register("Concat", Concat, {"dim": "dim"}, execution_op="merge_concat")

        class Unsqueeze(nn.Module):
            def __init__(self, dim: int = 0):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                return torch.unsqueeze(x, self.dim)

        cls.register("Unsqueeze", Unsqueeze, {"dim": "dim"})

        class Squeeze(nn.Module):
            def __init__(self, dim: int | None = None):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                if self.dim is None:
                    return torch.squeeze(x)
                return torch.squeeze(x, self.dim)

        cls.register("Squeeze", Squeeze, {"dim": "dim"})

        class ReduceMean(nn.Module):
            def __init__(self, dim=None, keepdim: bool = False):
                super().__init__()
                self.dim = dim
                self.keepdim = keepdim

            def forward(self, x):
                if self.dim is None:
                    dims = tuple(range(1, x.dim()))
                    return torch.mean(x, dim=dims, keepdim=self.keepdim)
                if isinstance(self.dim, (list, tuple)):
                    return torch.mean(x, dim=tuple(self.dim), keepdim=self.keepdim)
                return torch.mean(x, dim=self.dim, keepdim=self.keepdim)

        cls.register("ReduceMean", ReduceMean, {"dim": "dim", "keepdim": "keepdim", "axes": "dim"})

        class Transpose(nn.Module):
            def __init__(self, dim0: int, dim1: int):
                super().__init__()
                self.dim0 = dim0
                self.dim1 = dim1

            def forward(self, x):
                return torch.transpose(x, self.dim0, self.dim1)

        cls.register("Transpose", Transpose, {"dim0": "dim0", "dim1": "dim1"})

        class Permute(nn.Module):
            def __init__(self, dims):
                super().__init__()
                self.dims = tuple(dims)

            def forward(self, x):
                return x.permute(*self.dims)

        cls.register("Permute", Permute, {"dims": "dims"})

        class Reshape(nn.Module):
            def __init__(self, target_shape=None, **kwargs):
                super().__init__()
                self.target_shape = tuple(target_shape) if target_shape is not None else None

            def forward(self, x):
                if self.target_shape is None:
                    return x
                batch_size = x.shape[0]
                return x.reshape(batch_size, *self.target_shape)

        cls.register("Identity", nn.Identity, {})
        cls.register("Reshape", Reshape, {"target_shape": "target_shape"})
        cls.register("LayerNorm", nn.LayerNorm, {"normalized_shape": "normalized_shape", "eps": "eps", "elementwise_affine": "elementwise_affine"})
        cls.register("Embedding", nn.Embedding, {"num_embeddings": "num_embeddings", "embedding_dim": "embedding_dim", "padding_idx": "padding_idx"})
        cls.register("ConvTranspose1d", nn.ConvTranspose1d, {"in_channels": "in_channels", "out_channels": "out_channels", "kernel_size": "kernel_size", "stride": "stride", "padding": "padding"})
        cls.register("ConvTranspose2d", nn.ConvTranspose2d, {"in_channels": "in_channels", "out_channels": "out_channels", "kernel_size": "kernel_size", "stride": "stride", "padding": "padding"})
        cls.register("MaxPool1d", nn.MaxPool1d, {"kernel_size": "kernel_size", "stride": "stride", "padding": "padding"})
        cls.register("AvgPool1d", nn.AvgPool1d, {"kernel_size": "kernel_size", "stride": "stride", "padding": "padding"})
        cls.register("MaxPool2d", nn.MaxPool2d, {"kernel_size": "kernel_size", "stride": "stride", "padding": "padding"})
        cls.register("AvgPool2d", nn.AvgPool2d, {"kernel_size": "kernel_size", "stride": "stride", "padding": "padding"})

        class TransformerEncoderLayer(nn.Module):
            def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, batch_first: bool = True, norm_first: bool = False, **kwargs):
                super().__init__()
                self.layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    batch_first=batch_first,
                    norm_first=norm_first,
                    **kwargs,
                )

            def forward(self, x):
                return self.layer(x)

        cls.register(
            "TransformerEncoderLayer",
            TransformerEncoderLayer,
            {
                "d_model": "d_model",
                "nhead": "nhead",
                "dim_feedforward": "dim_feedforward",
                "dropout": "dropout",
                "batch_first": "batch_first",
                "norm_first": "norm_first",
            },
        )

        class SelfAttention(nn.Module):
            def __init__(self, embed_dim: int, num_heads: int, batch_first: bool = True, **kwargs):
                super().__init__()
                self.attention = nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    batch_first=batch_first,
                    **kwargs,
                )

            def forward(self, x):
                output, _ = self.attention(x, x, x, need_weights=False)
                return output

        cls.register(
            "MultiheadAttention",
            SelfAttention,
            {"embed_dim": "embed_dim", "num_heads": "num_heads", "batch_first": "batch_first"},
            execution_op="self_attention",
        )

# defaults
NodeRegistry._register_defaults()
