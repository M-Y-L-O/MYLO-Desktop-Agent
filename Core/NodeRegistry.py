import torch
import torch.nn as nn
from typing import Dict, Any, Type


class _Add(nn.Module):
    def forward(self, x):
        return x


class _Concat(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x


class _Unsqueeze(nn.Module):
    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.unsqueeze(x, self.dim)


class _Squeeze(nn.Module):
    def __init__(self, dim: int | None = None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.squeeze(x) if self.dim is None else torch.squeeze(x, self.dim)


class _ReduceMean(nn.Module):
    def __init__(self, dim=None, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        dims = tuple(range(1, x.dim())) if self.dim is None else self.dim
        if isinstance(dims, list):
            dims = tuple(dims)
        return torch.mean(x, dim=dims, keepdim=self.keepdim)


class _Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)


class _Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = tuple(dims)

    def forward(self, x):
        return x.permute(*self.dims)


class _Reshape(nn.Module):
    def __init__(self, target_shape=None, **kwargs):
        super().__init__()
        self.target_shape = tuple(target_shape) if target_shape is not None else None

    def forward(self, x):
        if self.target_shape is None:
            return x
        return x.reshape(x.shape[0], *self.target_shape)


class _TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, batch_first: bool = True,
                 norm_first: bool = False, **kwargs):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=batch_first, norm_first=norm_first,
            **kwargs,
        )

    def forward(self, x):
        return self.layer(x)


class _SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, batch_first: bool = True, **kwargs):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            batch_first=batch_first, **kwargs,
        )

    def forward(self, x):
        output, _ = self.attention(x, x, x, need_weights=False)
        return output

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
        cls.register("Input", nn.Identity, {})
        cls.register("Output", nn.Identity, {})
        convolution_params = {name: name for name in (
            "in_channels", "out_channels", "kernel_size", "stride", "padding",
            "dilation", "groups", "bias", "padding_mode",
        )}
        cls.register("Conv1d", nn.Conv1d, convolution_params)
        cls.register("Conv2d", nn.Conv2d, convolution_params)
        recurrent_params = {name: name for name in (
            "input_size", "hidden_size", "num_layers", "bias", "batch_first",
            "dropout", "bidirectional",
        )}
        cls.register("LSTM", nn.LSTM, recurrent_params, execution_op="recurrent_lstm")
        cls.register("GRU", nn.GRU, recurrent_params, execution_op="recurrent_gru")
        cls.register("ReLU", nn.ReLU, {})
        cls.register("GELU", nn.GELU, {"approximate": "approximate"})
        cls.register("SiLU", nn.SiLU, {"inplace": "inplace"})
        cls.register("Tanh", nn.Tanh, {})
        cls.register("Sigmoid", nn.Sigmoid, {})
        cls.register("Softmax", nn.Softmax, {"dim": "dim"})
        cls.register("LogSoftmax", nn.LogSoftmax, {"dim": "dim"})
        cls.register("Dropout", nn.Dropout, {"p": "p", "inplace": "inplace"})
        cls.register("Flatten", nn.Flatten, {"start_dim": "start_dim", "end_dim": "end_dim"})
        batch_norm_params = {name: name for name in (
            "num_features", "eps", "momentum", "affine", "track_running_stats",
        )}
        cls.register("BatchNorm1d", nn.BatchNorm1d, batch_norm_params)
        cls.register("BatchNorm2d", nn.BatchNorm2d, batch_norm_params)

        cls.register("Add", _Add, {}, execution_op="merge_add")
        cls.register("Concat", _Concat, {"dim": "dim"}, execution_op="merge_concat")
        cls.register("Unsqueeze", _Unsqueeze, {"dim": "dim"})
        cls.register("Squeeze", _Squeeze, {"dim": "dim"})
        cls.register("ReduceMean", _ReduceMean, {"dim": "dim", "keepdim": "keepdim", "axes": "dim"})
        cls.register("Transpose", _Transpose, {"dim0": "dim0", "dim1": "dim1"})
        cls.register("Permute", _Permute, {"dims": "dims"})
        cls.register("Identity", nn.Identity, {})
        cls.register("Reshape", _Reshape, {"target_shape": "target_shape"})
        cls.register("LayerNorm", nn.LayerNorm, {"normalized_shape": "normalized_shape", "eps": "eps", "elementwise_affine": "elementwise_affine", "bias": "bias"})
        cls.register("Embedding", nn.Embedding, {"num_embeddings": "num_embeddings", "embedding_dim": "embedding_dim", "padding_idx": "padding_idx", "scale_grad_by_freq": "scale_grad_by_freq", "sparse": "sparse"})
        transpose_convolution_params = dict(convolution_params, output_padding="output_padding")
        cls.register("ConvTranspose1d", nn.ConvTranspose1d, transpose_convolution_params)
        cls.register("ConvTranspose2d", nn.ConvTranspose2d, transpose_convolution_params)
        max_pool_params = {name: name for name in ("kernel_size", "stride", "padding", "dilation", "ceil_mode")}
        avg_pool_params = {name: name for name in ("kernel_size", "stride", "padding", "ceil_mode", "count_include_pad")}
        cls.register("MaxPool1d", nn.MaxPool1d, max_pool_params)
        cls.register("AvgPool1d", nn.AvgPool1d, avg_pool_params)
        cls.register("MaxPool2d", nn.MaxPool2d, max_pool_params)
        cls.register("AvgPool2d", nn.AvgPool2d, dict(avg_pool_params, divisor_override="divisor_override"))

        cls.register(
            "TransformerEncoderLayer",
            _TransformerEncoderLayer,
            {
                "d_model": "d_model",
                "nhead": "nhead",
                "dim_feedforward": "dim_feedforward",
                "dropout": "dropout",
                "batch_first": "batch_first",
                "norm_first": "norm_first",
            },
        )

        cls.register(
            "MultiheadAttention",
            _SelfAttention,
            {"embed_dim": "embed_dim", "num_heads": "num_heads", "batch_first": "batch_first"},
            execution_op="self_attention",
        )

# defaults
NodeRegistry._register_defaults()
