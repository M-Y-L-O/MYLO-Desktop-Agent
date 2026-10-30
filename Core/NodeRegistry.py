import inspect
import os
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Type

_TRUTHY = {"1", "true", "yes", "on"}


def experimental_nodes_enabled() -> bool:
    """Whether experimental / auto-discovered torch.nn nodes are active.

    Reads ``ENABLE_EXPERIMENTAL_NODES`` from the environment (``.env`` via
    ``load_dotenv``). Defaults to off.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return os.getenv("ENABLE_EXPERIMENTAL_NODES", "").strip().lower() in _TRUTHY


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


def _params_from_signature(signature: inspect.Signature, *, skip: frozenset[str] = frozenset()) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for name, param in signature.parameters.items():
        if name in ("self", "args", "kwargs") or name in skip:
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        params[name] = name
    return params


def get_constructor_params(module_cls: Type[nn.Module]) -> Dict[str, str]:
    """Map node param names to matching ``__init__`` parameter names.

    Walks the MRO when a class only exposes ``*args, **kwargs`` (e.g. LSTM/GRU).
    """
    skip_by_base = {
        "RNNBase": frozenset({"mode"}),
    }

    for candidate in module_cls.__mro__:
        if candidate is object:
            break
        init = getattr(candidate, "__init__", None)
        if init is None:
            continue
        try:
            signature = inspect.signature(init)
        except (TypeError, ValueError):
            continue
        skip = skip_by_base.get(candidate.__name__, frozenset())
        params = _params_from_signature(signature, skip=skip)
        if params:
            # Drop factory kwargs that nodes never supply explicitly.
            params.pop("device", None)
            params.pop("dtype", None)
            return params
    return {}


# Containers / wrappers that are not suitable as graph nodes.
EXCLUDED_MODULES = {
    "Module",
    "Sequential",
    "ModuleList",
    "ModuleDict",
    "ParameterList",
    "ParameterDict",
    "Container",
    "DataParallel",
}

# Explicit execution / module overrides for non-standard forward semantics.
SPECIAL_NODES: Dict[str, Dict[str, Any]] = {
    "LSTM": {
        "execution_op": "recurrent_lstm",
    },
    "GRU": {
        "execution_op": "recurrent_gru",
    },
    "MultiheadAttention": {
        "execution_op": "self_attention",
        "module_cls": _SelfAttention,
    },
    "TransformerEncoderLayer": {
        "module_cls": _TransformerEncoderLayer,
    },
}

# Graph-level / alias nodes that are not plain torch.nn classes.
CUSTOM_NODES: Dict[str, Dict[str, Any]] = {
    "Input": {
        "module_cls": nn.Identity,
        "param_mapping": {},
    },
    "Output": {
        "module_cls": nn.Identity,
        "param_mapping": {},
    },
    "Add": {
        "module_cls": _Add,
        "param_mapping": {},
        "execution_op": "merge_add",
    },
    "Concat": {
        "module_cls": _Concat,
        "param_mapping": {"dim": "dim"},
        "execution_op": "merge_concat",
    },
    "Unsqueeze": {
        "module_cls": _Unsqueeze,
        "param_mapping": {"dim": "dim"},
    },
    "Squeeze": {
        "module_cls": _Squeeze,
        "param_mapping": {"dim": "dim"},
    },
    "ReduceMean": {
        "module_cls": _ReduceMean,
        # "axes" is kept as a legacy alias for "dim"
        "param_mapping": {"dim": "dim", "keepdim": "keepdim", "axes": "dim"},
    },
    "Transpose": {
        "module_cls": _Transpose,
        "param_mapping": {"dim0": "dim0", "dim1": "dim1"},
    },
    "Permute": {
        "module_cls": _Permute,
        "param_mapping": {"dims": "dims"},
    },
    "Reshape": {
        "module_cls": _Reshape,
        "param_mapping": {"target_shape": "target_shape"},
    },
}

# Nodes the editor / parser should treat as first-class (not experimental).
SUPPORTED_NODES = {
    "Linear",
    "Input",
    "Output",
    "Conv1d",
    "Conv2d",
    "LSTM",
    "GRU",
    "ReLU",
    "GELU",
    "SiLU",
    "Tanh",
    "Sigmoid",
    "Softmax",
    "LogSoftmax",
    "Dropout",
    "Flatten",
    "BatchNorm1d",
    "BatchNorm2d",
    "Add",
    "Concat",
    "Unsqueeze",
    "Squeeze",
    "ReduceMean",
    "Transpose",
    "Permute",
    "Identity",
    "Reshape",
    "LayerNorm",
    "Embedding",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "MaxPool1d",
    "AvgPool1d",
    "MaxPool2d",
    "AvgPool2d",
    "TransformerEncoderLayer",
    "MultiheadAttention",
}


class NodeRegistry:
    """Registry that maps textual node types to PyTorch operations and semantics.

    Built once per process from ``ENABLE_EXPERIMENTAL_NODES`` and then cached.
    """

    _registry: Dict[str, Dict[str, Any]] = {}
    _defaults_registered: bool = False
    _include_experimental: bool = False

    @classmethod
    def register(
        cls,
        node_type: str,
        module_cls: Type[nn.Module],
        param_mapping: Optional[Dict[str, str]] = None,
        execution_op: str = "standard_forward",
        *,
        supported: Optional[bool] = None,
        experimental: Optional[bool] = None,
    ):
        is_supported = (node_type in SUPPORTED_NODES) if supported is None else supported
        is_experimental = (not is_supported) if experimental is None else experimental
        cls._registry[node_type] = {
            "module_cls": module_cls,
            "param_mapping": param_mapping if param_mapping is not None else {},
            "execution_op": execution_op,
            "supported": is_supported,
            "experimental": is_experimental,
        }

    @classmethod
    def auto_register(cls, namespace=nn, *, include_experimental: Optional[bool] = None) -> int:
        """Discover ``nn.Module`` subclasses and register them with constructor params."""
        if include_experimental is None:
            include_experimental = experimental_nodes_enabled()

        registered = 0
        for name, obj in inspect.getmembers(namespace):
            if not inspect.isclass(obj):
                continue
            try:
                if not issubclass(obj, nn.Module):
                    continue
            except TypeError:
                continue
            if inspect.isabstract(obj) or name.startswith("_"):
                continue
            if name in EXCLUDED_MODULES or name.startswith("Lazy"):
                continue
            if name.endswith("Loss"):
                continue

            is_supported = name in SUPPORTED_NODES
            if not is_supported and not include_experimental:
                continue

            special = SPECIAL_NODES.get(name, {})
            module_cls = special.get("module_cls", obj)
            param_mapping = special.get("param_mapping")
            if param_mapping is None:
                param_mapping = get_constructor_params(module_cls)

            cls.register(
                node_type=name,
                module_cls=module_cls,
                param_mapping=param_mapping,
                execution_op=special.get("execution_op", "standard_forward"),
                supported=is_supported,
            )
            registered += 1
        return registered

    @classmethod
    def register_custom_nodes(cls) -> None:
        for name, spec in CUSTOM_NODES.items():
            cls.register(
                node_type=name,
                module_cls=spec["module_cls"],
                param_mapping=spec.get("param_mapping", {}),
                execution_op=spec.get("execution_op", "standard_forward"),
                supported=name in SUPPORTED_NODES,
            )

    @classmethod
    def ensure_initialized(cls) -> None:
        """Build the registry once for this process if it has not been built yet."""
        cls._register_defaults()

    @classmethod
    def get_module(cls, node_type: str, params: Dict[str, Any]) -> nn.Module:
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
        cls._register_defaults()
        if node_type not in cls._registry:
            return "standard_forward"
        return cls._registry[node_type].get("execution_op", "standard_forward")

    @classmethod
    def is_supported(cls, node_type: str) -> bool:
        cls._register_defaults()
        entry = cls._registry.get(node_type)
        return bool(entry and entry.get("supported"))

    @classmethod
    def is_experimental(cls, node_type: str) -> bool:
        cls._register_defaults()
        entry = cls._registry.get(node_type)
        return bool(entry and entry.get("experimental"))

    @classmethod
    def experimental_enabled(cls) -> bool:
        cls._register_defaults()
        return cls._include_experimental

    @classmethod
    def list_nodes(cls, *, supported_only: bool = False) -> List[str]:
        cls._register_defaults()
        names = sorted(cls._registry.keys())
        if supported_only:
            return [name for name in names if cls._registry[name].get("supported")]
        return names

    @classmethod
    def get_entry(cls, node_type: str) -> Optional[Dict[str, Any]]:
        cls._register_defaults()
        return cls._registry.get(node_type)

    @classmethod
    def _register_defaults(cls):
        if cls._defaults_registered:
            return
        include_experimental = experimental_nodes_enabled()
        cls._include_experimental = include_experimental
        cls.auto_register(include_experimental=include_experimental)
        cls.register_custom_nodes()
        cls._defaults_registered = True


# Built lazily on first access so ``.env`` is loaded before registration.
