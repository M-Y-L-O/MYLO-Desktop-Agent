import inspect
from typing import Any, Dict, List, Optional, get_args, get_origin

from Core.NodeRegistry import NodeRegistry
from Processing.Models.ModelEditing import ModelEditEngine

# Process-lifetime cache: catalog is generated once from the registry + env flag.
_CATALOG_CACHE: Dict[bool, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Curated catalog overrides
# Types that appear in both NodeRegistry and these tables keep this hand-tuned
# structure. Everything else is generated from the registry at catalog build time.
# ---------------------------------------------------------------------------

# Grouping of node types for UI catalogs
NODE_GROUPS = {
    "Layers": [
        "Input",
        "Output",
        "Linear",
        "Dropout",
        "LayerNorm",
        "BatchNorm1d",
        "BatchNorm2d",
        "Flatten",
    ],
    "Activations": [
        "ReLU",
        "Tanh",
        "Sigmoid",
        "Identity",
    ],
    "Recurrent": [
        "LSTM",
        "GRU",
        "MultiheadAttention",
    ],
    "Merges": [
        "Add",
        "Concat",
    ]
}

# Blueprint detailing parameters, types, UI widgets, validation limits, and descriptions
NODE_PARAMETER_BLUEPRINTS = {
    "Input": {},
    "Output": {},
    "Linear": {
        "in_features": {
            "type": "integer",
            "label": "Input Features",
            "description": "Number of input features. Automatically inferred/propagated from preceding layer if left blank.",
            "default": 64,
            "min": 1,
            "max": 8192,
            "step": 1,
            "placeholder": "e.g. 64 (or blank to auto-infer)",
            "group": "Dimensions",
            "advanced": False,
            "required": False,
            "helpUrl": "https://pytorch.org/docs/stable/generated/torch.nn.Linear.html"
        },
        "out_features": {
            "type": "integer",
            "label": "Output Features",
            "description": "Number of output features.",
            "default": 64,
            "min": 1,
            "max": 8192,
            "step": 1,
            "placeholder": "e.g. 128",
            "group": "Dimensions",
            "advanced": False,
            "required": True,
            "helpUrl": "https://pytorch.org/docs/stable/generated/torch.nn.Linear.html"
        },
        "bias": {
            "type": "boolean",
            "label": "Use Bias",
            "description": "Whether the layer uses additive bias weights.",
            "default": True,
            "group": "Configuration",
            "advanced": True,
            "required": False
        }
    },
    "Dropout": {
        "p": {
            "type": "float",
            "label": "Dropout Rate (p)",
            "description": "Probability of an element to be zeroed out during training.",
            "default": 0.2,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "placeholder": "e.g. 0.2",
            "group": "Configuration",
            "advanced": False,
            "required": True,
            "helpUrl": "https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html"
        }
    },
    "LayerNorm": {
        "normalized_shape": {
            "type": "integer",
            "label": "Normalized Shape",
            "description": "Input dimensions to normalize. Inferred automatically if left blank.",
            "default": 64,
            "min": 1,
            "max": 8192,
            "step": 1,
            "placeholder": "e.g. 64 (or blank to auto-infer)",
            "group": "Dimensions",
            "advanced": False,
            "required": False,
            "helpUrl": "https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html"
        },
        "eps": {
            "type": "float",
            "label": "Epsilon",
            "description": "A value added to the denominator for numerical stability.",
            "default": 1e-5,
            "step": 1e-6,
            "placeholder": "e.g. 1e-5",
            "group": "Advanced Settings",
            "advanced": True,
            "required": False
        },
        "elementwise_affine": {
            "type": "boolean",
            "label": "Elementwise Affine",
            "description": "Whether the module has learnable per-element affine parameters.",
            "default": True,
            "group": "Advanced Settings",
            "advanced": True,
            "required": False
        }
    },
    "BatchNorm1d": {
        "num_features": {
            "type": "integer",
            "label": "Number of Features",
            "description": "Number of features (C) from an expected input of size (N, C) or (N, C, L). Use for Linear or Recurrent layers. Inferred automatically if left blank.",
            "default": 64,
            "min": 1,
            "max": 8192,
            "step": 1,
            "placeholder": "e.g. 64 (or blank to auto-infer)",
            "group": "Dimensions",
            "advanced": False,
            "required": False,
            "helpUrl": "https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html"
        },
        "eps": {
            "type": "float",
            "label": "Epsilon",
            "description": "A value added to the denominator for numerical stability.",
            "default": 1e-5,
            "step": 1e-6,
            "placeholder": "e.g. 1e-5",
            "group": "Advanced Settings",
            "advanced": True,
            "required": False
        },
        "momentum": {
            "type": "float",
            "label": "Momentum",
            "description": "The value used for the running_mean and running_var computation.",
            "default": 0.1,
            "step": 0.01,
            "placeholder": "e.g. 0.1",
            "group": "Advanced Settings",
            "advanced": True,
            "required": False
        },
        "affine": {
            "type": "boolean",
            "label": "Affine",
            "description": "Whether this module has learnable affine parameters (gamma/beta).",
            "default": True,
            "group": "Configuration",
            "advanced": True,
            "required": False
        },
        "track_running_stats": {
            "type": "boolean",
            "label": "Track Running Stats",
            "description": "When set to True, tracks running mean and variance during training.",
            "default": True,
            "group": "Configuration",
            "advanced": True,
            "required": False
        }
    },
    "BatchNorm2d": {
        "num_features": {
            "type": "integer",
            "label": "Number of Features",
            "description": "Number of features (C) from an expected input of size (N, C, H, W). Use for 2D Conv/Image layers. Inferred automatically if left blank.",
            "default": 64,
            "min": 1,
            "max": 8192,
            "step": 1,
            "placeholder": "e.g. 64 (or blank to auto-infer)",
            "group": "Dimensions",
            "advanced": False,
            "required": False,
            "helpUrl": "https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html"
        },
        "eps": {
            "type": "float",
            "label": "Epsilon",
            "description": "A value added to the denominator for numerical stability.",
            "default": 1e-5,
            "step": 1e-6,
            "placeholder": "e.g. 1e-5",
            "group": "Advanced Settings",
            "advanced": True,
            "required": False
        },
        "momentum": {
            "type": "float",
            "label": "Momentum",
            "description": "The value used for the running_mean and running_var computation.",
            "default": 0.1,
            "step": 0.01,
            "placeholder": "e.g. 0.1",
            "group": "Advanced Settings",
            "advanced": True,
            "required": False
        },
        "affine": {
            "type": "boolean",
            "label": "Affine",
            "description": "Whether this module has learnable affine parameters (gamma/beta).",
            "default": True,
            "group": "Configuration",
            "advanced": True,
            "required": False
        },
        "track_running_stats": {
            "type": "boolean",
            "label": "Track Running Stats",
            "description": "When set to True, tracks running mean and variance during training.",
            "default": True,
            "group": "Configuration",
            "advanced": True,
            "required": False
        }
    },
    "Flatten": {
        "start_dim": {
            "type": "integer",
            "label": "Start Dimension",
            "description": "First dimension to flatten.",
            "default": 1,
            "step": 1,
            "group": "Configuration",
            "advanced": False,
            "required": False,
            "helpUrl": "https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html"
        },
        "end_dim": {
            "type": "integer",
            "label": "End Dimension",
            "description": "Last dimension to flatten.",
            "default": -1,
            "step": 1,
            "group": "Configuration",
            "advanced": True,
            "required": False,
            "helpUrl": "https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html"
        }
    },
    "ReLU": {},
    "Tanh": {},
    "Sigmoid": {},
    "Identity": {},
    "LSTM": {
        "input_size": {
            "type": "integer",
            "label": "Input Size",
            "description": "The number of expected features in the input sequence. Inferred automatically if left blank.",
            "default": 64,
            "min": 1,
            "max": 8192,
            "step": 1,
            "placeholder": "e.g. 64 (or blank to auto-infer)",
            "group": "Dimensions",
            "advanced": False,
            "required": False,
            "helpUrl": "https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html"
        },
        "hidden_size": {
            "type": "integer",
            "label": "Hidden Size",
            "description": "The number of features in the hidden state h.",
            "default": 64,
            "min": 1,
            "max": 8192,
            "step": 1,
            "placeholder": "e.g. 128",
            "group": "Dimensions",
            "advanced": False,
            "required": True,
            "helpUrl": "https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html"
        },
        "num_layers": {
            "type": "integer",
            "label": "Number of Layers",
            "description": "Number of recurrent layers.",
            "default": 1,
            "min": 1,
            "max": 10,
            "step": 1,
            "placeholder": "e.g. 1",
            "group": "Configuration",
            "advanced": False,
            "required": False
        },
        "batch_first": {
            "type": "boolean",
            "label": "Batch First",
            "description": "If True, then input/output tensors are provided as (batch, seq, feature).",
            "default": True,
            "group": "Configuration",
            "advanced": True,
            "required": False
        },
        "dropout": {
            "type": "float",
            "label": "Dropout",
            "description": "If non-zero, introduces a Dropout layer on outputs of each recurrent layer except the last.",
            "default": 0.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "placeholder": "e.g. 0.1",
            "group": "Configuration",
            "advanced": True,
            "required": False
        },
        "bidirectional": {
            "type": "boolean",
            "label": "Bidirectional",
            "description": "If True, becomes a bidirectional recurrent layer.",
            "default": False,
            "group": "Configuration",
            "advanced": False,
            "required": False
        }
    },
    "GRU": {
        "input_size": {
            "type": "integer",
            "label": "Input Size",
            "description": "The number of expected features in the input sequence. Inferred automatically if left blank.",
            "default": 64,
            "min": 1,
            "max": 8192,
            "step": 1,
            "placeholder": "e.g. 64 (or blank to auto-infer)",
            "group": "Dimensions",
            "advanced": False,
            "required": False,
            "helpUrl": "https://pytorch.org/docs/stable/generated/torch.nn.GRU.html"
        },
        "hidden_size": {
            "type": "integer",
            "label": "Hidden Size",
            "description": "The number of features in the GRU hidden state.",
            "default": 64,
            "min": 1,
            "max": 8192,
            "step": 1,
            "placeholder": "e.g. 128",
            "group": "Dimensions",
            "advanced": False,
            "required": True,
            "helpUrl": "https://pytorch.org/docs/stable/generated/torch.nn.GRU.html"
        },
        "num_layers": {
            "type": "integer",
            "label": "Number of Layers",
            "description": "Number of recurrent layers.",
            "default": 1,
            "min": 1,
            "max": 10,
            "step": 1,
            "placeholder": "e.g. 1",
            "group": "Configuration",
            "advanced": False,
            "required": False
        },
        "batch_first": {
            "type": "boolean",
            "label": "Batch First",
            "description": "If True, then input/output tensors are provided as (batch, seq, feature).",
            "default": True,
            "group": "Configuration",
            "advanced": True,
            "required": False
        },
        "dropout": {
            "type": "float",
            "label": "Dropout",
            "description": "If non-zero, introduces a Dropout layer on outputs of each recurrent layer except the last.",
            "default": 0.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "placeholder": "e.g. 0.1",
            "group": "Configuration",
            "advanced": True,
            "required": False
        },
        "bidirectional": {
            "type": "boolean",
            "label": "Bidirectional",
            "description": "If True, becomes a bidirectional GRU.",
            "default": False,
            "group": "Configuration",
            "advanced": False,
            "required": False
        }
    },
    "MultiheadAttention": {
        "embed_dim": {
            "type": "integer",
            "label": "Embedding Dimension",
            "description": "Total dimension of the attention module.",
            "default": 64,
            "min": 1,
            "max": 8192,
            "step": 1,
            "placeholder": "e.g. 64",
            "group": "Dimensions",
            "advanced": False,
            "required": True,
            "helpUrl": "https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html"
        },
        "num_heads": {
            "type": "integer",
            "label": "Number of Heads",
            "description": "Number of parallel attention heads. Must divide embed_dim evenly.",
            "default": 8,
            "min": 1,
            "max": 64,
            "step": 1,
            "placeholder": "e.g. 8",
            "group": "Configuration",
            "advanced": False,
            "required": True,
            "helpUrl": "https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html"
        },
        "dropout": {
            "type": "float",
            "label": "Dropout Probability",
            "description": "Dropout probability on attention weights.",
            "default": 0.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "placeholder": "e.g. 0.1",
            "group": "Configuration",
            "advanced": True,
            "required": False
        },
        "batch_first": {
            "type": "boolean",
            "label": "Batch First",
            "description": "If True, then input/output tensors are provided as (batch, seq, feature).",
            "default": True,
            "group": "Configuration",
            "advanced": True,
            "required": False
        }
    },
    "Concat": {
        "dim": {
            "type": "integer",
            "label": "Concatenation Dimension",
            "description": "The dimension along which to concatenate the tensors.",
            "default": -1,
            "step": 1,
            "placeholder": "e.g. -1",
            "group": "Configuration",
            "advanced": False,
            "required": True,
            "helpUrl": "https://pytorch.org/docs/stable/generated/torch.cat.html"
        }
    },
    "Add": {}
}

PORT_DEFINITIONS = {
    "Input": {
        "inputs": [],
        "outputs": [{"id": "output", "label": "Output"}],
    },
    "Output": {
        "inputs": [{"id": "input", "label": "Input"}],
        "outputs": [],
    },
    "Linear": {
        "inputs":  [{"id": "input", "label": "Input"}],
        "outputs": [{"id": "output", "label": "Output"}],
    },
    "MultiheadAttention": {
        "inputs": [
            {"id": "query", "label": "Query", "required": True},
            {"id": "key",   "label": "Key",   "required": True},
            {"id": "value", "label": "Value", "required": True},
            {"id": "mask",  "label": "Key Padding Mask", "required": False},
        ],
        "outputs": [
            {"id": "attn_output",  "label": "Attention Output"},
            {"id": "attn_weights", "label": "Attention Weights"},
        ],
    },
    "LSTM": {
        "inputs": [
            {"id": "input", "label": "Input", "required": True},
            {"id": "h0",    "label": "h₀ (initial hidden)", "required": False},
            {"id": "c0",    "label": "c₀ (initial cell)",   "required": False},
        ],
        "outputs": [
            {"id": "output", "label": "Output sequence"},
            {"id": "h_n",    "label": "h_n (final hidden)"},
            {"id": "c_n",    "label": "c_n (final cell)"},
        ],
    },
    "GRU": {
        "inputs": [
            {"id": "input", "label": "Input", "required": True},
            {"id": "h0",    "label": "h₀ (initial hidden)", "required": False}
        ],
        "outputs": [
            {"id": "output", "label": "Output sequence"},
            {"id": "h_n",    "label": "h_n (final hidden)"}
        ],
    },
    "Concat": {
        "inputs":  [{"id": f"in_{i}", "label": f"Input {i+1}", "required": True} for i in range(2)],
        "outputs": [{"id": "output", "label": "Output"}],
        "dynamicInputs": {"min": 2, "max": 8},
    },
    "Add": {
        "inputs": [{"id": "in_0", "label": "Input 1", "required": True}, {"id": "in_1", "label": "Input 2", "required": True}],
        "outputs": [{"id": "output", "label": "Output"}],
        "dynamicInputs": {"min": 2, "max": 16},
    }
}

DEFAULT_PORTS = {
    "inputs":  [{"id": "input", "label": "Input"}],
    "outputs": [{"id": "output", "label": "Output"}],
}

# Compatibility hints for visual connections
COMPATIBILITY_HINTS = {
    "BatchNorm1d": {
        "accepts": ["Linear", "Identity", "Dropout", "LSTM", "GRU", "Flatten"],
        "note": "Requires 2D or 3D input (N, C) or (N, C, L)."
    },
    "BatchNorm2d": {
        "accepts": ["Conv2d", "Identity"],
        "note": "Requires 4D input (N, C, H, W)."
    },
    "Flatten": {
        "accepts": ["Linear", "Identity", "Dropout", "Conv1d", "Conv2d"],
        "note": "Use before Linear layers to reshape multi-dimensional inputs."
    },
    "Linear": {
        "accepts": ["*"],
        "note": "Typically placed after Flatten or other Linear layers."
    },
    "MultiheadAttention": {
        "accepts": ["Linear", "Identity", "LayerNorm"],
        "note": "Embedding dim must match evenly across Query, Key, and Value inputs."
    },
    "Concat": {
        "accepts": ["*"],
        "note": "All inputs must have the same shape except along the concatenation dimension."
    },
    "LayerNorm": {
        "accepts": ["Linear", "LSTM", "GRU", "MultiheadAttention", "Identity"],
        "note": "normalized_shape must match the last dimension of the incoming tensor."
    }
}

SOURCE_ONLY = ["*"]
TARGET_BLOCKLIST = ["*"]

# Documentation & dynamic forms definitions for each graph operation
OPERATION_SCHEMAS = {
    "add_node": {
        "required": ["nodeId", "type"],
        "optional": ["params", "connectFrom", "connectTo", "sourcePort", "targetPort"],
        "fields": {
            "nodeId":      {"type": "string",  "label": "Node ID (unique)", "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$"},
            "type":        {"type": "enum",    "label": "Node Type", "values": "nodeGroups"},
            "params":      {"type": "object",  "label": "Parameters", "schema": "nodeParameterBlueprints[type]"},
            "connectFrom": {"type": "nodeRef", "label": "Connect from node ID (optional)"},
            "connectTo":   {"type": "nodeRef", "label": "Connect to node ID (optional)"},
            "sourcePort":  {"type": "string",  "label": "Source Port (for connectFrom)", "default": "output"},
            "targetPort":  {"type": "string",  "label": "Target Port (for connectTo)", "default": "input"},
        },
        "example": {"nodeId": "fc1", "type": "Linear", "params": {"out_features": 128}, "connectFrom": "input"},
    },
    "remove_node": {
        "required": ["nodeId"],
        "optional": ["force"],
        "fields": {
            "nodeId": {"type": "nodeRef", "label": "Node to remove"},
            "force":  {"type": "boolean", "label": "Force bypass safety checks", "default": False}
        }
    },
    "update_node": {
        "required": ["nodeId"],
        "optional": ["type", "params", "newNodeId"],
        "fields": {
            "nodeId":    {"type": "nodeRef", "label": "Node to update"},
            "type":      {"type": "enum",    "label": "Change Type", "values": "nodeGroups"},
            "params":    {"type": "object",  "label": "New Parameters"},
            "newNodeId": {"type": "string",  "label": "Rename Node ID"}
        }
    },
    "swap_node_type": {
        "required": ["nodeId", "type"],
        "fields": {
            "nodeId": {"type": "nodeRef", "label": "Node to swap type"},
            "type":   {"type": "enum",    "label": "New Type"}
        }
    },
    "add_edge": {
        "required": ["source", "target"],
        "optional": ["sourcePort", "targetPort"],
        "fields": {
            "source":      {"type": "nodeRef", "label": "Source Node ID"},
            "target":      {"type": "nodeRef", "label": "Target Node ID"},
            "sourcePort":  {"type": "string",  "label": "Source Port", "default": "output"},
            "targetPort":  {"type": "string",  "label": "Target Port", "default": "input"}
        }
    },
    "remove_edge": {
        "required": ["source", "target"],
        "fields": {
            "source": {"type": "nodeRef", "label": "Source Node ID"},
            "target": {"type": "nodeRef", "label": "Target Node ID"}
        }
    },
    "insert_after": {
        "required": ["afterNodeId", "type"],
        "optional": ["nodeId", "params"],
        "fields": {
            "afterNodeId": {"type": "nodeRef", "label": "Insert after this node"},
            "type":        {"type": "enum",    "label": "Node Type", "values": "nodeGroups"},
            "nodeId":      {"type": "string",  "label": "New Node ID (auto-generated if omitted)"},
            "params":      {"type": "object",  "label": "Parameters"}
        },
        "example": {"afterNodeId": "fc1", "type": "ReLU"},
    },
    "scale_width": {
        "required": ["nodeId", "factor"],
        "fields": {
            "nodeId": {"type": "nodeRef", "label": "Node to scale"},
            "factor": {"type": "float",   "label": "Width scale factor", "min": 0.1, "max": 10.0, "default": 1.5}
        }
    },
    "add_skip_connection": {
        "required": ["from", "to"],
        "fields": {
            "from": {"type": "nodeRef", "label": "Source Node ID"},
            "to":   {"type": "nodeRef", "label": "Target Node ID"}
        }
    }
}

# Model architectural templates for starting designs.
# IMPORTANT: "input" and "output" are virtual terminals in ArchitectureDescriptor —
# do not create real nodes with those ids. Wire edges from "input" → … → "output".
TEMPLATES = {
    "mlp_classifier": {
        "label": "MLP Classifier (Tabular)",
        "description": "Fully-connected sequential network for tabular classification. Customizable hidden sizes, dropout, and depth.",
        "descriptor": {
            "model_name": "mlp_classifier",
            "input_shape": [-1, 64],
            "output_shape": [-1, 10],
            "nodes": [
                {"id": "flat",   "type": "Flatten",  "params": {"start_dim": 1}},
                {"id": "fc1",    "type": "Linear",   "params": {"in_features": 64, "out_features": 128}},
                {"id": "relu1",  "type": "ReLU",     "params": {}},
                {"id": "drop1",  "type": "Dropout",  "params": {"p": 0.3}},
                {"id": "fc2",    "type": "Linear",   "params": {"in_features": 128, "out_features": 64}},
                {"id": "relu2",  "type": "ReLU",     "params": {}},
                {"id": "out",    "type": "Linear",   "params": {"in_features": 64, "out_features": 10}},
            ],
            "edges": [
                {"source": "input", "target": "flat", "source_port": "output", "target_port": "input"},
                {"source": "flat",  "target": "fc1", "source_port": "output", "target_port": "input"},
                {"source": "fc1",   "target": "relu1", "source_port": "output", "target_port": "input"},
                {"source": "relu1", "target": "drop1", "source_port": "output", "target_port": "input"},
                {"source": "drop1", "target": "fc2", "source_port": "output", "target_port": "input"},
                {"source": "fc2",   "target": "relu2", "source_port": "output", "target_port": "input"},
                {"source": "relu2", "target": "out", "source_port": "output", "target_port": "input"},
                {"source": "out",   "target": "output", "source_port": "output", "target_port": "input"},
            ],
        },
    },
    "lstm_forecaster": {
        "label": "LSTM Time-Series Forecaster",
        "description": "Sequence model structure for predicting time-series values.",
        "descriptor": {
            "model_name": "lstm_forecaster",
            "input_shape": [-1, 10, 64],
            "output_shape": [-1, 1],
            "nodes": [
                {"id": "lstm1", "type": "LSTM", "params": {"input_size": 64, "hidden_size": 128, "batch_first": True}},
                {"id": "relu", "type": "ReLU", "params": {}},
                {"id": "out", "type": "Linear", "params": {"in_features": 128, "out_features": 1}},
            ],
            "edges": [
                {"source": "input", "target": "lstm1", "source_port": "output", "target_port": "input"},
                {"source": "lstm1", "target": "relu", "source_port": "output", "target_port": "input"},
                {"source": "relu", "target": "out", "source_port": "output", "target_port": "input"},
                {"source": "out", "target": "output", "source_port": "output", "target_port": "input"},
            ],
        },
    },
    "cnn_classifier": {
        "label": "1D CNN Classifier",
        "description": "Base network layout for 1D signals (like audio, sequences, sensor feeds).",
        "descriptor": {
            "model_name": "cnn_classifier",
            "input_shape": [-1, 1, 128],
            "output_shape": [-1, 5],
            "nodes": [
                {"id": "conv1", "type": "Conv1d", "params": {"in_channels": 1, "out_channels": 16, "kernel_size": 1}},
                {"id": "relu1", "type": "ReLU", "params": {}},
                {"id": "flat", "type": "Flatten", "params": {"start_dim": 1}},
                {"id": "fc", "type": "Linear", "params": {"in_features": 2048, "out_features": 5}},
            ],
            "edges": [
                {"source": "input", "target": "conv1", "source_port": "output", "target_port": "input"},
                {"source": "conv1", "target": "relu1", "source_port": "output", "target_port": "input"},
                {"source": "relu1", "target": "flat", "source_port": "output", "target_port": "input"},
                {"source": "flat", "target": "fc", "source_port": "output", "target_port": "input"},
                {"source": "fc", "target": "output", "source_port": "output", "target_port": "input"},
            ],
        },
    },
}


# ---------------------------------------------------------------------------
# Automatic catalog generation from NodeRegistry
# ---------------------------------------------------------------------------

_DIMENSION_PARAMS = {
    "in_features", "out_features", "in_channels", "out_channels", "num_features",
    "hidden_size", "input_size", "embed_dim", "num_embeddings", "embedding_dim",
    "normalized_shape", "d_model", "num_heads", "nhead", "dim_feedforward",
    "num_groups", "num_channels", "kernel_size", "stride", "padding", "output_padding",
    "dilation", "groups", "start_dim", "end_dim", "dim", "dim0", "dim1", "dims",
    "target_shape", "axes",
}

_ADVANCED_PARAMS = {
    "eps", "momentum", "inplace", "bias", "affine", "track_running_stats",
    "elementwise_affine", "padding_mode", "ceil_mode", "count_include_pad",
    "divisor_override", "scale_grad_by_freq", "sparse", "padding_idx",
    "norm_first", "approximate", "proj_size", "device", "dtype",
}

_ACTIVATION_NAMES = {
    "ReLU", "RReLU", "LeakyReLU", "PReLU", "Softplus", "ELU", "CELU", "SELU",
    "GELU", "SiLU", "Mish", "Hardswish", "Hardtanh", "Hardsigmoid", "Hardshrink",
    "Softshrink", "Tanh", "Tanhshrink", "Sigmoid", "Softmax", "LogSoftmax",
    "Softmin", "GLU", "Identity",
}


def _humanize_param(name: str) -> str:
    return name.replace("_", " ").strip().title()


def _type_name(annotation: Any) -> str:
    return getattr(annotation, "__name__", "") or ""


def _annotation_widget_type(annotation: Any, default: Any) -> str:
    from typing import Literal, Union

    if annotation is not inspect.Parameter.empty:
        origin = get_origin(annotation)
        args = [a for a in get_args(annotation) if a is not type(None)]

        if annotation is bool or _type_name(annotation) == "bool":
            return "boolean"
        if annotation is int or _type_name(annotation) == "int":
            return "integer"
        if annotation is float or _type_name(annotation) == "float":
            return "float"
        if annotation is str or _type_name(annotation) == "str":
            return "string"
        if origin is list or origin is tuple:
            return "array"
        if origin is Literal:
            return "enum"
        if origin is Union:
            if any(get_origin(a) is Literal for a in args):
                return "enum"
            if any(a is bool or _type_name(a) == "bool" for a in args) and len(args) == 1:
                return "boolean"
            if any(a is int or _type_name(a) == "int" for a in args):
                return "integer"
            if any(a is float or _type_name(a) == "float" for a in args):
                return "float"
            if any(a is str or _type_name(a) == "str" for a in args):
                return "string"
            if any(get_origin(a) in (list, tuple) or a in (list, tuple) for a in args):
                return "array"

    if default is not inspect.Parameter.empty and default is not None:
        if isinstance(default, bool):
            return "boolean"
        if isinstance(default, int):
            return "integer"
        if isinstance(default, float):
            return "float"
        if isinstance(default, str):
            return "string"
        if isinstance(default, (list, tuple)):
            return "array"

    return "string"


def _heuristic_widget_type(param_name: str, annotation: Any, default: Any) -> str:
    widget = _annotation_widget_type(annotation, default)
    if widget != "string":
        return widget
    if param_name in _DIMENSION_PARAMS or param_name.endswith(("_dim", "_size", "_features", "_channels")):
        return "integer"
    if param_name in {"p", "dropout", "eps", "momentum", "negative_slope", "alpha"}:
        return "float"
    if param_name.startswith("is_") or param_name.startswith("use_") or param_name in {
        "bias", "affine", "inplace", "bidirectional", "batch_first", "keepdim",
    }:
        return "boolean"
    return widget


def _literal_values(annotation: Any) -> Optional[List[Any]]:
    from typing import Literal, Union

    if annotation is inspect.Parameter.empty:
        return None
    origin = get_origin(annotation)
    if origin is Literal:
        return list(get_args(annotation))
    if origin is Union:
        values: List[Any] = []
        for arg in get_args(annotation):
            if get_origin(arg) is Literal:
                values.extend(get_args(arg))
        return values or None
    return None


def _pytorch_help_url(node_type: str, module_cls: type) -> Optional[str]:
    module_name = getattr(module_cls, "__module__", "") or ""
    if module_name.startswith("torch.nn"):
        return f"https://pytorch.org/docs/stable/generated/torch.nn.{node_type}.html"
    # Wrappers that stand in for a public torch.nn class
    if node_type in {"MultiheadAttention", "TransformerEncoderLayer"}:
        return f"https://pytorch.org/docs/stable/generated/torch.nn.{node_type}.html"
    return None


def _json_safe_value(value: Any) -> Any:
    """Coerce constructor defaults into JSON-serializable catalog values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, list):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if callable(value):
        return getattr(value, "__name__", None) or repr(value)
    # torch.dtype / enums / odd objects → stable string label
    return str(value)


def _param_field_blueprint(
    param_name: str,
    annotation: Any,
    default: Any,
    *,
    help_url: Optional[str] = None,
) -> Dict[str, Any]:
    has_default = default is not inspect.Parameter.empty
    widget = _heuristic_widget_type(param_name, annotation, default)
    # Callable defaults (e.g. Transformer.activation=relu) are exposed as strings.
    if has_default and callable(default) and not isinstance(default, type):
        widget = "string"

    field: Dict[str, Any] = {
        "type": widget,
        "label": _humanize_param(param_name),
        "description": f"Constructor parameter `{param_name}`.",
        "group": (
            "Dimensions" if param_name in _DIMENSION_PARAMS
            else "Advanced Settings" if param_name in _ADVANCED_PARAMS
            else "Configuration"
        ),
        "advanced": param_name in _ADVANCED_PARAMS or (has_default and param_name not in _DIMENSION_PARAMS),
        "required": not has_default,
    }

    if has_default and default is not None:
        field["default"] = _json_safe_value(default)

    if widget == "integer":
        field.setdefault("step", 1)
        if param_name in _DIMENSION_PARAMS:
            field.setdefault("min", 1)
            field.setdefault("max", 8192)
            field.setdefault("default", field.get("default", 64))
    elif widget == "float":
        field.setdefault("step", 0.01 if param_name not in {"eps"} else 1e-6)
        if param_name in {"p", "dropout", "negative_slope"}:
            field.setdefault("min", 0.0)
            field.setdefault("max", 1.0)

    literal_vals = _literal_values(annotation)
    if widget == "enum" and literal_vals:
        field["values"] = [_json_safe_value(v) for v in literal_vals]
        if "default" not in field and field["values"]:
            field["default"] = field["values"][0]

    if help_url:
        field["helpUrl"] = help_url

    return field


def _resolve_init_signature(module_cls: type):
    """Return (signature, skip_names) walking MRO like NodeRegistry param discovery."""
    skip_by_base = {"RNNBase": frozenset({"mode"})}
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
        usable = [
            (n, p) for n, p in signature.parameters.items()
            if n not in ("self", "args", "kwargs")
            and n not in skip
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            and n not in ("device", "dtype")
        ]
        if usable:
            return usable
    return []


def generate_parameter_blueprint(node_type: str) -> Dict[str, Any]:
    """Build a UI parameter blueprint from a registry module constructor."""
    entry = NodeRegistry.get_entry(node_type)
    if not entry:
        return {}

    module_cls = entry["module_cls"]
    param_mapping: Dict[str, str] = dict(entry.get("param_mapping") or {})
    help_url = _pytorch_help_url(node_type, module_cls)
    sig_by_name = {name: param for name, param in _resolve_init_signature(module_cls)}

    if not param_mapping:
        param_mapping = {name: name for name in sig_by_name}

    blueprint: Dict[str, Any] = {}
    seen_ctor_names = set()

    for node_param, ctor_name in param_mapping.items():
        if node_param.startswith("_") or ctor_name.startswith("_"):
            continue

        param = sig_by_name.get(ctor_name)

        # Alias for an already-emitted constructor param (e.g. axes -> dim)
        if ctor_name in seen_ctor_names and node_param != ctor_name:
            target = blueprint.get(ctor_name)
            if target:
                alias_field = dict(target)
                alias_field["label"] = _humanize_param(node_param)
                alias_field["description"] = f"Alias for `{ctor_name}`."
                alias_field["advanced"] = True
                blueprint[node_param] = alias_field
            continue

        if param is None:
            continue

        seen_ctor_names.add(ctor_name)
        blueprint[node_param] = _param_field_blueprint(
            node_param,
            param.annotation,
            param.default,
            help_url=help_url,
        )

    return blueprint


def _auto_group_for(node_type: str) -> str:
    if node_type in {"Add", "Concat"}:
        return "Merges"
    if node_type in _ACTIVATION_NAMES:
        return "Activations"
    if "Pool" in node_type:
        return "Pooling"
    if node_type.startswith("Conv"):
        return "Convolution"
    if "Norm" in node_type:
        return "Normalization"
    if node_type in {"LSTM", "GRU", "RNN"} or node_type.endswith("Cell"):
        return "Recurrent"
    if "Attention" in node_type or "Transformer" in node_type:
        return "Attention"
    if node_type in {
        "Unsqueeze", "Squeeze", "Transpose", "Permute", "Reshape", "Flatten", "ReduceMean",
    }:
        return "Shape"
    if node_type == "Embedding":
        return "Layers"
    return "Layers"


def build_parameter_blueprints(*, include_experimental: bool = False) -> Dict[str, Dict[str, Any]]:
    """Curated blueprints win; remaining registry nodes are auto-generated."""
    node_names = (
        NodeRegistry.list_nodes()
        if include_experimental
        else NodeRegistry.list_nodes(supported_only=True)
    )

    blueprints: Dict[str, Dict[str, Any]] = {}
    for name in node_names:
        if name in NODE_PARAMETER_BLUEPRINTS:
            blueprints[name] = NODE_PARAMETER_BLUEPRINTS[name]
        else:
            blueprints[name] = generate_parameter_blueprint(name)

    # Keep any curated entries that might not be in the filtered registry list
    for name, curated in NODE_PARAMETER_BLUEPRINTS.items():
        blueprints.setdefault(name, curated)

    return blueprints


def build_node_groups(*, include_experimental: bool = False) -> Dict[str, List[str]]:
    """Start from curated groups, then append registry nodes that are missing."""
    groups: Dict[str, List[str]] = {key: list(values) for key, values in NODE_GROUPS.items()}
    listed = {name for values in groups.values() for name in values}

    for name in NodeRegistry.list_nodes(supported_only=True):
        if name in listed:
            continue
        group_name = _auto_group_for(name)
        groups.setdefault(group_name, []).append(name)
        listed.add(name)

    if include_experimental:
        experimental = [
            name for name in NodeRegistry.list_nodes()
            if name not in listed and NodeRegistry.is_experimental(name)
        ]
        if experimental:
            groups["Experimental"] = sorted(experimental)

    return groups


def build_node_ports(node_types: List[str]) -> Dict[str, Any]:
    """Curated port definitions win; everyone else gets DEFAULT_PORTS."""
    return {name: PORT_DEFINITIONS.get(name, DEFAULT_PORTS) for name in node_types}


def _default_params(node_type: str) -> Dict[str, Any]:
    """Return default params for a given node type."""
    curated = ModelEditEngine._default_params_for_type(node_type, 64)
    if curated:
        return curated

    blueprint = NODE_PARAMETER_BLUEPRINTS.get(node_type)
    if blueprint is None:
        blueprint = generate_parameter_blueprint(node_type)

    defaults: Dict[str, Any] = {}
    for key, field in (blueprint or {}).items():
        if isinstance(field, dict) and "default" in field:
            defaults[key] = field["default"]
    return defaults


def _build_catalog(*, include_experimental: bool) -> dict:
    """Build a fresh catalog snapshot (not cached)."""
    NodeRegistry.ensure_initialized()
    blueprints = build_parameter_blueprints(include_experimental=include_experimental)
    groups = build_node_groups(include_experimental=include_experimental)
    all_types = sorted({name for names in groups.values() for name in names})

    return {
        "experimentalNodesEnabled": include_experimental,
        "insertableTypes": ModelEditEngine.INSERTABLE_TYPES,
        "swappableActivations": ModelEditEngine.SWAPPABLE_ACTIVATIONS,
        "protectedTypes": sorted(ModelEditEngine.PROTECTED_TYPES),
        "supportedTypes": NodeRegistry.list_nodes(supported_only=True),
        "experimentalTypes": [
            name for name in NodeRegistry.list_nodes()
            if NodeRegistry.is_experimental(name)
        ] if include_experimental else [],
        "limits": {
            "maxNodes": getattr(ModelEditEngine, "MAX_NODES", None),
            "maxDepth": getattr(ModelEditEngine, "MAX_DEPTH", None),
        },
        "operations": [
            "add_node", "remove_node", "update_node", "swap_node_type",
            "add_edge", "remove_edge", "insert_after", "scale_width", "add_skip_connection",
        ],
        "viewModes": ["summary", "detailed", "hybrid"],
        "nodeGroups": groups,
        "nodeSchemas": {node_type: _default_params(node_type) for node_type in all_types},
        "nodeParameterBlueprints": blueprints,
        "nodePorts": build_node_ports(all_types),
        "compatibilityHints": COMPATIBILITY_HINTS,
        "sourceOnly": list(SOURCE_ONLY),
        "targetBlocklist": list(TARGET_BLOCKLIST),
        "operationSchemas": OPERATION_SCHEMAS,
    }


def get_catalog(*, include_experimental: Optional[bool] = None) -> dict:
    """Return the editor catalog, generated once per process for the active env flag.

    ``ENABLE_EXPERIMENTAL_NODES`` controls whether auto-discovered experimental
    nodes are included. Pass ``include_experimental`` only to override (e.g. tests).
    """
    if include_experimental is None:
        # Prefer the registry's frozen decision after first init.
        NodeRegistry.ensure_initialized()
        include_experimental = NodeRegistry.experimental_enabled()

    cached = _CATALOG_CACHE.get(include_experimental)
    if cached is not None:
        return cached

    catalog = _build_catalog(include_experimental=include_experimental)
    _CATALOG_CACHE[include_experimental] = catalog
    return catalog


def clear_catalog_cache() -> None:
    """Drop cached catalogs (tests / rare runtime reconfiguration)."""
    _CATALOG_CACHE.clear()
