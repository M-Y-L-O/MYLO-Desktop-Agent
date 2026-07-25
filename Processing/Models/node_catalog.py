import json
from Processing.Models.ModelEditing import ModelEditEngine

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


def _default_params(node_type: str):
    """Return default params for a given node type using ModelEditEngine helper."""
    return ModelEditEngine._default_params_for_type(node_type, 64)


def get_catalog() -> dict:
    """Return the full editor catalog, including groups, limits, operations, and per-node parameter schemas."""
    all_types = list(set(ModelEditEngine.INSERTABLE_TYPES + list(ModelEditEngine.PROTECTED_TYPES) + ["Concat", "Add"]))
    catalog = {
        "insertableTypes": ModelEditEngine.INSERTABLE_TYPES,
        "swappableActivations": ModelEditEngine.SWAPPABLE_ACTIVATIONS,
        "protectedTypes": sorted(ModelEditEngine.PROTECTED_TYPES),
        "limits": {
            "maxNodes": getattr(ModelEditEngine, "MAX_NODES", None),
            "maxDepth": getattr(ModelEditEngine, "MAX_DEPTH", None),
        },
        "operations": [
            "add_node", "remove_node", "update_node", "swap_node_type",
            "add_edge", "remove_edge", "insert_after", "scale_width", "add_skip_connection",
        ],
        "viewModes": ["summary", "detailed", "hybrid"],
        "nodeGroups": NODE_GROUPS,
        "nodeSchemas": {node_type: _default_params(node_type) for node_type in all_types},
        "nodeParameterBlueprints": NODE_PARAMETER_BLUEPRINTS,
        "nodePorts": {ntype: PORT_DEFINITIONS.get(ntype, DEFAULT_PORTS) for ntype in all_types},
        "compatibilityHints": COMPATIBILITY_HINTS,
        "sourceOnly": list(SOURCE_ONLY),
        "targetBlocklist": list(TARGET_BLOCKLIST),
        "operationSchemas": OPERATION_SCHEMAS,
    }
    return catalog
