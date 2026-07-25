
import copy
import json
import math
import os
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam

from Core.ArchitectureDescriptor import ArchitectureDescriptor
from Core.DescriptorModelBuilder import DescriptorModelBuilder
from Core.WeightCompatibilityEngine import WeightCompatibilityEngine
from Processing.Models.DescriptorHandling import descriptorToOnnx, dummyShapeFromDescriptor, extractStateDict
from Processing.Models.ONNXProcessing import analyseOnnx
from Processing.Data.DataPipeline import DataPipeline
from Utils.Other import make_json_serializable


try:
    from thop import profile as _thop_profile
    _THOP_AVAILABLE = True
except Exception:
    _THOP_AVAILABLE = False




def _project_path(filename: str) -> str:
    return os.path.join("temp_project", filename)





def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _tensor_summary(tensor: torch.Tensor) -> Dict[str, Any]:
    detached = tensor.detach().to("cpu")
    summary: Dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype).replace("torch.", ""),
        "device": str(detached.device),
        "numel": int(detached.numel()),
    }

    if detached.numel() == 0:
        summary["empty"] = True
        return summary

    floating = detached.float()
    summary.update(
        {
            "mean": _safe_float(floating.mean().item()),
            "std": _safe_float(floating.std(unbiased=False).item()),
            "min": _safe_float(floating.min().item()),
            "max": _safe_float(floating.max().item()),
            "abs_mean": _safe_float(floating.abs().mean().item()),
            "nan_count": int(torch.isnan(floating).sum().item()),
            "inf_count": int(torch.isinf(floating).sum().item()),
        }
    )
    return summary


def _summarize_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _tensor_summary(value)
    if isinstance(value, dict):
        return {str(key): _summarize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summarize_value(item) for item in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _count_parameters(module: torch.nn.Module) -> Dict[str, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    buffers = sum(buffer.numel() for buffer in module.buffers())
    return {
        "total": int(total),
        "trainable": int(trainable),
        "buffers": int(buffers),
    }


def _node_degree_maps(descriptor: ArchitectureDescriptor) -> Dict[str, Dict[str, int]]:
    incoming = Counter()
    outgoing = Counter()
    for edge in descriptor.edges:
        incoming[edge.target] += 1
        outgoing[edge.source] += 1
    return {
        "incoming": {node.id: int(incoming.get(node.id, 0)) for node in descriptor.nodes},
        "outgoing": {node.id: int(outgoing.get(node.id, 0)) for node in descriptor.nodes},
    }


def _module_kind(module: torch.nn.Module) -> str:
    return module.__class__.__name__




def _load_weights_if_available(model: torch.nn.Module, weights_path: str) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "requested": bool(weights_path),
        "loaded": False,
        "path": weights_path or "",
    }

    if not weights_path:
        return report

    full_path = weights_path if os.path.isabs(weights_path) else _project_path(weights_path)
    if not os.path.exists(full_path):
        report["error"] = f"Weights file not found: {weights_path}"
        return report

    try:
        with open(full_path, "rb") as weight_file:
            state_dict = extractStateDict(weight_file.read(), is_pytorch=True)
        if not state_dict:
            report["error"] = "Could not extract a state_dict from the weights file"
            return report

        transfer_report = WeightCompatibilityEngine.transfer_weights(state_dict, model)
        report.update(
            {
                "loaded": True,
                "matched_keys": transfer_report.get("matched_keys", []),
                "unmatched_source": transfer_report.get("unmatched_source", []),
                "unmatched_target": transfer_report.get("unmatched_target", []),
                "matched_key_count": len(transfer_report.get("matched_keys", [])),
                "unmatched_source_count": len(transfer_report.get("unmatched_source", [])),
                "unmatched_target_count": len(transfer_report.get("unmatched_target", [])),
            }
        )
    except Exception as exc:
        report["error"] = str(exc)

    return report




def _build_adjacency(descriptor: ArchitectureDescriptor) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Returns (forward adjacency, reverse adjacency) for the descriptor graph."""
    forward: Dict[str, List[str]] = defaultdict(list)
    reverse: Dict[str, List[str]] = defaultdict(list)
    for edge in descriptor.edges:
        forward[edge.source].append(edge.target)
        reverse[edge.target].append(edge.source)
    return forward, reverse


def _bfs_reachable(start_nodes: Set[str], adjacency: Dict[str, List[str]]) -> Set[str]:
    visited: Set[str] = set(start_nodes)
    queue: List[str] = list(start_nodes)
    while queue:
        current = queue.pop(0)
        for nxt in adjacency.get(current, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    return visited


def _graph_reachability(descriptor: ArchitectureDescriptor) -> Dict[str, Any]:

    forward, reverse = _build_adjacency(descriptor)


    output_side_starts: Set[str] = set()
    for edge in descriptor.edges:
        if edge.target == "output":
            output_side_starts.add(edge.source)
    reaches_output: Set[str] = _bfs_reachable(output_side_starts, reverse)
    reaches_output.add("output")


    input_side_starts: Set[str] = set()
    for edge in descriptor.edges:
        if edge.source == "input":
            input_side_starts.add(edge.target)
    reachable_from_input: Set[str] = _bfs_reachable(input_side_starts, forward)

    classification: Dict[str, str] = {}

    for node in descriptor.nodes:
        nid = node.id
        in_reach = nid in reachable_from_input
        out_reach = nid in reaches_output
        if in_reach and out_reach:
            classification[nid] = "alive"
        elif in_reach and not out_reach:
            classification[nid] = "dead_end"
        else:
            classification[nid] = "orphan"

    return {
        "classification": classification,
        "reachable_from_input": reachable_from_input,
        "reaches_output": reaches_output,
    }







def _flops_conv(module: nn.Module, output_shape: List[int]) -> int:
    if not isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                               nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        return 0
    if not output_shape or len(output_shape) < 2:
        return 0

    out_c = output_shape[1]
    spatial = output_shape[2:]
    spatial_elems = 1
    for s in spatial:
        spatial_elems *= max(int(s), 1)

    in_c = getattr(module, "in_channels", 0)
    groups = max(getattr(module, "groups", 1), 1)
    kernel = getattr(module, "kernel_size", (1,))
    k_ops = 1
    for k in kernel:
        k_ops *= int(k)


    flops = 2 * spatial_elems * out_c * (in_c // groups) * k_ops
    if getattr(module, "bias", None) is not None:
        flops += spatial_elems * out_c
    return int(flops)


def _flops_linear(module: nn.Module, output_shape: List[int]) -> int:
    if not isinstance(module, nn.Linear):
        return 0
    in_features = module.in_features
    out_features = module.out_features
    batch = 1
    for s in (output_shape or [])[:-1]:
        batch *= max(int(s), 1)
    flops = 2 * in_features * out_features * batch
    if module.bias is not None:
        flops += out_features * batch
    return int(flops)


def _flops_norm(module: nn.Module, output_shape: List[int]) -> int:
    norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                  nn.SyncBatchNorm, nn.LayerNorm, nn.GroupNorm,
                  nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)
    if not isinstance(module, norm_types):
        return 0
    numel = 1
    for s in (output_shape or []):
        numel *= max(int(s), 1)


    return int(numel * 5)


def _flops_pool(module: nn.Module, output_shape: List[int]) -> int:
    pool_names = (
        "MaxPool1d", "MaxPool2d", "MaxPool3d",
        "AvgPool1d", "AvgPool2d", "AvgPool3d",
        "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d",
        "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d",
        "FractionalMaxPool2d", "FractionalMaxPool3d",
        "LPPool1d", "LPPool2d",
    )
    if module.__class__.__name__ not in pool_names:
        return 0
    numel = 1
    for s in (output_shape or []):
        numel *= max(int(s), 1)
    return int(numel)


def _flops_rnn(module: nn.Module, output_shape: List[int]) -> int:
    if not isinstance(module, (nn.RNN, nn.GRU, nn.LSTM)):
        return 0
    if not output_shape or len(output_shape) < 3:
        return 0
    seq_len, batch, _ = (int(output_shape[0]), int(output_shape[1]), int(output_shape[2]))
    input_size = int(getattr(module, "input_size", 0))
    hidden_size = int(module.hidden_size)
    num_layers = int(module.num_layers)
    bidirectional = bool(getattr(module, "bidirectional", False))
    direction_mult = 2 if bidirectional else 1

    name = module.__class__.__name__
    if name == "LSTM":
        gates = 4
    elif name == "GRU":
        gates = 3
    else:
        gates = 1

    per_step = gates * hidden_size * (input_size + hidden_size + 1)
    return int(seq_len * batch * num_layers * direction_mult * per_step)


def _flops_embedding(module: nn.Module, output_shape: List[int]) -> int:
    if not isinstance(module, nn.Embedding):
        return 0
    return 0


def _flops_activation(module: nn.Module, output_shape: List[int]) -> int:
    act_types = (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.PReLU, nn.RReLU,
                 nn.SELU, nn.CELU, nn.GELU, nn.SiLU, nn.Mish,
                 nn.Sigmoid, nn.Tanh, nn.LogSigmoid, nn.Softplus,
                 nn.Softsign, nn.Hardtanh, nn.Hardsigmoid, nn.Hardswish,
                 nn.Softmax, nn.LogSoftmax)
    if not isinstance(module, act_types):
        return 0
    numel = 1
    for s in (output_shape or []):
        numel *= max(int(s), 1)
    return int(numel)


def _flops_dropout(module: nn.Module, output_shape: List[int]) -> int:
    drop_types = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
                  nn.AlphaDropout, nn.FeatureAlphaDropout)
    if not isinstance(module, drop_types):
        return 0
    return 0


def _flops_for_module(module: nn.Module, output_shape: Optional[List[int]]) -> int:
    if not output_shape:
        return 0
    return (
        _flops_conv(module, output_shape)
        + _flops_linear(module, output_shape)
        + _flops_norm(module, output_shape)
        + _flops_pool(module, output_shape)
        + _flops_rnn(module, output_shape)
        + _flops_embedding(module, output_shape)
        + _flops_activation(module, output_shape)
        + _flops_dropout(module, output_shape)
    )


def _estimate_flops_with_thop(model: nn.Module, dummy_input: torch.Tensor) -> Optional[Dict[str, Any]]:
    if not _THOP_AVAILABLE:
        return None
    try:
        with torch.no_grad():
            macs, _params = _thop_profile(model, inputs=(dummy_input,), verbose=False)
        return {
            "total_macs": int(macs),
            "total_flops": int(macs * 2),
            "source": "thop",
        }
    except Exception as exc:
        return {"error": str(exc), "source": "thop"}


def _estimate_flops_manual(node_modules: Dict[str, nn.Module],
                           activation_profile: Dict[str, Any]) -> Dict[str, Any]:
    per_node: Dict[str, int] = {}
    total = 0
    unsupported: List[str] = []
    for node_id, module in node_modules.items():
        activation = activation_profile.get(node_id, {})
        output = activation.get("output", {}) if isinstance(activation, dict) else {}
        out_shape = output.get("shape") if isinstance(output, dict) else None
        flops = _flops_for_module(module, out_shape)
        per_node[node_id] = int(flops)
        total += int(flops)
        if flops == 0 and out_shape is None and not isinstance(module, _NO_INPUT_MODULES):
            unsupported.append(node_id)
    return {
        "total_macs": total // 2,
        "total_flops": total,
        "per_node": per_node,
        "unsupported_nodes": unsupported,
        "source": "manual",
    }


_NO_INPUT_MODULES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
                     nn.Identity, nn.Flatten, nn.Unflatten)



_DTYPE_BYTES = {
    torch.float32: 4, torch.float: 4,
    torch.float16: 2, torch.half: 2,
    torch.bfloat16: 2,
    torch.float64: 8, torch.double: 8,
    torch.int64: 8, torch.long: 8,
    torch.int32: 4, torch.int: 4,
    torch.int16: 2, torch.short: 2,
    torch.int8: 1,
    torch.uint8: 1,
    torch.bool: 1,
}






def _dtype_size(dtype: torch.dtype) -> int:
    return _DTYPE_BYTES.get(dtype, 4)


def _estimate_memory(model: nn.Module,
                     node_modules: Dict[str, nn.Module],
                     activation_profile: Dict[str, Any],
                     dummy_input: torch.Tensor) -> Dict[str, Any]:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    input_bytes = dummy_input.numel() * dummy_input.element_size()

    activation_bytes_total = 0
    per_node_bytes: Dict[str, int] = {}
    for node_id, activation in activation_profile.items():
        if not isinstance(activation, dict):
            continue
        out = activation.get("output", {})
        if isinstance(out, dict) and "numel" in out and "dtype" in out:
            try:
                dt = getattr(torch, out["dtype"].split(".")[-1] if "." in out["dtype"] else out["dtype"], torch.float32)
            except Exception:
                dt = torch.float32
            size = int(out["numel"]) * _dtype_size(dt)
            per_node_bytes[node_id] = size
            activation_bytes_total += size



    forward_peak_estimate = input_bytes + activation_bytes_total + param_bytes + buffer_bytes



    adam_state_bytes = param_bytes * 2
    sgd_momentum_bytes = param_bytes
    training_peak_with_adam = forward_peak_estimate + param_bytes + adam_state_bytes
    training_peak_with_sgd = forward_peak_estimate + param_bytes + sgd_momentum_bytes

    def _mb(x: int) -> float:
        return round(x / (1024 * 1024), 6)

    return {
        "parameter_mb": _mb(param_bytes),
        "buffer_mb": _mb(buffer_bytes),
        "input_mb": _mb(input_bytes),
        "activation_mb_sum": _mb(activation_bytes_total),
        "activation_mb_peak_estimate": _mb(int(activation_bytes_total * 1.5)),
        "per_node_activation_mb": {k: _mb(v) for k, v in per_node_bytes.items()},
        "inference_peak_mb_estimate": _mb(int(forward_peak_estimate)),
        "training_peak_mb_estimate_adam": _mb(int(training_peak_with_adam)),
        "training_peak_mb_estimate_sgd_momentum": _mb(int(training_peak_with_sgd)),
    }







def _analyze_parameter_density(node_summaries: List[Dict[str, Any]],
                               parameter_details: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_params = sum(n["parameter_count"] for n in node_summaries)
    if total_params <= 0:
        return {
            "total_parameters": 0,
            "concentration_risk": "unknown",
            "per_node_pct": {},
            "gini": 0.0,
        }

    per_node_pct: Dict[str, float] = {}
    per_node_cum: List[Tuple[str, float]] = []
    for n in node_summaries:
        pct = (n["parameter_count"] / total_params) * 100.0
        per_node_pct[n["id"]] = round(pct, 4)
        per_node_cum.append((n["id"], pct))

    per_node_cum.sort(key=lambda x: x[1], reverse=True)
    top1_id, top1_pct = per_node_cum[0]
    top3_pct = sum(p for _, p in per_node_cum[:3])
    top5_pct = sum(p for _, p in per_node_cum[:5])


    if top1_pct >= 50:
        risk = "high"
    elif top1_pct >= 25 or top3_pct >= 80:
        risk = "medium"
    elif top3_pct >= 60:
        risk = "low"
    else:
        risk = "minimal"


    values = sorted(p for _, p in per_node_cum)
    n = len(values)
    gini = 0.0
    if n > 0 and sum(values) > 0:
        cumulative = 0.0
        for i, v in enumerate(values, start=1):
            cumulative += v
            gini += (2 * i - n - 1) * v
        gini = gini / (n * sum(values))


    type_breakdown: Dict[str, int] = {}
    for n in node_summaries:
        t = n.get("type", "unknown")
        type_breakdown[t] = type_breakdown.get(t, 0) + n["parameter_count"]

    return {
        "total_parameters": int(total_params),
        "largest_layer_id": top1_id,
        "largest_layer_pct": round(top1_pct, 4),
        "top3_concentration_pct": round(top3_pct, 4),
        "top5_concentration_pct": round(top5_pct, 4),
        "concentration_risk": risk,
        "per_node_pct": per_node_pct,
        "top_layers": [{"id": nid, "pct": round(p, 4)} for nid, p in per_node_cum[:5]],
        "gini": round(gini, 4),
        "parameters_by_node_type": {k: int(v) for k, v in type_breakdown.items()},
    }







_RNN_TYPES = {"LSTM", "GRU", "RNN", "LSTMCell", "GRUCell", "RNNCell"}
_ATTENTION_TYPES = {"MultiheadAttention", "Attention", "SelfAttention", "Transformer", "TransformerEncoderLayer", "TransformerDecoderLayer"}
_NORMALIZATION_TYPES = {"BatchNorm", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
                        "LayerNorm", "GroupNorm", "InstanceNorm", "InstanceNorm1d", "InstanceNorm2d"}
_DROPOUT_TYPES = {"Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "AlphaDropout", "FeatureAlphaDropout"}


def _detect_warnings(
    *,
    descriptor: ArchitectureDescriptor,
    shape_map: Dict[str, List[int]],
    node_summaries: List[Dict[str, Any]],
    parameter_details: List[Dict[str, Any]],
    forward_profile: Dict[str, Any],
    activation_profile: Dict[str, Any],
    weights_report: Dict[str, Any],
    onnx_report: Dict[str, Any],
    reachability: Dict[str, Any],
    degree_maps: Dict[str, Dict[str, int]],
    node_modules: Dict[str, nn.Module],
) -> Dict[str, Any]:
    warnings: List[Dict[str, Any]] = []
    counter = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}

    def _add(severity: str, category: str, message: str, *, node_id: Optional[str] = None,
             details: Optional[Dict[str, Any]] = None, suggestion: Optional[str] = None) -> None:
        if severity in counter:
            counter[severity] += 1
        warnings.append({
            "severity": severity,
            "category": category,
            "message": message,
            "node_id": node_id,
            "details": details or {},
            "suggestion": suggestion,
        })

    classification = reachability["classification"]
    reachable = reachability["reachable_from_input"]
    reaches_output = reachability["reaches_output"]


    for node in descriptor.nodes:
        nid = node.id
        kind = classification.get(nid, "orphan")
        if kind == "dead_end":
            _add(
                "high", "dead_layer",
                f"Layer '{nid}' is reachable from input but does not connect to the output — it has no effect on predictions.",
                node_id=nid,
                details={
                    "in_degree": degree_maps["incoming"].get(nid, 0),
                    "out_degree": degree_maps["outgoing"].get(nid, 0),
                },
                suggestion="Remove this branch or wire its output into the model's output path.",
            )
        elif kind == "orphan":
            _add(
                "high", "orphan_layer",
                f"Layer '{nid}' is not reachable from the input — it will never execute in a forward pass.",
                node_id=nid,
                suggestion="Check that this node is connected to the input or to a preceding layer.",
            )

    for node in descriptor.nodes:
        nid = node.id
        if nid in ("input", "output"):
            continue
        in_d = degree_maps["incoming"].get(nid, 0)
        out_d = degree_maps["outgoing"].get(nid, 0)
        if in_d == 0:
            _add(
                "medium", "dangling_input",
                f"Layer '{nid}' has no incoming edges (other than from 'input').",
                node_id=nid,
            )
        if out_d == 0:
            _add(
                "medium", "dangling_output",
                f"Layer '{nid}' has no outgoing edges (does not feed 'output' or any other layer).",
                node_id=nid,
            )


    if shape_map:

        for node in descriptor.nodes:
            shape = shape_map.get(node.id)
            if shape is None:
                _add(
                    "high", "shape_unknown",
                    f"Output shape could not be inferred for layer '{node.id}'.",
                    node_id=node.id,
                    suggestion="Check the input shape to this layer and the parameters of its operation.",
                )
                continue
            if any(s == 0 for s in shape):
                _add(
                    "high", "shape_invalid",
                    f"Layer '{node.id}' has a zero-sized dimension: {shape}.",
                    node_id=node.id,
                    details={"shape": list(shape)},
                )
            if any(isinstance(s, int) and s < 0 for s in shape):
                _add(
                    "low", "shape_dynamic",
                    f"Layer '{node.id}' has a dynamic dimension: {shape}.",
                    node_id=node.id,
                    details={"shape": list(shape)},
                )





        conv_like_kinds = {"Conv1d", "Conv2d", "Conv3d",
                           "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"}
        for edge in descriptor.edges:
            src_shape = shape_map.get(edge.source)
            if src_shape is None or len(src_shape) < 2:
                continue
            tgt_node = next((n for n in descriptor.nodes if n.id == edge.target), None)
            if tgt_node is None or tgt_node.id in ("input", "output"):
                continue
            tgt_module = node_modules[tgt_node.id] if tgt_node.id in node_modules else None
            if tgt_module is None:
                continue
            if _module_kind(tgt_module) not in conv_like_kinds:
                continue
            expected_in_channels = getattr(tgt_module, "in_channels", None)
            if expected_in_channels is None or expected_in_channels <= 0:
                continue
            actual_channels = src_shape[1]
            if actual_channels != expected_in_channels:
                _add(
                    "high", "shape_mismatch",
                    f"Edge {edge.source} -> {edge.target}: source output has "
                    f"{actual_channels} channels but target expects {expected_in_channels}.",
                    details={"source": edge.source, "target": edge.target,
                             "source_shape": list(src_shape), "expected_in_channels": expected_in_channels},
                    suggestion="Add a projection (e.g. Conv1x1) to align channel counts.",
                )


    type_sequence = [n.type for n in descriptor.nodes]

    for i in range(len(type_sequence) - 2):
        if (type_sequence[i] in _RNN_TYPES
                and type_sequence[i + 1] in _ATTENTION_TYPES
                and type_sequence[i + 2] in _RNN_TYPES):
            _add(
                "low", "suspicious_pattern",
                f"Unusual sequence: {type_sequence[i]} -> {type_sequence[i+1]} -> {type_sequence[i+2]} "
                "(stacking recurrent layers on both sides of attention is fragile and rarely beneficial).",
                details={"sequence": [type_sequence[i], type_sequence[i+1], type_sequence[i+2]]},
                suggestion="Pick one paradigm (recurrent or attention) and use it consistently.",
            )


    norm_run = 0
    max_norm_run = 0
    for t in type_sequence:
        if t in _NORMALIZATION_TYPES:
            norm_run += 1
            max_norm_run = max(max_norm_run, norm_run)
        else:
            norm_run = 0
    if max_norm_run >= 3:
        _add(
            "low", "suspicious_pattern",
            f"Detected {max_norm_run} consecutive normalization layers — likely redundant.",
            details={"max_consecutive_norm": max_norm_run},
        )


    activation_types = {"ReLU", "LeakyReLU", "GELU", "Sigmoid", "Tanh", "SiLU", "Mish"}
    prev_was_activation = False
    for t in type_sequence:
        if t in activation_types and prev_was_activation:
            _add(
                "low", "suspicious_pattern",
                f"Back-to-back activations: {t} preceded by another activation. One of them is redundant.",
            )
            break
        prev_was_activation = t in activation_types


    if len(descriptor.nodes) > 100:
        _add(
            "low", "deep_network",
            f"Model is very deep: {len(descriptor.nodes)} layers.",
            details={"node_count": len(descriptor.nodes)},
            suggestion="Consider residual/skip connections to ease gradient flow.",
        )


    total_params = sum(n["parameter_count"] for n in node_summaries)
    if total_params > 0:
        for n in node_summaries:
            pct = (n["parameter_count"] / total_params) * 100.0
            n["parameter_pct"] = round(pct, 4)
            if pct >= 50:
                _add(
                    "high", "param_concentration",
                    f"Layer '{n['id']}' holds {pct:.1f}% of all parameters — single point of failure / memory bottleneck.",
                    node_id=n["id"],
                    details={"pct": round(pct, 2), "params": n["parameter_count"]},
                    suggestion="Consider factorizing this layer (low-rank, depthwise separable, grouped).",
                )
            elif pct >= 25:
                _add(
                    "medium", "param_concentration",
                    f"Layer '{n['id']}' holds {pct:.1f}% of all parameters.",
                    node_id=n["id"],
                    details={"pct": round(pct, 2), "params": n["parameter_count"]},
                )


    for n in node_summaries:
        kind = n.get("module_kind", "")
        nid = n["id"]
        if kind in _DROPOUT_TYPES:
            _add(
                "info", "dropout_in_inference",
                f"Dropout layer '{nid}' is present — correctly disabled in eval(), but contributes forward overhead.",
                node_id=nid,
                details={"module": kind, "params": n["parameter_count"]},
            )
        if kind.startswith("BatchNorm"):
            _add(
                "info", "batchnorm_eval",
                f"BatchNorm '{nid}' uses running statistics in eval mode (correct).",
                node_id=nid,
                details={"module": kind},
            )


    if not forward_profile.get("succeeded", False):
        _add(
            "critical", "forward_failed",
            f"Forward pass failed: {forward_profile.get('error', 'unknown error')}",
            details={"error": forward_profile.get("error")},
            suggestion="Inspect the input shape and the parameters of the first failing layer.",
        )
    else:
        latency_ms = forward_profile.get("latency_ms")
        if latency_ms is not None and latency_ms > 1000:
            _add(
                "medium", "slow_inference",
                f"Forward pass took {latency_ms:.1f} ms on CPU — consider optimizing or running on GPU.",
                details={"latency_ms": latency_ms},
            )
        out = forward_profile.get("output", {})
        if isinstance(out, dict):
            if out.get("nan_count", 0) > 0 or out.get("inf_count", 0) > 0:
                _add(
                    "critical", "numerical_issue",
                    f"Forward output contains {out.get('nan_count', 0)} NaN and {out.get('inf_count', 0)} Inf values.",
                    details={"nan_count": out.get("nan_count"), "inf_count": out.get("inf_count")},
                    suggestion="Check weight initialization and any division / log / softmax layers.",
                )


    for node_id, activation in activation_profile.items():
        if not isinstance(activation, dict):
            continue
        out = activation.get("output", {})
        if not isinstance(out, dict):
            continue
        if out.get("nan_count", 0) > 0 or out.get("inf_count", 0) > 0:
            _add(
                "high", "activation_numerical_issue",
                f"Layer '{node_id}' output contains NaN/Inf — {out.get('nan_count', 0)} NaN, {out.get('inf_count', 0)} Inf.",
                node_id=node_id,
                details={"nan_count": out.get("nan_count"), "inf_count": out.get("inf_count")},
            )
        std = out.get("std")
        mean = out.get("mean")
        if std is not None and mean is not None and abs(mean) < 1e-7 and std < 1e-7 and out.get("numel", 0) > 1:
            _add(
                "medium", "dead_activations",
                f"Layer '{node_id}' output is effectively constant (mean≈0, std≈0). "
                "Likely dying/collapsed activations.",
                node_id=node_id,
                details={"mean": mean, "std": std},
            )


    if weights_report.get("requested"):
        if not weights_report.get("loaded", False):
            _add(
                "high", "weights_not_loaded",
                f"Weights file was specified but could not be loaded: {weights_report.get('error', 'unknown error')}",
                details={"error": weights_report.get("error")},
            )
        else:
            unmatched_src = weights_report.get("unmatched_source_count", 0)
            unmatched_tgt = weights_report.get("unmatched_target_count", 0)
            if unmatched_src > 0 or unmatched_tgt > 0:
                severity = "medium" if (unmatched_src + unmatched_tgt) < 5 else "high"
                _add(
                    severity, "weight_mismatch",
                    f"Weight transfer: {unmatched_src} source keys and {unmatched_tgt} target keys did not match.",
                    details={
                        "unmatched_source": weights_report.get("unmatched_source", [])[:20],
                        "unmatched_target": weights_report.get("unmatched_target", [])[:20],
                    },
                    suggestion="Architecture and weights are not fully compatible — some parameters will stay at their initial values.",
                )


    if onnx_report.get("requested") and not onnx_report.get("exported", False):
        _add(
            "high", "onnx_export_failed",
            f"ONNX export failed: {onnx_report.get('error', 'unknown error')}",
            details={"error": onnx_report.get("error")},
        )


    total_warnings = len(warnings)
    by_category: Dict[str, int] = {}
    for w in warnings:
        by_category[w["category"]] = by_category.get(w["category"], 0) + 1

    return {
        "total": total_warnings,
        "by_severity": counter,
        "by_category": by_category,
        "items": warnings,
    }







def _build_performance_insights(
    *,
    model: nn.Module,
    node_modules: Dict[str, nn.Module],
    dummy_input: torch.Tensor,
    activation_profile: Dict[str, Any],
    node_summaries: List[Dict[str, Any]],
    parameter_details: List[Dict[str, Any]],
    forward_profile: Dict[str, Any],
) -> Dict[str, Any]:

    thop_result = _estimate_flops_with_thop(model, dummy_input)
    manual_result = _estimate_flops_manual(node_modules, activation_profile)

    if thop_result and "total_flops" in thop_result:
        flops_block = {
            "total_flops": int(thop_result["total_flops"]),
            "total_macs": int(thop_result["total_macs"]),
            "giga_flops": round(thop_result["total_flops"] / 1e9, 6),
            "mega_flops": round(thop_result["total_flops"] / 1e6, 6),
            "per_node": manual_result.get("per_node", {}),
            "source": thop_result["source"],
            "per_node_source": manual_result["source"],
        }
    else:
        flops_block = {
            "total_flops": int(manual_result["total_flops"]),
            "total_macs": int(manual_result["total_macs"]),
            "giga_flops": round(manual_result["total_flops"] / 1e9, 6),
            "mega_flops": round(manual_result["total_flops"] / 1e6, 6),
            "per_node": manual_result["per_node"],
            "source": manual_result["source"],
            "unsupported_nodes": manual_result.get("unsupported_nodes", []),
        }


    memory_block = _estimate_memory(model, node_modules, activation_profile, dummy_input)


    density_block = _analyze_parameter_density(node_summaries, parameter_details)

    latency_ms = forward_profile.get("latency_ms")

    return {
        "flops": flops_block,
        "memory": memory_block,
        "parameter_density": density_block,
        "latency_ms": latency_ms,
    }


def _primary_tensor(value: Any) -> torch.Tensor:
    """Extract the prediction tensor from common model return containers."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)) and value:
        return _primary_tensor(value[0])
    if isinstance(value, dict) and value:
        for key in ("logits", "output", "prediction", "predictions"):
            if key in value:
                return _primary_tensor(value[key])
        return _primary_tensor(next(iter(value.values())))
    raise TypeError(f"Model returned unsupported output type: {type(value).__name__}")


def _align_training_tensors(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    problem_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if problem_type == "classification":
        if predictions.ndim == 1:
            predictions = predictions.unsqueeze(0)
        if predictions.ndim > 2:
            predictions = predictions.reshape(-1, predictions.shape[-1])
        return predictions, targets.reshape(-1).long()

    predictions = predictions.float()
    targets = targets.float()
    if predictions.shape != targets.shape:
        if predictions.numel() == targets.numel():
            targets = targets.reshape_as(predictions)
        elif predictions.ndim == 1 and targets.ndim == 2 and targets.shape[-1] == 1:
            targets = targets.squeeze(-1)
        elif predictions.ndim == 2 and predictions.shape[-1] == 1 and targets.ndim == 1:
            targets = targets.unsqueeze(-1)
        else:
            raise ValueError(
                f"Prediction shape {list(predictions.shape)} does not match "
                f"target shape {list(targets.shape)}"
            )
    return predictions, targets


def _evaluate_dataset(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    problem_type: str,
    target_scaler: Any = None,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    sample_count = 0
    predictions_all: List[torch.Tensor] = []
    targets_all: List[torch.Tensor] = []
    started = time.perf_counter()

    with torch.no_grad():
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            predictions, targets = _align_training_tensors(
                _primary_tensor(model(features)), targets, problem_type
            )
            loss = criterion(predictions, targets)
            batch_count = int(targets.shape[0])
            total_loss += float(loss.item()) * batch_count
            sample_count += batch_count
            predictions_all.append(predictions.detach().cpu())
            targets_all.append(targets.detach().cpu())

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    result: Dict[str, Any] = {
        "sample_count": sample_count,
        "loss": round(total_loss / max(sample_count, 1), 8),
        "evaluation_time_ms": round(elapsed_ms, 4),
        "throughput_samples_per_second": round(sample_count / max(elapsed_ms / 1000.0, 1e-9), 4),
    }
    if not predictions_all:
        return result

    predictions = torch.cat(predictions_all)
    targets = torch.cat(targets_all)
    if problem_type == "classification":
        predicted_classes = predictions.argmax(dim=-1)
        class_count = int(predictions.shape[-1])
        accuracy = (predicted_classes == targets).float().mean().item()
        per_class = []
        confusion = torch.zeros((class_count, class_count), dtype=torch.int64)
        for actual, predicted in zip(targets.tolist(), predicted_classes.tolist()):
            if 0 <= actual < class_count and 0 <= predicted < class_count:
                confusion[actual, predicted] += 1
        for class_index in range(class_count):
            tp = int(confusion[class_index, class_index])
            fp = int(confusion[:, class_index].sum()) - tp
            fn = int(confusion[class_index, :].sum()) - tp
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            per_class.append({
                "class_index": class_index,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "support": int(confusion[class_index, :].sum()),
            })
        result["metrics"] = {
            "accuracy": round(accuracy, 6),
            "macro_precision": round(sum(x["precision"] for x in per_class) / max(class_count, 1), 6),
            "macro_recall": round(sum(x["recall"] for x in per_class) / max(class_count, 1), 6),
            "macro_f1": round(sum(x["f1"] for x in per_class) / max(class_count, 1), 6),
            "per_class": per_class,
            "confusion_matrix": confusion.tolist(),
        }
    else:
        if target_scaler is not None:
            scale = torch.as_tensor(target_scaler.scale_, dtype=predictions.dtype)
            mean = torch.as_tensor(target_scaler.mean_, dtype=predictions.dtype)
            predictions = predictions * scale + mean
            targets = targets * scale + mean
        errors = predictions - targets
        mse = torch.mean(errors.square()).item()
        mae = torch.mean(errors.abs()).item()
        target_mean = torch.mean(targets)
        ss_res = torch.sum(errors.square()).item()
        ss_total = torch.sum((targets - target_mean).square()).item()
        r2 = 1.0 - (ss_res / ss_total) if ss_total > 1e-12 else None
        nonzero = targets.abs() > 1e-8
        mape = (
            torch.mean((errors[nonzero].abs() / targets[nonzero].abs())).item() * 100.0
            if nonzero.any() else None
        )
        result["metrics"] = {
            "mae": round(mae, 8),
            "mse": round(mse, 8),
            "rmse": round(math.sqrt(mse), 8),
            "r2": round(r2, 8) if r2 is not None else None,
            "mape_percent": round(mape, 6) if mape is not None else None,
        }
    return result


def _retrain_and_validate(
    model: nn.Module,
    descriptor: ArchitectureDescriptor,
    *,
    csv_path: str,
    input_features: List[str],
    output_features: List[str],
    epochs: int,
    problem_type: str,
    batch_size: int,
    validation_split: float,
    learning_rate: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not csv_path:
        raise ValueError("csv_path is required when mode='retrain'")
    if not input_features:
        raise ValueError("input_features must contain at least one column")
    if not output_features:
        raise ValueError("output_features must contain at least one column")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")

    sequence_length = descriptor.input_shape[1] if len(descriptor.input_shape) == 3 else None
    pipeline = DataPipeline.prepare_data(
        csv_path,
        input_features,
        output_features,
        problem_type=problem_type,
        batch_size=batch_size,
        val_split=validation_split,
        sequence_length=sequence_length,
    )
    if descriptor.input_shape[-1] != pipeline.input_shape[-1]:
        raise ValueError(
            f"Selected input feature count ({pipeline.input_shape[-1]}) does not match "
            f"the model input size ({descriptor.input_shape[-1]}). Retraining does not alter architecture."
        )
    if descriptor.output_shape[-1] != pipeline.output_shape[-1]:
        raise ValueError(
            f"Selected output size ({pipeline.output_shape[-1]}) does not match "
            f"the model output size ({descriptor.output_shape[-1]}). Retraining does not alter architecture."
        )

    problem_type = problem_type.lower()
    criterion = nn.CrossEntropyLoss() if problem_type == "classification" else nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    target_scaler = pipeline.scalers.get("y")
    baseline = _evaluate_dataset(
        model, pipeline.val_loader, criterion, device, problem_type, target_scaler
    )
    history: List[Dict[str, Any]] = []
    started = time.perf_counter()

    for epoch_index in range(epochs):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        gradient_norm_sum = 0.0
        batch_count = 0
        epoch_started = time.perf_counter()
        for features, targets in pipeline.train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions, targets = _align_training_tensors(
                _primary_tensor(model(features)), targets, problem_type
            )
            loss = criterion(predictions, targets)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered at epoch {epoch_index + 1}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            count = int(targets.shape[0])
            loss_sum += float(loss.item()) * count
            sample_count += count
            gradient_norm_sum += float(gradient_norm)
            batch_count += 1

        validation = _evaluate_dataset(
            model, pipeline.val_loader, criterion, device, problem_type, target_scaler
        )
        history.append({
            "epoch": epoch_index + 1,
            "train_loss": round(loss_sum / max(sample_count, 1), 8),
            "validation_loss": validation["loss"],
            "gradient_norm": round(gradient_norm_sum / max(batch_count, 1), 8),
            "learning_rate": learning_rate,
            "duration_ms": round((time.perf_counter() - epoch_started) * 1000.0, 4),
        })

    final_validation = _evaluate_dataset(
        model, pipeline.val_loader, criterion, device, problem_type, target_scaler
    )
    elapsed = time.perf_counter() - started
    baseline_loss = baseline.get("loss")
    final_loss = final_validation.get("loss")
    improvement_pct = (
        ((baseline_loss - final_loss) / baseline_loss) * 100.0
        if baseline_loss and final_loss is not None else None
    )
    training = {
        "performed": True,
        "device": str(device),
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "duration_seconds": round(elapsed, 4),
        "history": history,
    }
    validation = {
        "problem_type": problem_type,
        "split": validation_split,
        "metric_scale": (
            "original_target_units" if problem_type == "regression" else "class_indices"
        ),
        "loss_scale": (
            "standardized_targets" if problem_type == "regression" else "cross_entropy"
        ),
        "baseline": baseline,
        "final": final_validation,
        "loss_improvement_percent": round(improvement_pct, 6) if improvement_pct is not None else None,
    }
    return training, validation







def generateModelDiagnostics(
    descriptor: ArchitectureDescriptor,
    weights_path: str = "",
    export_name: str = "model_diagnostics.onnx",
    report_name: str = "model_diagnostics_report.json",
    mode: str = "dummy",
    csv_path: str = "",
    input_features: Optional[List[str]] = None,
    output_features: Optional[List[str]] = None,
    epochs: int = 10,
    problem_type: str = "regression",
    batch_size: int = 32,
    validation_split: float = 0.2,
    learning_rate: float = 1e-3,
) -> Dict[str, Any]:
    mode = str(mode or "dummy").strip().lower()
    if mode not in ("dummy", "retrain"):
        raise ValueError("mode must be either 'dummy' or 'retrain'")
    problem_type = str(problem_type or "regression").strip().lower()
    if problem_type not in ("regression", "classification"):
        raise ValueError("problem_type must be 'regression' or 'classification'")
    input_features = (
        list(input_features) if isinstance(input_features, (list, tuple))
        else ([str(input_features)] if input_features else [])
    )
    output_features = (
        list(output_features) if isinstance(output_features, (list, tuple))
        else ([str(output_features)] if output_features else [])
    )
    epochs = int(epochs)
    batch_size = int(batch_size)
    validation_split = float(validation_split)
    learning_rate = float(learning_rate)
    descriptor_copy = copy.deepcopy(descriptor)
    descriptor_copy.normalize_inplace()
    descriptor_copy.validate()

    model = DescriptorModelBuilder.build(descriptor_copy)
    model.eval()

    validation_summary = {
        "validated": True,
        "model_name": descriptor_copy.model_name,
        "input_shape": list(descriptor_copy.input_shape),
        "output_shape": list(descriptor_copy.output_shape),
    }

    shape_map: Dict[str, List[int]] = {}
    shape_error: Optional[str] = None
    try:
        shape_map = descriptor_copy._propagate_shapes(mutate=False)
    except Exception as exc:
        shape_error = str(exc)

    degree_maps = _node_degree_maps(descriptor_copy)
    node_type_counts = Counter(node.type for node in descriptor_copy.nodes)
    execution_counts: Counter = Counter()
    node_summaries: List[Dict[str, Any]] = []
    node_modules = getattr(model, "node_modules", {})

    for node in descriptor_copy.nodes:
        node_module = node_modules[node.id] if node.id in node_modules else None
        execution_op = _module_kind(node_module) if node_module else "virtual"
        if node_module:
            execution_counts[_module_kind(node_module)] += 1
        counts = _count_parameters(node_module) if node_module else {"total": 0, "trainable": 0, "buffers": 0}
        node_summaries.append(
            {
                "id": node.id,
                "type": node.type,
                "module_kind": execution_op,
                "params": node.params,
                "input_degree": degree_maps["incoming"].get(node.id, 0),
                "output_degree": degree_maps["outgoing"].get(node.id, 0),
                "parameter_count": int(counts["total"]),
                "trainable_parameter_count": int(counts["trainable"]),
                "buffer_count": int(counts["buffers"]),
                "shape": shape_map.get(node.id),
            }
        )

    model_parameter_summary = _count_parameters(model)
    parameter_details = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype).replace("torch.", ""),
            "numel": int(parameter.numel()),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in model.named_parameters()
    ]

    buffer_details = [
        {
            "name": name,
            "shape": list(buffer.shape),
            "dtype": str(buffer.dtype).replace("torch.", ""),
            "numel": int(buffer.numel()),
        }
        for name, buffer in model.named_buffers()
    ]

    dummy_shape = dummyShapeFromDescriptor(descriptor_copy.input_shape)
    dummy_input = torch.randn(*dummy_shape, dtype=torch.float32)

    activation_profile: Dict[str, Any] = {}
    hook_handles = []

    def _capture_activation(node_id: str):
        def _hook(module, inputs, output):
            activation_profile[node_id] = {
                "module_type": _module_kind(module),
                "output": _summarize_value(output),
            }

        return _hook

    for node_id, node_module in node_modules.items():
        hook_handles.append(node_module.register_forward_hook(_capture_activation(node_id)))

    weights_report = _load_weights_if_available(model, weights_path)

    forward_profile: Dict[str, Any] = {
        "dummy_input_shape": dummy_shape,
        "succeeded": False,
    }
    inference_output: Any = None
    forward_time_ms: Optional[float] = None
    try:
        with torch.no_grad():
            start_time = time.perf_counter()
            inference_output = model(dummy_input)
            forward_time_ms = (time.perf_counter() - start_time) * 1000.0
        forward_profile.update(
            {
                "succeeded": True,
                "latency_ms": round(forward_time_ms, 4) if forward_time_ms is not None else None,
                "output": _summarize_value(inference_output),
            }
        )
    except Exception as exc:
        forward_profile["error"] = str(exc)
    finally:
        for handle in hook_handles:
            handle.remove()

    training_report: Dict[str, Any] = {"performed": False}
    validation_metrics: Dict[str, Any] = {
        "problem_type": problem_type,
        "available": False,
        "reason": "Dummy mode performs structural and inference diagnostics only.",
    }
    if mode == "retrain":
        training_report, validation_metrics = _retrain_and_validate(
            model,
            descriptor_copy,
            csv_path=csv_path,
            input_features=input_features,
            output_features=output_features,
            epochs=epochs,
            problem_type=problem_type,
            batch_size=batch_size,
            validation_split=validation_split,
            learning_rate=learning_rate,
        )
        validation_metrics["available"] = True

    retrained_weights_path: Optional[str] = None
    if mode == "retrain":
        retrained_weights_path = _project_path("diagnostics_retrained_weights.pth")
        torch.save(
            {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "model_config": descriptor_copy.to_dict(),
                "training": training_report,
                "validation": validation_metrics,
            },
            retrained_weights_path,
        )

    export_path = _project_path(export_name)
    onnx_report: Dict[str, Any] = {
        "requested": True,
        "path": export_path,
        "exported": False,
    }
    try:
        descriptorToOnnx(model, descriptor_copy, export_path)
        onnx_report["exported"] = True
        onnx_analysis = analyseOnnx(export_path)
        onnx_report["valid"] = not (
            isinstance(onnx_analysis, dict) and onnx_analysis.get("error")
        )
        if os.path.exists(export_path):
            onnx_report["size_bytes"] = os.path.getsize(export_path)
    except Exception as exc:
        onnx_report["error"] = str(exc)

    graph_stats = {
        "model_name": descriptor_copy.model_name,
        "node_count": len(descriptor_copy.nodes),
        "edge_count": len(descriptor_copy.edges),
        "topological_order": descriptor_copy._topological_node_order(),
        "node_type_counts": dict(node_type_counts),
        "execution_counts": dict(execution_counts),
        "input_nodes": [edge.target for edge in descriptor_copy.edges if edge.source == "input"],
        "output_sources": [edge.source for edge in descriptor_copy.edges if edge.target == "output"],
        "source_nodes": [node.id for node in descriptor_copy.nodes if degree_maps["incoming"].get(node.id, 0) == 0],
        "sink_nodes": [node.id for node in descriptor_copy.nodes if degree_maps["outgoing"].get(node.id, 0) == 0],
        "max_in_degree": max(degree_maps["incoming"].values() or [0]),
        "max_out_degree": max(degree_maps["outgoing"].values() or [0]),
        "avg_in_degree": round(sum(degree_maps["incoming"].values()) / max(len(descriptor_copy.nodes), 1), 4),
        "avg_out_degree": round(sum(degree_maps["outgoing"].values()) / max(len(descriptor_copy.nodes), 1), 4),
        "density": round(len(descriptor_copy.edges) / max(len(descriptor_copy.nodes) * max(len(descriptor_copy.nodes) - 1, 1), 1), 6),
    }

    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    performance_summary = {
        "parameter_bytes": int(parameter_bytes),
        "buffer_bytes": int(buffer_bytes),
        "parameter_megabytes": round(parameter_bytes / (1024 * 1024), 6),
        "buffer_megabytes": round(buffer_bytes / (1024 * 1024), 6),
        "forward_latency_ms": forward_profile.get("latency_ms"),
    }


    reachability = _graph_reachability(descriptor_copy)
    warnings_block = _detect_warnings(
        descriptor=descriptor_copy,
        shape_map=shape_map,
        node_summaries=node_summaries,
        parameter_details=parameter_details,
        forward_profile=forward_profile,
        activation_profile=activation_profile,
        weights_report=weights_report,
        onnx_report=onnx_report,
        reachability=reachability,
        degree_maps=degree_maps,
        node_modules=node_modules,
    )


    performance_block = _build_performance_insights(
        model=model,
        node_modules=node_modules,
        dummy_input=dummy_input,
        activation_profile=activation_profile,
        node_summaries=node_summaries,
        parameter_details=parameter_details,
        forward_profile=forward_profile,
    )


    for n in node_summaries:
        n["reachability"] = reachability["classification"].get(n["id"], "orphan")

    result = {
        "status": "success",
        "schema_version": "2.0",
        "request": {
            "mode": mode,
            "input_features": input_features,
            "output_features": output_features,
            "epochs": epochs if mode == "retrain" else 0,
            "problem_type": problem_type,
        },
        "overview": {
            **validation_summary,
            "diagnostic_mode": mode,
            "weights_loaded": weights_report.get("loaded", False),
            "forward_pass_succeeded": forward_profile.get("succeeded", False),
            "onnx_exported": onnx_report.get("exported", False),
        },
        "architecture": {
            "node_count": graph_stats["node_count"],
            "edge_count": graph_stats["edge_count"],
            "node_type_counts": graph_stats["node_type_counts"],
            "graph_density": graph_stats["density"],
            "shape_inference_available": bool(shape_map),
            "shape_inference_error": shape_error,
            "total_parameters": model_parameter_summary["total"],
            "trainable_parameters": model_parameter_summary["trainable"],
            "buffer_parameters": model_parameter_summary["buffers"],
            "parameter_bytes": performance_summary["parameter_bytes"],
            "parameter_megabytes": performance_summary["parameter_megabytes"],
            "parameter_density": performance_block["parameter_density"],
        },
        "inference": {
            "dummy_input_shape": forward_profile["dummy_input_shape"],
            "succeeded": forward_profile.get("succeeded", False),
            "latency_ms": forward_profile.get("latency_ms"),
            "output_summary": forward_profile.get("output"),
            "error": forward_profile.get("error"),
            "compute": performance_block["flops"],
            "memory": performance_block["memory"],
        },
        "training": training_report,
        "validation": validation_metrics,
        "warnings": warnings_block,
        "weights": {
            "requested": weights_report.get("requested", False),
            "loaded": weights_report.get("loaded", False),
            "matched_key_count": weights_report.get("matched_key_count", 0),
            "unmatched_source_count": weights_report.get("unmatched_source_count", 0),
            "unmatched_target_count": weights_report.get("unmatched_target_count", 0),
            "error": weights_report.get("error"),
        },
        "onnx": {
            "exported": onnx_report.get("exported", False),
            "valid": onnx_report.get("valid"),
            "size_bytes": onnx_report.get("size_bytes"),
            "error": onnx_report.get("error"),
        },
    }

    report_path = _project_path(report_name)
    result["report_path"] = report_path
    result["artifacts"] = {
        "report": report_path,
        "onnx": export_path if onnx_report.get("exported") else None,
        "retrained_weights": retrained_weights_path,
    }

    try:
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(make_json_serializable(result), report_file, indent=2, default=str)
    except Exception as exc:
        result["report_write_error"] = str(exc)

    return make_json_serializable(result)
