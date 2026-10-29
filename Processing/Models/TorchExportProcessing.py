from __future__ import annotations

import operator
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.fx import Node as FxNode

from Core.ArchitectureDescriptor import ArchitectureDescriptor, Edge, Node, TensorContract
from Processing.Models.TorchPackageProcessing import (
    TorchPackageImportError,
    _fx_dependencies,
    _primary_output,
    _verify_descriptor,
)
from Utils.Other import make_json_serializable


class NotTorchExportArtifact(ValueError):
    pass


class TorchExportImportError(TorchPackageImportError):
    pass


def load_exported_program(path: str):
    try:
        return torch.export.load(path)
    except Exception as exc:
        raise NotTorchExportArtifact(str(exc)) from exc


def _kind_name(spec: Any) -> str:
    return str(getattr(getattr(spec, "kind", None), "name", getattr(spec, "kind", ""))).upper()


def _argument_name(spec: Any) -> str:
    return str(getattr(getattr(spec, "arg", None), "name", ""))


def _parameter_targets(exported_program: Any) -> Dict[str, str]:
    result = {}
    for spec in exported_program.graph_signature.input_specs:
        if _kind_name(spec) in ("PARAMETER", "BUFFER"):
            result[_argument_name(spec)] = str(getattr(spec, "target", ""))
    return result


def _user_input_names(exported_program: Any) -> List[str]:
    return [
        _argument_name(spec)
        for spec in exported_program.graph_signature.input_specs
        if _kind_name(spec) == "USER_INPUT"
    ]


def _meta_tensor(node: FxNode) -> Optional[torch.Tensor]:
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


def _meta_shape(node: FxNode, dynamic_batch: bool = True) -> Optional[List[int]]:
    value = _meta_tensor(node)
    if value is None:
        return None
    shape = []
    for item in value.shape:
        try:
            shape.append(int(item))
        except (TypeError, ValueError, RuntimeError):
            shape.append(-1)
    if dynamic_batch and shape:
        shape[0] = -1
    return shape


def _meta_dtype(node: FxNode) -> str:
    value = _meta_tensor(node)
    return str(value.dtype).replace("torch.", "") if value is not None else "float32"


def _example_for_node(node: FxNode) -> torch.Tensor:
    value = _meta_tensor(node)
    if value is None:
        raise TorchExportImportError("The exported user input has no tensor metadata.")
    shape = []
    for index, item in enumerate(value.shape):
        try:
            shape.append(int(item))
        except (TypeError, ValueError, RuntimeError):
            shape.append(2 if index == 0 else 1)
    dtype = value.dtype
    if dtype.is_floating_point or dtype.is_complex:
        return torch.randn(*shape, dtype=dtype)
    if dtype == torch.bool:
        return torch.zeros(*shape, dtype=dtype)
    return torch.zeros(*shape, dtype=dtype)


def _target_name(target: Any) -> str:
    return str(target)


def _module_info(node: FxNode) -> Tuple[str, str]:
    stack = node.meta.get("nn_module_stack") or {}
    selected_fqn = ""
    selected_type = ""
    for value in stack.values():
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            fqn, module_type = str(value[0]), str(value[1])
            if fqn:
                selected_fqn, selected_type = fqn, module_type
    return selected_fqn, selected_type


def _state_target(argument: Any, parameter_targets: Dict[str, str]) -> str:
    if isinstance(argument, FxNode):
        return parameter_targets.get(argument.name, str(argument.target))
    return ""


def _state_tensor(state: Dict[str, torch.Tensor], argument: Any, targets: Dict[str, str]) -> Optional[torch.Tensor]:
    return state.get(_state_target(argument, targets))


def _constant(node: FxNode, position: int, keyword: str, default: Any) -> Any:
    value = _argument(node, position, keyword, default)
    return default if isinstance(value, FxNode) else value


def _argument(node: FxNode, position: int, keyword: str, default: Any = None) -> Any:
    return node.kwargs.get(keyword, node.args[position] if len(node.args) > position else default)


def _allocate_id(preferred: str, used: set) -> str:
    normalized = preferred.replace(".", "_").replace("/", "_") or "operation"
    if normalized in ("input", "output"):
        normalized = f"export_{normalized}"
    candidate = normalized
    index = 2
    while candidate in used:
        candidate = f"{normalized}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def _operation_spec(
    node: FxNode,
    state: Dict[str, torch.Tensor],
    parameter_targets: Dict[str, str],
) -> Optional[Tuple[str, Dict[str, Any], str]]:
    target = _target_name(node.target)
    fqn, module_type = _module_info(node)

    if target in ("aten.lstm.input", "aten.gru.input"):
        weights = node.args[2] if len(node.args) > 2 else []
        input_weight = _state_tensor(state, weights[0], parameter_targets) if weights else None
        hidden_weight = _state_tensor(state, weights[1], parameter_targets) if len(weights) > 1 else None
        if input_weight is None or hidden_weight is None:
            return None
        node_type = "LSTM" if "lstm" in target else "GRU"
        return node_type, {
            "input_size": int(input_weight.shape[1]),
            "hidden_size": int(hidden_weight.shape[1]),
            "bias": bool(_constant(node, 3, "has_biases", True)),
            "num_layers": int(_constant(node, 4, "num_layers", 1)),
            "dropout": float(_constant(node, 5, "dropout", 0.0)),
            "bidirectional": bool(_constant(node, 7, "bidirectional", False)),
            "batch_first": bool(_constant(node, 8, "batch_first", False)),
        }, fqn or node.name

    if target == "aten.linear.default":
        weight_arg = _argument(node, 1, "weight")
        bias_arg = _argument(node, 2, "bias")
        weight = _state_tensor(state, weight_arg, parameter_targets)
        if weight is None:
            return None
        bias_target = _state_target(bias_arg, parameter_targets)
        return "Linear", {
            "in_features": int(weight.shape[1]),
            "out_features": int(weight.shape[0]),
            "bias": bool(bias_target and bias_target in state),
        }, fqn or node.name

    if target in ("aten.conv1d.default", "aten.conv2d.default"):
        weight_arg = _argument(node, 1, "weight")
        bias_arg = _argument(node, 2, "bias")
        weight = _state_tensor(state, weight_arg, parameter_targets)
        if weight is None:
            return None
        rank = 1 if "conv1d" in target else 2
        groups = int(_constant(node, 6, "groups", 1))
        return ("Conv1d" if rank == 1 else "Conv2d"), {
            "in_channels": int(weight.shape[1]) * groups,
            "out_channels": int(weight.shape[0]),
            "kernel_size": [int(item) for item in weight.shape[2:]],
            "stride": make_json_serializable(_constant(node, 3, "stride", [1] * rank)),
            "padding": make_json_serializable(_constant(node, 4, "padding", [0] * rank)),
            "dilation": make_json_serializable(_constant(node, 5, "dilation", [1] * rank)),
            "groups": groups,
            "bias": _state_tensor(state, bias_arg, parameter_targets) is not None,
        }, fqn or node.name

    simple = {
        "aten.relu.default": ("ReLU", {}),
        "aten.sigmoid.default": ("Sigmoid", {}),
        "aten.tanh.default": ("Tanh", {}),
        "aten.silu.default": ("SiLU", {"inplace": False}),
        "aten.add.Tensor": ("Add", {}),
    }
    if target in simple:
        node_type, params = simple[target]
        return node_type, params, fqn or node.name
    if target == "aten.gelu.default":
        return "GELU", {"approximate": str(_constant(node, 1, "approximate", "none"))}, fqn or node.name
    if target in ("aten.softmax.int", "aten.log_softmax.int"):
        node_type = "LogSoftmax" if "log_softmax" in target else "Softmax"
        return node_type, {"dim": int(_constant(node, 1, "dim", -1))}, fqn or node.name
    if target == "aten.dropout.default":
        return "Dropout", {
            "p": float(_constant(node, 1, "p", 0.5)),
            "inplace": False,
        }, fqn or node.name

    if target in ("aten.cat.default", "aten.cat.out"):
        return "Concat", {"dim": int(_constant(node, 1, "dim", 0))}, fqn or node.name
    if target == "aten.layer_norm.default":
        weight = _state_tensor(state, _argument(node, 2, "weight"), parameter_targets)
        bias = _state_tensor(state, _argument(node, 3, "bias"), parameter_targets)
        return "LayerNorm", {
            "normalized_shape": make_json_serializable(_constant(node, 1, "normalized_shape", [])),
            "eps": float(_constant(node, 4, "eps", 1e-5)),
            "elementwise_affine": weight is not None,
            "bias": bias is not None,
        }, fqn or node.name
    if target == "aten.embedding.default":
        weight = _state_tensor(state, _argument(node, 0, "weight"), parameter_targets)
        if weight is None:
            return None
        padding_idx = int(_constant(node, 2, "padding_idx", -1))
        return "Embedding", {
            "num_embeddings": int(weight.shape[0]),
            "embedding_dim": int(weight.shape[1]),
            "padding_idx": None if padding_idx < 0 else padding_idx,
            "scale_grad_by_freq": bool(_constant(node, 3, "scale_grad_by_freq", False)),
            "sparse": bool(_constant(node, 4, "sparse", False)),
        }, fqn or node.name
    if target in ("aten.conv_transpose1d.default", "aten.conv_transpose2d.input"):
        weight = _state_tensor(state, _argument(node, 1, "weight"), parameter_targets)
        if weight is None:
            return None
        rank = 1 if "1d" in target else 2
        groups = int(_constant(node, 6, "groups", 1))
        return ("ConvTranspose1d" if rank == 1 else "ConvTranspose2d"), {
            "in_channels": int(weight.shape[0]),
            "out_channels": int(weight.shape[1]) * groups,
            "kernel_size": [int(item) for item in weight.shape[2:]],
            "stride": make_json_serializable(_constant(node, 3, "stride", [1] * rank)),
            "padding": make_json_serializable(_constant(node, 4, "padding", [0] * rank)),
            "output_padding": make_json_serializable(_constant(node, 5, "output_padding", [0] * rank)),
            "groups": groups,
            "dilation": make_json_serializable(_constant(node, 7, "dilation", [1] * rank)),
            "bias": _state_tensor(state, _argument(node, 2, "bias"), parameter_targets) is not None,
        }, fqn or node.name
    if target in (
        "aten.max_pool1d.default", "aten.avg_pool1d.default",
        "aten.max_pool2d.default", "aten.avg_pool2d.default",
    ):
        is_max = "max_pool" in target
        rank = 1 if "1d" in target else 2
        params = {
            "kernel_size": make_json_serializable(_constant(node, 1, "kernel_size", [1] * rank)),
            "stride": make_json_serializable(_constant(node, 2, "stride", [])) or None,
            "padding": make_json_serializable(_constant(node, 3, "padding", [0] * rank)),
            "ceil_mode": bool(_constant(node, 5 if is_max else 4, "ceil_mode", False)),
        }
        if is_max:
            params["dilation"] = make_json_serializable(_constant(node, 4, "dilation", [1] * rank))
        else:
            params["count_include_pad"] = bool(_constant(node, 5, "count_include_pad", True))
            if rank == 2:
                params["divisor_override"] = _constant(node, 6, "divisor_override", None)
        prefix = "MaxPool" if is_max else "AvgPool"
        return f"{prefix}{rank}d", params, fqn or node.name
    if target == "aten._native_batch_norm_legit_no_training.default":
        running_mean = _state_tensor(state, _argument(node, 3, "running_mean"), parameter_targets)
        input_meta = _meta_tensor(_argument(node, 0, "input"))
        if running_mean is None:
            return None
        rank = input_meta.dim() if input_meta is not None else 0
        node_type = "BatchNorm1d" if rank in (2, 3) else "BatchNorm2d" if rank == 4 else ""
        if not node_type:
            return None
        return node_type, {
            "num_features": int(running_mean.shape[0]),
            "eps": float(_constant(node, 6, "eps", 1e-5)),
            "momentum": float(_constant(node, 5, "momentum", 0.1)),
            "affine": _state_tensor(state, _argument(node, 1, "weight"), parameter_targets) is not None,
            "track_running_stats": True,
        }, fqn or node.name
    if target in ("aten.view.default", "aten.reshape.default"):
        output_shape = _meta_shape(node, dynamic_batch=False)
        input_deps = _fx_dependencies(node.args[0] if node.args else None)
        input_shape = _meta_shape(input_deps[0], dynamic_batch=False) if input_deps else None
        if output_shape and input_shape and len(output_shape) == 2 and len(input_shape) > 2:
            return "Flatten", {"start_dim": 1, "end_dim": -1}, fqn or node.name
        if output_shape:
            return "Reshape", {"target_shape": output_shape[1:]}, fqn or node.name
    if target == "aten.mean.dim":
        return "ReduceMean", {
            "dim": make_json_serializable(_constant(node, 1, "dim", None)),
            "keepdim": bool(_constant(node, 2, "keepdim", False)),
        }, fqn or node.name
    if target == "aten.unsqueeze.default":
        return "Unsqueeze", {"dim": int(_constant(node, 1, "dim", 0))}, fqn or node.name
    if target == "aten.squeeze.dim":
        return "Squeeze", {"dim": int(_constant(node, 1, "dim", 0))}, fqn or node.name
    if target == "aten.transpose.int":
        return "Transpose", {
            "dim0": int(_constant(node, 1, "dim0", 0)),
            "dim1": int(_constant(node, 2, "dim1", 1)),
        }, fqn or node.name
    if target == "aten.permute.default":
        return "Permute", {"dims": make_json_serializable(_constant(node, 1, "dims", []))}, fqn or node.name
    return None


def _is_full_slice(node: FxNode) -> bool:
    if _target_name(node.target) != "aten.slice.Tensor":
        return False
    start = _constant(node, 2, "start", 0)
    end = _constant(node, 3, "end", 9223372036854775807)
    step = _constant(node, 4, "step", 1)
    return int(start) == 0 and int(end) >= 9223372036854775807 and int(step) == 1


def descriptor_from_exported_program(exported_program: Any):
    graph_module = exported_program.graph_module
    graph = graph_module.graph
    state = exported_program.state_dict
    parameter_targets = _parameter_targets(exported_program)
    user_names = _user_input_names(exported_program)
    user_inputs = [node for node in graph.nodes if node.op == "placeholder" and node.name in user_names]
    if len(user_inputs) != 1:
        raise TorchExportImportError(
            f"MYLO descriptors currently support one tensor input; this export has {len(user_inputs)} user inputs."
        )

    input_node = user_inputs[0]
    example_input = _example_for_node(input_node)
    input_shape = _meta_shape(input_node)
    if not input_shape:
        raise TorchExportImportError("Could not determine the exported model input shape.")

    descriptor_nodes: List[Node] = []
    descriptor_edges: List[Edge] = []
    tensor_contracts: Dict[str, TensorContract] = {
        "input": TensorContract(input_shape, _meta_dtype(input_node))
    }
    resolved: Dict[FxNode, str] = {input_node: "input"}
    aliases: Dict[FxNode, FxNode] = {}
    used_ids = {"input", "output"}
    unsupported: List[str] = []

    def resolve(source: FxNode) -> Optional[str]:
        seen = set()
        while source in aliases and source not in seen:
            seen.add(source)
            source = aliases[source]
        return resolved.get(source)

    for fx_node in graph.nodes:
        if fx_node.op in ("placeholder", "get_attr", "output") or not fx_node.users:
            continue

        target = _target_name(fx_node.target)
        dependencies = _fx_dependencies(fx_node.args)
        data_dependencies = [dependency for dependency in dependencies if resolve(dependency)]

        if fx_node.op == "call_function" and fx_node.target is operator.getitem:
            index = fx_node.args[1] if len(fx_node.args) > 1 else None
            if len(data_dependencies) == 1 and index == 0:
                aliases[fx_node] = data_dependencies[0]
                continue

        if _is_full_slice(fx_node) and len(data_dependencies) == 1:
            aliases[fx_node] = data_dependencies[0]
            continue

        if target in (
            "aten.alias.default",
            "aten.clone.default",
            "aten.contiguous.default",
            "aten.detach.default",
        ) and len(data_dependencies) == 1:
            aliases[fx_node] = data_dependencies[0]
            continue

        if target == "aten.select.int" and len(data_dependencies) == 1:
            dim = int(_constant(fx_node, 1, "dim", 0))
            index = int(_constant(fx_node, 2, "index", 0))
            source_id = resolve(data_dependencies[0])
            source_node = next((node for node in descriptor_nodes if node.id == source_id), None)
            if index == -1 and source_node and source_node.type in ("LSTM", "GRU"):
                source_node.execution["output_mode"] = "last_timestep"
                aliases[fx_node] = data_dependencies[0]
                continue

        spec = _operation_spec(fx_node, state, parameter_targets) if fx_node.op == "call_function" else None
        if spec is None:
            if not data_dependencies and target.startswith("aten.zeros"):
                continue
            unsupported.append(f"{fx_node.name}={target}")
            continue

        node_type, params, preferred_id = spec
        descriptor_id = _allocate_id(preferred_id, used_ids)
        execution = {
            "source": "torch.export",
            "aten_target": target,
            "editable": True,
        }
        module_fqn, _ = _module_info(fx_node)
        if module_fqn:
            execution.update({"fx_op": "call_module", "fx_target": module_fqn})
        if node_type in ("LSTM", "GRU"):
            execution["output_mode"] = "sequence"
        descriptor_nodes.append(Node(
            id=descriptor_id,
            type=node_type,
            params=make_json_serializable(params),
            execution=execution,
        ))
        resolved[fx_node] = descriptor_id

        source_ids = []
        for dependency in data_dependencies:
            source_id = resolve(dependency)
            if source_id and source_id not in source_ids:
                source_ids.append(source_id)
        if len(source_ids) > 1 and node_type not in ("Add", "Concat"):
            unsupported.append(f"{fx_node.name}={target} (multiple tensor inputs)")
            continue
        descriptor_edges.extend(Edge(source=source_id, target=descriptor_id) for source_id in source_ids)
        shape = _meta_shape(fx_node)
        if shape:
            tensor_contracts[descriptor_id] = TensorContract(shape, _meta_dtype(fx_node))

    if unsupported:
        preview = ", ".join(unsupported[:6])
        suffix = "" if len(unsupported) <= 6 else f" (+{len(unsupported) - 6} more)"
        raise TorchExportImportError(
            f"The torch.export graph contains operations MYLO cannot faithfully edit yet: {preview}{suffix}."
        )

    output_fx_nodes = [node for node in graph.nodes if node.op == "output"]
    output_dependencies = _fx_dependencies(output_fx_nodes[0].args if output_fx_nodes else ())
    output_ids = []
    output_tensor_node = None
    for dependency in output_dependencies:
        source_id = resolve(dependency)
        if source_id and source_id not in output_ids:
            output_ids.append(source_id)
            output_tensor_node = dependency
    if len(output_ids) != 1:
        raise TorchExportImportError(
            f"MYLO descriptors currently require one tensor output; this export exposes {len(output_ids)} outputs."
        )
    descriptor_edges.append(Edge(output_ids[0], "output"))

    exported_module = exported_program.module()
    with torch.no_grad():
        output = _primary_output(exported_module(example_input))
    if output is None:
        raise TorchExportImportError("The exported program does not return a single primary tensor.")
    output_shape = [int(item) for item in output.shape]
    if output_shape:
        output_shape[0] = -1
    tensor_contracts["output"] = TensorContract(output_shape, str(output.dtype).replace("torch.", ""))

    descriptor = ArchitectureDescriptor(
        model_name="ExportedProgram",
        input_shape=input_shape,
        output_shape=output_shape,
        nodes=descriptor_nodes,
        edges=descriptor_edges,
        tensor_contracts=tensor_contracts,
        propagation_rules="torch_export_aten_import",
    )
    descriptor.normalize_inplace(strict=True)
    descriptor.validate(strict=True)
    rebuilt, verification = _verify_descriptor(exported_module, descriptor, example_input)
    report = {
        "format": "torch.export",
        "editable": True,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "exported_node_count": len(list(graph.nodes)),
        "descriptor_node_count": len(descriptor.nodes),
        "verification": verification,
    }
    return descriptor, rebuilt, report


def import_torch_export(path: str):
    exported_program = load_exported_program(path)
    try:
        return descriptor_from_exported_program(exported_program)
    except TorchExportImportError:
        raise
    except Exception as exc:
        raise TorchExportImportError(f"Could not convert the torch.export graph into a MYLO descriptor: {exc}") from exc


def extract_exported_state_dict(path: str) -> Optional[Dict[str, torch.Tensor]]:
    try:
        _, rebuilt, _ = import_torch_export(path)
        return rebuilt.state_dict()
    except Exception:
        return None
