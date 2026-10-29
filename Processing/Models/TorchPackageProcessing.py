from __future__ import annotations

import json
import io
import logging
import operator
import os
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.fx import GraphModule, Node as FxNode, symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from torch.package import PackageExporter, PackageImporter

from Core.ArchitectureDescriptor import ArchitectureDescriptor, Edge, Node, TensorContract
from Core.DescriptorModelBuilder import DescriptorModelBuilder
from Core.WeightCompatibilityEngine import WeightCompatibilityEngine
from Utils.Other import make_json_serializable


logger = logging.getLogger(__name__)

TORCH_PACKAGE_EXTENSIONS = {".ptpkg", ".torchpackage"}
MYLO_PACKAGE_NAMESPACE = "mylo"
MYLO_MODEL_RESOURCE = "model.pkl"
MYLO_MANIFEST_RESOURCE = "manifest.json"


class TorchPackageImportError(ValueError):
    pass


@dataclass
class TorchPackageImportResult:
    model: nn.Module
    descriptor: ArchitectureDescriptor
    rebuilt_model: nn.Module
    report: Dict[str, Any]


def is_torch_package_extension(filename: str) -> bool:
    return os.path.splitext(str(filename or ""))[1].lower() in TORCH_PACKAGE_EXTENSIONS


def is_torch_package_archive(path: str) -> bool:
    try:
        with open(path, "rb") as package_file:
            PackageImporter(io.BytesIO(package_file.read()))
        return True
    except Exception:
        return False


def _archive_relative_names(path: str) -> List[str]:
    with zipfile.ZipFile(path, "r") as archive:
        names = [name.replace("\\", "/").lstrip("/") for name in archive.namelist()]

    root = ""
    for name in names:
        marker = ".data/extern_modules"
        if marker in name:
            root = name[: name.index(marker)]
            break
    if not root and names:
        first_parts = [name.split("/", 1)[0] for name in names if "/" in name]
        if first_parts and len(set(first_parts)) == 1:
            root = first_parts[0] + "/"
    return [name[len(root):] if root and name.startswith(root) else name for name in names]


def _pickle_candidates(path: str, manifest: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []
    if manifest:
        location = manifest.get("model") or manifest.get("model_location")
        if isinstance(location, dict):
            package = str(location.get("package", MYLO_PACKAGE_NAMESPACE))
            resource = str(location.get("resource", MYLO_MODEL_RESOURCE))
            candidates.append((package, resource))

    candidates.extend([
        (MYLO_PACKAGE_NAMESPACE, MYLO_MODEL_RESOURCE),
        ("model", "model.pkl"),
        ("models", "model.pkl"),
    ])

    try:
        paths = _archive_relative_names(path)
    except (OSError, zipfile.BadZipFile):
        paths = []

    discovered = []
    for archive_path in paths:
        if archive_path.startswith(".data/") or not archive_path.lower().endswith((".pkl", ".pickle")):
            continue
        directory, resource = os.path.split(archive_path)
        if not directory:
            continue
        package = directory.replace("/", ".").replace("\\", ".")
        score = 0 if "model" in resource.lower() or "model" in package.lower() else 1
        discovered.append((score, package, resource))
    candidates.extend((package, resource) for _, package, resource in sorted(discovered))

    unique: List[Tuple[str, str]] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _try_load_manifest(importer: PackageImporter) -> Optional[Dict[str, Any]]:
    for package in (MYLO_PACKAGE_NAMESPACE, "model", "metadata"):
        try:
            payload = importer.load_text(package, MYLO_MANIFEST_RESOURCE)
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _extract_module(payload: Any) -> Optional[nn.Module]:
    if isinstance(payload, nn.Module):
        return payload
    if isinstance(payload, dict):
        for key in ("model", "module", "network", "net"):
            candidate = payload.get(key)
            if isinstance(candidate, nn.Module):
                return candidate
    return None


def load_packaged_module(path: str) -> Tuple[nn.Module, Dict[str, Any]]:
    try:
        with open(path, "rb") as package_file:
            importer = PackageImporter(io.BytesIO(package_file.read()))
    except Exception as exc:
        raise TorchPackageImportError(f"Not a valid torch.package archive: {exc}") from exc

    manifest = _try_load_manifest(importer)
    failures: List[str] = []
    for package, resource in _pickle_candidates(path, manifest):
        try:
            try:
                payload = importer.load_pickle(package, resource, map_location="cpu")
            except TypeError:
                payload = importer.load_pickle(package, resource)
            model = _extract_module(payload)
            if model is not None:
                model.to("cpu")
                model.eval()
                return model, {
                    "package": package,
                    "resource": resource,
                    "manifest": manifest or {},
                }
            failures.append(f"{package}/{resource} did not contain an nn.Module")
        except Exception as exc:
            failures.append(f"{package}/{resource}: {exc}")

    hint = " Save the model with PackageExporter.save_pickle('model', 'model.pkl', model)."
    detail = failures[0] if failures else "No pickle resources were found."
    raise TorchPackageImportError(f"No packaged nn.Module could be loaded. {detail}.{hint}")


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Size):
        return [int(item) for item in value]
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _module_spec(module: nn.Module) -> Optional[Tuple[str, Dict[str, Any]]]:
    if isinstance(module, nn.Linear):
        return "Linear", {"in_features": module.in_features, "out_features": module.out_features, "bias": module.bias is not None}
    if isinstance(module, nn.Conv1d):
        return "Conv1d", _conv_params(module)
    if isinstance(module, nn.Conv2d):
        return "Conv2d", _conv_params(module)
    if isinstance(module, nn.ConvTranspose1d):
        return "ConvTranspose1d", _conv_params(module, transpose=True)
    if isinstance(module, nn.ConvTranspose2d):
        return "ConvTranspose2d", _conv_params(module, transpose=True)
    if isinstance(module, nn.LSTM):
        return "LSTM", _recurrent_params(module)
    if isinstance(module, nn.GRU):
        return "GRU", _recurrent_params(module)
    if isinstance(module, nn.ReLU):
        return "ReLU", {"inplace": bool(module.inplace)}
    if isinstance(module, nn.GELU):
        return "GELU", {"approximate": module.approximate}
    if isinstance(module, nn.SiLU):
        return "SiLU", {"inplace": bool(module.inplace)}
    if isinstance(module, nn.Tanh):
        return "Tanh", {}
    if isinstance(module, nn.Sigmoid):
        return "Sigmoid", {}
    if isinstance(module, nn.Softmax):
        return "Softmax", {"dim": module.dim}
    if isinstance(module, nn.LogSoftmax):
        return "LogSoftmax", {"dim": module.dim}
    if isinstance(module, nn.Dropout):
        return "Dropout", {"p": module.p, "inplace": bool(module.inplace)}
    if isinstance(module, nn.Flatten):
        return "Flatten", {"start_dim": module.start_dim, "end_dim": module.end_dim}
    if isinstance(module, nn.BatchNorm1d):
        return "BatchNorm1d", _batch_norm_params(module)
    if isinstance(module, nn.BatchNorm2d):
        return "BatchNorm2d", _batch_norm_params(module)
    if isinstance(module, nn.LayerNorm):
        return "LayerNorm", {
            "normalized_shape": _json_value(module.normalized_shape),
            "eps": module.eps,
            "elementwise_affine": bool(module.elementwise_affine),
            "bias": module.bias is not None,
        }
    if isinstance(module, nn.Embedding):
        return "Embedding", {
            "num_embeddings": module.num_embeddings,
            "embedding_dim": module.embedding_dim,
            "padding_idx": module.padding_idx,
            "scale_grad_by_freq": bool(module.scale_grad_by_freq),
            "sparse": bool(module.sparse),
        }
    if isinstance(module, nn.MaxPool1d):
        return "MaxPool1d", _pool_params(module)
    if isinstance(module, nn.AvgPool1d):
        return "AvgPool1d", _pool_params(module)
    if isinstance(module, nn.MaxPool2d):
        return "MaxPool2d", _pool_params(module)
    if isinstance(module, nn.AvgPool2d):
        return "AvgPool2d", _pool_params(module)
    if isinstance(module, nn.Identity):
        return "Identity", {}
    if isinstance(module, nn.MultiheadAttention):
        return "MultiheadAttention", {
            "embed_dim": module.embed_dim,
            "num_heads": module.num_heads,
            "batch_first": bool(module.batch_first),
        }
    if isinstance(module, nn.TransformerEncoderLayer):
        return "TransformerEncoderLayer", {
            "d_model": module.self_attn.embed_dim,
            "nhead": module.self_attn.num_heads,
            "dim_feedforward": module.linear1.out_features,
            "dropout": module.dropout.p,
            "batch_first": bool(module.self_attn.batch_first),
            "norm_first": bool(module.norm_first),
        }
    return None


def _conv_params(module: nn.Module, transpose: bool = False) -> Dict[str, Any]:
    params = {
        "in_channels": module.in_channels,
        "out_channels": module.out_channels,
        "kernel_size": _json_value(module.kernel_size),
        "stride": _json_value(module.stride),
        "padding": _json_value(module.padding),
        "dilation": _json_value(module.dilation),
        "groups": module.groups,
        "bias": module.bias is not None,
        "padding_mode": module.padding_mode,
    }
    if transpose:
        params["output_padding"] = _json_value(module.output_padding)
    return params


def _recurrent_params(module: nn.Module) -> Dict[str, Any]:
    return {
        "input_size": module.input_size,
        "hidden_size": module.hidden_size,
        "num_layers": module.num_layers,
        "bias": bool(module.bias),
        "batch_first": bool(module.batch_first),
        "dropout": module.dropout,
        "bidirectional": bool(module.bidirectional),
    }


def _batch_norm_params(module: nn.Module) -> Dict[str, Any]:
    return {
        "num_features": module.num_features,
        "eps": module.eps,
        "momentum": module.momentum,
        "affine": bool(module.affine),
        "track_running_stats": bool(module.track_running_stats),
    }


def _pool_params(module: nn.Module) -> Dict[str, Any]:
    params = {
        "kernel_size": _json_value(module.kernel_size),
        "stride": _json_value(module.stride),
        "padding": _json_value(module.padding),
        "ceil_mode": bool(module.ceil_mode),
    }
    if hasattr(module, "dilation"):
        params["dilation"] = _json_value(module.dilation)
    if hasattr(module, "count_include_pad"):
        params["count_include_pad"] = bool(module.count_include_pad)
    if hasattr(module, "divisor_override"):
        params["divisor_override"] = module.divisor_override
    return params


def _shape_from_tensor(value: torch.Tensor) -> List[int]:
    shape = [int(item) for item in value.shape]
    if shape:
        shape[0] = -1
    return shape


def _example_from_attribute(model: nn.Module) -> Optional[torch.Tensor]:
    for name in ("example_input_array", "example_input", "example_inputs"):
        value = getattr(model, name, None)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], torch.Tensor):
            return value[0].detach().cpu()
    return None


def _shape_from_manifest(manifest: Dict[str, Any]) -> Optional[List[int]]:
    value = manifest.get("input_shape") or manifest.get("example_input_shape")
    if not isinstance(value, (list, tuple)) or not value:
        return None
    try:
        shape = [1 if int(dim) <= 0 else int(dim) for dim in value]
    except (TypeError, ValueError):
        return None
    shape[0] = 1
    return shape


def _infer_example_input(model: nn.Module, graph_module: GraphModule, manifest: Dict[str, Any]) -> Tuple[torch.Tensor, str]:
    attributed = _example_from_attribute(model)
    if attributed is not None:
        return attributed, "model_attribute"

    manifest_shape = _shape_from_manifest(manifest)
    if manifest_shape:
        dtype_name = str(manifest.get("input_dtype", "float32")).replace("torch.", "")
        dtype = getattr(torch, dtype_name, torch.float32)
        if dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
            return torch.zeros(*manifest_shape, dtype=dtype), "package_manifest"
        return torch.randn(*manifest_shape, dtype=dtype), "package_manifest"

    modules = dict(graph_module.named_modules())
    for fx_node in graph_module.graph.nodes:
        if fx_node.op != "call_module":
            continue
        module = modules.get(str(fx_node.target))
        if isinstance(module, nn.Linear):
            return torch.randn(1, module.in_features), "first_linear"
        if isinstance(module, nn.Conv1d):
            return torch.randn(1, module.in_channels, 16), "first_conv1d"
        if isinstance(module, nn.Conv2d):
            return torch.randn(1, module.in_channels, 16, 16), "first_conv2d"
        if isinstance(module, (nn.LSTM, nn.GRU)):
            shape = (1, 4, module.input_size) if module.batch_first else (4, 1, module.input_size)
            return torch.randn(*shape), "first_recurrent"
        if isinstance(module, nn.Embedding):
            return torch.zeros(1, 4, dtype=torch.long), "first_embedding"
        if isinstance(module, nn.LayerNorm):
            return torch.randn(1, *_json_value(module.normalized_shape)), "first_layer_norm"
        if isinstance(module, nn.MultiheadAttention):
            shape = (1, 4, module.embed_dim) if module.batch_first else (4, 1, module.embed_dim)
            return torch.randn(*shape), "first_attention"

    raise TorchPackageImportError(
        "MYLO could not infer an example input shape from the packaged model. "
        "Expose model.example_input_array or include input_shape in the package manifest."
    )


def _tensor_meta_shape(node: FxNode) -> Optional[List[int]]:
    metadata = node.meta.get("tensor_meta")
    shape = getattr(metadata, "shape", None)
    if shape is None:
        return None
    return [int(item) for item in shape]


def _tensor_meta_dtype(node: FxNode) -> str:
    metadata = node.meta.get("tensor_meta")
    dtype = getattr(metadata, "dtype", torch.float32)
    return str(dtype).replace("torch.", "")


def _fx_dependencies(value: Any) -> List[FxNode]:
    if isinstance(value, FxNode):
        return [value]
    if isinstance(value, (tuple, list)):
        result: List[FxNode] = []
        for item in value:
            result.extend(_fx_dependencies(item))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_fx_dependencies(item))
        return result
    return []


def _constant_arg(node: FxNode, position: int, keyword: str, default: Any) -> Any:
    if keyword in node.kwargs:
        value = node.kwargs[keyword]
    elif len(node.args) > position:
        value = node.args[position]
    else:
        value = default
    return value if not isinstance(value, FxNode) else default


def _functional_spec(node: FxNode) -> Optional[Tuple[str, Dict[str, Any]]]:
    target = node.target
    if target in (operator.add, torch.add):
        return "Add", {}
    if target is torch.cat:
        return "Concat", {"dim": int(_constant_arg(node, 1, "dim", 0))}
    if target in (torch.relu, functional.relu):
        return "ReLU", {"inplace": bool(_constant_arg(node, 1, "inplace", False))}
    if target in (torch.sigmoid, functional.sigmoid):
        return "Sigmoid", {}
    if target in (torch.tanh, functional.tanh):
        return "Tanh", {}
    if target is functional.gelu:
        return "GELU", {"approximate": str(_constant_arg(node, 1, "approximate", "none"))}
    if target is functional.silu:
        return "SiLU", {"inplace": bool(_constant_arg(node, 1, "inplace", False))}
    if target is functional.dropout:
        return "Dropout", {
            "p": float(_constant_arg(node, 1, "p", 0.5)),
            "inplace": bool(_constant_arg(node, 3, "inplace", False)),
        }
    if target is functional.softmax:
        return "Softmax", {"dim": int(_constant_arg(node, 1, "dim", -1))}
    if target is functional.log_softmax:
        return "LogSoftmax", {"dim": int(_constant_arg(node, 1, "dim", -1))}
    if target is torch.flatten:
        return "Flatten", {
            "start_dim": int(_constant_arg(node, 1, "start_dim", 0)),
            "end_dim": int(_constant_arg(node, 2, "end_dim", -1)),
        }
    if target is torch.squeeze:
        return "Squeeze", {"dim": _constant_arg(node, 1, "dim", None)}
    if target is torch.unsqueeze:
        return "Unsqueeze", {"dim": int(_constant_arg(node, 1, "dim", 0))}
    if target is torch.mean:
        return "ReduceMean", {
            "dim": _json_value(_constant_arg(node, 1, "dim", None)),
            "keepdim": bool(_constant_arg(node, 2, "keepdim", False)),
        }
    if target is torch.transpose:
        return "Transpose", {
            "dim0": int(_constant_arg(node, 1, "dim0", 0)),
            "dim1": int(_constant_arg(node, 2, "dim1", 1)),
        }
    return None


def _method_spec(node: FxNode) -> Optional[Tuple[str, Dict[str, Any]]]:
    method = str(node.target)
    if method in ("relu", "sigmoid", "tanh"):
        return {"relu": ("ReLU", {}), "sigmoid": ("Sigmoid", {}), "tanh": ("Tanh", {})}[method]
    if method == "flatten":
        return "Flatten", {
            "start_dim": int(_constant_arg(node, 1, "start_dim", 0)),
            "end_dim": int(_constant_arg(node, 2, "end_dim", -1)),
        }
    if method in ("view", "reshape"):
        output_shape = _tensor_meta_shape(node)
        if output_shape is None or len(output_shape) < 1:
            return None
        return "Reshape", {"target_shape": output_shape[1:]}
    if method == "mean":
        return "ReduceMean", {
            "dim": _json_value(_constant_arg(node, 1, "dim", None)),
            "keepdim": bool(_constant_arg(node, 2, "keepdim", False)),
        }
    if method == "squeeze":
        return "Squeeze", {"dim": _constant_arg(node, 1, "dim", None)}
    if method == "unsqueeze":
        return "Unsqueeze", {"dim": int(_constant_arg(node, 1, "dim", 0))}
    if method == "transpose":
        return "Transpose", {
            "dim0": int(_constant_arg(node, 1, "dim0", 0)),
            "dim1": int(_constant_arg(node, 2, "dim1", 1)),
        }
    if method == "permute":
        dims = node.args[1:]
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = tuple(dims[0])
        return "Permute", {"dims": [int(item) for item in dims]}
    return None


def _is_last_timestep_index(value: Any) -> bool:
    if not isinstance(value, tuple) or len(value) < 2:
        return False
    return isinstance(value[0], slice) and value[1] == -1


def _primary_output(value: Any) -> Optional[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    if isinstance(value, dict):
        tensors = [item for item in value.values() if isinstance(item, torch.Tensor)]
        if len(tensors) == 1:
            return tensors[0]
    return None


def descriptor_from_fx(
    model: nn.Module,
    graph_module: GraphModule,
    example_input: torch.Tensor,
) -> Tuple[ArchitectureDescriptor, Dict[str, Any]]:
    placeholders = [node for node in graph_module.graph.nodes if node.op == "placeholder"]
    if len(placeholders) != 1:
        raise TorchPackageImportError(
            f"MYLO descriptors currently support one tensor input; the packaged forward graph has {len(placeholders)} inputs."
        )

    try:
        ShapeProp(graph_module).propagate(example_input)
    except Exception as exc:
        raise TorchPackageImportError(f"FX shape propagation failed for the inferred example input: {exc}") from exc

    modules = dict(graph_module.named_modules())
    descriptor_nodes: List[Node] = []
    descriptor_edges: List[Edge] = []
    fx_to_descriptor: Dict[FxNode, str] = {placeholders[0]: "input"}
    aliases: Dict[FxNode, FxNode] = {}
    unsupported: List[Dict[str, str]] = []
    used_descriptor_ids = {"input", "output"}

    def allocate_id(preferred: str) -> str:
        candidate = preferred
        if candidate in used_descriptor_ids:
            candidate = f"fx_{candidate}"
        suffix = 2
        base = candidate
        while candidate in used_descriptor_ids:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used_descriptor_ids.add(candidate)
        return candidate

    def resolve(source: FxNode) -> Optional[str]:
        visited = set()
        while source in aliases and source not in visited:
            visited.add(source)
            source = aliases[source]
        return fx_to_descriptor.get(source)

    for fx_node in graph_module.graph.nodes:
        if fx_node.op in ("placeholder", "output"):
            continue
        if not fx_node.users:
            continue

        if fx_node.op == "call_function" and fx_node.target is operator.getitem:
            dependencies = _fx_dependencies(fx_node.args[0] if fx_node.args else None)
            index = fx_node.args[1] if len(fx_node.args) > 1 else None
            if len(dependencies) == 1 and (index == 0 or _is_last_timestep_index(index)):
                aliases[fx_node] = dependencies[0]
                source_id = resolve(dependencies[0])
                if _is_last_timestep_index(index) and source_id:
                    source_node = next((item for item in descriptor_nodes if item.id == source_id), None)
                    if source_node and source_node.type in ("LSTM", "GRU"):
                        source_node.execution["output_mode"] = "last_timestep"
                continue

        spec: Optional[Tuple[str, Dict[str, Any]]] = None
        target_label = str(fx_node.target)
        if fx_node.op == "call_module":
            module = modules.get(str(fx_node.target))
            spec = _module_spec(module) if module is not None else None
        elif fx_node.op == "call_function":
            spec = _functional_spec(fx_node)
        elif fx_node.op == "call_method":
            if str(fx_node.target) in ("contiguous", "clone", "detach"):
                deps = _fx_dependencies(fx_node.args[0] if fx_node.args else None)
                if len(deps) == 1:
                    aliases[fx_node] = deps[0]
                    continue
            spec = _method_spec(fx_node)

        if spec is None:
            unsupported.append({"node": fx_node.name, "op": fx_node.op, "target": target_label})
            continue

        node_type, params = spec
        execution = {
            "source": "torch.fx",
            "fx_op": fx_node.op,
            "fx_target": target_label,
            "editable": True,
        }
        if node_type in ("LSTM", "GRU"):
            execution["output_mode"] = "sequence"
        descriptor_id = allocate_id(fx_node.name)
        descriptor_node = Node(id=descriptor_id, type=node_type, params=make_json_serializable(params), execution=execution)
        descriptor_nodes.append(descriptor_node)
        fx_to_descriptor[fx_node] = descriptor_id

        dependencies = []
        for dependency in _fx_dependencies((fx_node.args, fx_node.kwargs)):
            source_id = resolve(dependency)
            if source_id and source_id not in dependencies:
                dependencies.append(source_id)
        if len(dependencies) > 1 and node_type not in ("Add", "Concat", "MultiheadAttention"):
            unsupported.append({
                "node": fx_node.name,
                "op": fx_node.op,
                "target": f"{target_label} (multiple tensor inputs)",
            })
            descriptor_nodes.pop()
            used_descriptor_ids.discard(descriptor_id)
            fx_to_descriptor.pop(fx_node, None)
            continue
        for source_id in dependencies:
            descriptor_edges.append(Edge(source=source_id, target=descriptor_id))

    if unsupported:
        preview = ", ".join(f"{item['node']}={item['target']}" for item in unsupported[:6])
        suffix = "" if len(unsupported) <= 6 else f" (+{len(unsupported) - 6} more)"
        raise TorchPackageImportError(
            "The packaged model was loaded and traced, but contains operations that MYLO cannot faithfully edit yet: "
            f"{preview}{suffix}."
        )

    output_nodes = [node for node in graph_module.graph.nodes if node.op == "output"]
    output_dependencies = _fx_dependencies(output_nodes[0].args if output_nodes else ())
    output_ids = []
    for dependency in output_dependencies:
        source_id = resolve(dependency)
        if source_id and source_id not in output_ids:
            output_ids.append(source_id)
    if len(output_ids) != 1:
        raise TorchPackageImportError(
            f"MYLO descriptors currently require one tensor output; the FX graph exposes {len(output_ids)} outputs."
        )
    descriptor_edges.append(Edge(source=output_ids[0], target="output"))

    input_shape = _shape_from_tensor(example_input)
    with torch.no_grad():
        original_output = _primary_output(model(example_input))
    if original_output is None:
        raise TorchPackageImportError("The packaged model does not return a single primary tensor output.")
    output_shape = _shape_from_tensor(original_output)

    tensor_contracts: Dict[str, TensorContract] = {
        "input": TensorContract(shape=input_shape, dtype=str(example_input.dtype).replace("torch.", "")),
        "output": TensorContract(shape=output_shape, dtype=str(original_output.dtype).replace("torch.", "")),
    }
    for fx_node, descriptor_id in fx_to_descriptor.items():
        if descriptor_id == "input":
            continue
        shape = _tensor_meta_shape(fx_node)
        if shape:
            shape[0] = -1
            tensor_contracts[descriptor_id] = TensorContract(shape=shape, dtype=_tensor_meta_dtype(fx_node))

    descriptor = ArchitectureDescriptor(
        model_name=model.__class__.__name__,
        input_shape=input_shape,
        output_shape=output_shape,
        nodes=descriptor_nodes,
        edges=descriptor_edges,
        tensor_contracts=tensor_contracts,
        propagation_rules="torch_fx_import",
    )
    descriptor.normalize_inplace(strict=True)
    descriptor.validate(strict=True)
    return descriptor, {
        "fx_node_count": len(list(graph_module.graph.nodes)),
        "descriptor_node_count": len(descriptor.nodes),
        "input_shape": input_shape,
        "output_shape": output_shape,
    }


def _verify_descriptor(
    original: nn.Module,
    descriptor: ArchitectureDescriptor,
    example_input: torch.Tensor,
) -> Tuple[nn.Module, Dict[str, Any]]:
    rebuilt = DescriptorModelBuilder.build(descriptor)
    original_state = original.state_dict()
    remapped_state: Dict[str, torch.Tensor] = {}
    module_nodes = [
        node for node in descriptor.nodes
        if (node.execution or {}).get("fx_op") == "call_module"
    ]
    for source_key, tensor in original_state.items():
        mapped_key = source_key
        best_target = ""
        best_node_id = ""
        for node in module_nodes:
            fx_target = str((node.execution or {}).get("fx_target", ""))
            if fx_target and (source_key == fx_target or source_key.startswith(f"{fx_target}.")):
                if len(fx_target) > len(best_target):
                    best_target = fx_target
                    best_node_id = node.id
        if best_target:
            remainder = source_key[len(best_target):].lstrip(".")
            mapped_key = f"node_modules.{best_node_id}"
            best_node = next((node for node in module_nodes if node.id == best_node_id), None)
            if best_node and best_node.type == "MultiheadAttention":
                mapped_key = f"{mapped_key}.attention"
            elif best_node and best_node.type == "TransformerEncoderLayer":
                mapped_key = f"{mapped_key}.layer"
            if remainder:
                mapped_key = f"{mapped_key}.{remainder}"
        remapped_state[mapped_key] = tensor
    transfer = WeightCompatibilityEngine.transfer_weights(remapped_state, rebuilt)
    try:
        original.eval()
    except NotImplementedError:
        pass
    rebuilt.eval()
    with torch.no_grad():
        expected = _primary_output(original(example_input))
        actual = _primary_output(rebuilt(example_input))
    if expected is None or actual is None:
        raise TorchPackageImportError("Equivalence validation requires a tensor output from both models.")
    if expected.shape != actual.shape:
        raise TorchPackageImportError(
            f"Generated descriptor output shape {list(actual.shape)} does not match packaged model output {list(expected.shape)}."
        )
    if expected.dtype.is_floating_point or expected.dtype.is_complex:
        equivalent = bool(torch.allclose(expected, actual, rtol=1e-4, atol=1e-5, equal_nan=True))
        max_error = float((expected - actual).abs().max().item()) if expected.numel() else 0.0
    else:
        equivalent = bool(torch.equal(expected, actual))
        max_error = 0.0 if equivalent else None
    if not equivalent:
        raise TorchPackageImportError(
            "The FX graph was converted, but the rebuilt descriptor did not reproduce the packaged model output "
            f"(maximum absolute error: {max_error}). The model was not imported as editable."
        )
    return rebuilt, {
        "equivalent": True,
        "max_absolute_error": max_error,
        "matched_weight_keys": len(transfer.get("matched_keys", [])),
        "unmatched_source_keys": transfer.get("unmatched_source", []),
        "unmatched_target_keys": transfer.get("unmatched_target", []),
    }


def import_torch_package(path: str) -> TorchPackageImportResult:
    model, package_info = load_packaged_module(path)
    manifest = package_info.get("manifest", {})
    try:
        graph_module = symbolic_trace(model)
    except Exception as exc:
        raise TorchPackageImportError(f"torch.fx could not trace the packaged model: {exc}") from exc

    example_input, input_source = _infer_example_input(model, graph_module, manifest)
    try:
        embedded_descriptor = manifest.get("descriptor") if isinstance(manifest, dict) else None
        if isinstance(embedded_descriptor, dict):
            descriptor = ArchitectureDescriptor.from_dict(embedded_descriptor)
            descriptor.normalize_inplace(strict=True)
            descriptor.validate(strict=True)
            conversion = {
                "origin": "embedded_mylo_descriptor",
                "descriptor_node_count": len(descriptor.nodes),
                "input_shape": descriptor.input_shape,
                "output_shape": descriptor.output_shape,
            }
        else:
            descriptor, conversion = descriptor_from_fx(model, graph_module, example_input)
        rebuilt, verification = _verify_descriptor(model, descriptor, example_input)
    except TorchPackageImportError:
        raise
    except Exception as exc:
        raise TorchPackageImportError(f"Could not generate a valid MYLO descriptor from the FX graph: {exc}") from exc
    report = {
        "format": "torch.package",
        "editable": True,
        "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "payload": {"package": package_info["package"], "resource": package_info["resource"]},
        "input_inference": input_source,
        "conversion": conversion,
        "verification": verification,
        "security": {
            "trusted_code_required": True,
            "in_process_sandbox": False,
            "warning": "torch.package loading executes Python/pickle code; isolate untrusted imports at the OS level",
        },
    }
    return TorchPackageImportResult(model=model, descriptor=descriptor, rebuilt_model=rebuilt, report=report)


def extract_packaged_state_dict(path: str) -> Optional[Dict[str, torch.Tensor]]:
    try:
        imported = import_torch_package(path)
        return imported.rebuilt_model.state_dict()
    except Exception:
        return None


def export_mylo_torch_package(
    model: nn.Module,
    descriptor: ArchitectureDescriptor,
    output_path: str,
) -> str:
    manifest = {
        "schema_version": "1.0",
        "producer": "MYLO",
        "model": {"package": MYLO_PACKAGE_NAMESPACE, "resource": MYLO_MODEL_RESOURCE},
        "input_shape": descriptor.input_shape,
        "input_dtype": descriptor.tensor_contracts.get("input", TensorContract(descriptor.input_shape)).dtype,
        "descriptor": descriptor.to_dict(),
    }
    with PackageExporter(output_path) as exporter:
        exporter.extern("torch.**")
        exporter.intern("Core.**")
        exporter.save_pickle(MYLO_PACKAGE_NAMESPACE, MYLO_MODEL_RESOURCE, model)
        exporter.save_text(MYLO_PACKAGE_NAMESPACE, MYLO_MANIFEST_RESOURCE, json.dumps(manifest, indent=2))
    return output_path
