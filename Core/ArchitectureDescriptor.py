import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Set, Tuple


@dataclass
class Edge:
    source: str
    target: str
    source_port: str = "output"
    target_port: str = "input"


@dataclass
class Node:
    id: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    io: Dict[str, List[str]] = field(default_factory=lambda: {"inputs": ["input"], "outputs": ["output"]})
    execution: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TensorContract:
    shape: List[int]
    dtype: str = "float32"


@dataclass
class ArchitectureDescriptor:
    model_name: str
    input_shape: List[int]
    output_shape: List[int]
    nodes: List[Node]
    edges: List[Edge]
    tensor_contracts: Dict[str, TensorContract] = field(default_factory=dict)
    merge_rules: Dict[str, Any] = field(default_factory=dict)
    propagation_rules: str = "deterministic_forward"

    def validate(self, strict: bool = True) -> bool:
        """Perform static validation of the descriptor graph and tensor contracts."""
        node_ids = {node.id for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("Duplicate node ids detected in descriptor")

        for edge in self.edges:
            if edge.source not in node_ids and edge.source != "input":
                raise ValueError(f"Edge source not found: {edge.source}")
            if edge.target not in node_ids and edge.target != "output":
                raise ValueError(f"Edge target not found: {edge.target}")

        self._validate_acyclic()
        self._validate_reachability(node_ids, strict=strict)
        self._validate_ports(node_ids, strict=strict)
        self._validate_multi_input_nodes(strict=strict)
        self._validate_shape_propagation(strict=strict)
        return True

    def normalize_inplace(self, strict: bool = True) -> "ArchitectureDescriptor":
        """Apply deterministic shape propagation to node parameters in place."""
        self._propagate_shapes(mutate=True, strict=strict)
        return self

    def _validate_acyclic(self):
        indegree: Dict[str, int] = {node.id: 0 for node in self.nodes}
        indegree["input"] = 0
        indegree["output"] = 0
        adj: Dict[str, List[str]] = {node.id: [] for node in self.nodes}
        adj["input"] = []

        for edge in self.edges:
            if edge.target in indegree:
                indegree[edge.target] += 1
            if edge.source in adj:
                adj[edge.source].append(edge.target)

        queue = [name for name, deg in indegree.items() if deg == 0]
        visited = 0
        while queue:
            curr = queue.pop(0)
            visited += 1
            for neighbor in adj.get(curr, []):
                if neighbor in indegree:
                    indegree[neighbor] -= 1
                    if indegree[neighbor] == 0:
                        queue.append(neighbor)

        if visited != len(indegree):
            raise ValueError("Descriptor graph contains cycles")

    def _validate_reachability(self, node_ids: Set[str], strict: bool = True):
        reachable = {"input"}
        queue = ["input"]

        adjacency: Dict[str, List[str]] = {"input": []}
        for node_id in node_ids:
            adjacency[node_id] = []

        for edge in self.edges:
            adjacency.setdefault(edge.source, []).append(edge.target)

        while queue:
            current = queue.pop(0)
            for neighbor in adjacency.get(current, []):
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)

        if strict and "output" not in reachable:
            raise ValueError("Descriptor has no path to output from input")

        if strict:
            disconnected_nodes = [node_id for node_id in node_ids if node_id not in reachable]
            if disconnected_nodes:
                raise ValueError(f"Descriptor contains disconnected nodes: {disconnected_nodes}")

    def _validate_ports(self, node_ids: Set[str], strict: bool = True):
        incoming: Dict[str, List[str]] = {node_id: [] for node_id in node_ids}
        outgoing: Dict[str, List[str]] = {node_id: [] for node_id in node_ids}

        for edge in self.edges:
            if edge.source in node_ids:
                outgoing[edge.source].append(edge.target)
            if edge.target in node_ids:
                incoming[edge.target].append(edge.source)

        sources_to_output = [edge.source for edge in self.edges if edge.target == "output"]
        if strict and not sources_to_output:
            raise ValueError("Descriptor has no path to output sink")

        for node in self.nodes:
            if node.type == "Input":
                continue
            if not incoming[node.id] and not any(edge.source == "input" and edge.target == node.id for edge in self.edges):
                if strict and node.id not in sources_to_output:
                    raise ValueError(f"Dangling node with no inputs: {node.id}")

    def _validate_multi_input_nodes(self, strict: bool = True):
        incoming_counts: Dict[str, int] = {node.id: 0 for node in self.nodes}
        for edge in self.edges:
            if edge.target in incoming_counts:
                incoming_counts[edge.target] += 1

        allowed_multi_input = {"Add", "Concat"}
        for node in self.nodes:
            if incoming_counts.get(node.id, 0) > 1 and node.type not in allowed_multi_input:
                if strict:
                    raise ValueError(
                        f"Node {node.id} has multiple inputs but type {node.type} is not an explicit merge op"
                    )

    def _validate_shape_propagation(self, strict: bool = True):
        try:
            shapes = self._propagate_shapes(mutate=False, strict=strict)
        except Exception as e:
            if strict:
                raise e
            return

        output_sources = [edge.source for edge in self.edges if edge.target == "output"]
        if output_sources:
            final_shape = shapes.get(output_sources[0])
            if final_shape and not self._shapes_compatible(final_shape, self.output_shape):
                if strict:
                    raise ValueError(
                        f"Output shape mismatch: propagated {final_shape}, expected {self.output_shape}"
                    )

    def _propagate_shapes(self, mutate: bool, strict: bool = True) -> Dict[str, List[int]]:
        shapes: Dict[str, List[int]] = {"input": list(self.input_shape)}
        try:
            sorted_nodes = self._topological_node_order()
        except Exception as e:
            if strict:
                raise e
            return shapes

        for node_id in sorted_nodes:
            node = next(n for n in self.nodes if n.id == node_id)
            input_shapes = self._collect_input_shapes(node_id, shapes)
            if not input_shapes:
                continue

            try:
                in_shape = input_shapes[0] if len(input_shapes) == 1 else self._merge_shapes(input_shapes, node, strict=strict)
                in_features = self._infer_in_features(node, in_shape)

                if node.type == "Linear":
                    expected = node.params.get("in_features")
                    if expected is not None and expected != in_features and not mutate:
                        if strict:
                            raise ValueError(f"Linear node {node.id} expects in_features={expected}, got {in_features}")
                    if mutate:
                        node.params["in_features"] = in_features
                elif node.type in ("Conv1d", "Conv2d", "ConvTranspose1d", "ConvTranspose2d"):
                    expected = node.params.get("in_channels")
                    if expected is not None and expected != in_features and not mutate:
                        if strict:
                            raise ValueError(f"{node.type} node {node.id} expects in_channels={expected}, got {in_features}")
                    if mutate:
                        node.params["in_channels"] = in_features
                elif node.type in ("LSTM", "GRU"):
                    expected = node.params.get("input_size")
                    if expected is not None and expected != in_features and not mutate:
                        if strict:
                            raise ValueError(f"{node.type} node {node.id} expects input_size={expected}, got {in_features}")
                    if mutate:
                        node.params["input_size"] = in_features
                elif node.type in ("BatchNorm1d", "BatchNorm2d"):
                    expected = node.params.get("num_features")
                    if expected is not None and expected != in_features and not mutate:
                        if strict:
                            raise ValueError(f"{node.type} node {node.id} expects num_features={expected}, got {in_features}")
                    if mutate:
                        node.params["num_features"] = in_features

                out_shape = self._infer_output_shape(node, in_shape)
                shapes[node_id] = out_shape
            except Exception as e:
                if strict:
                    raise e
                if input_shapes:
                    shapes[node_id] = input_shapes[0]

        return shapes

    def _infer_in_features(self, node: Node, in_shape: List[int]) -> int:
        if node.type in ("Conv1d", "Conv2d", "ConvTranspose1d", "ConvTranspose2d", "BatchNorm1d", "BatchNorm2d"):
            if len(in_shape) >= 2:
                return in_shape[1]
            return in_shape[-1]
        if node.type in ("LSTM", "GRU"):
            if len(in_shape) >= 3:
                return in_shape[-1]
            return in_shape[-1]
        return in_shape[-1]

    def _topological_node_order(self) -> List[str]:
        indegree = {node.id: 0 for node in self.nodes}
        adj = {node.id: [] for node in self.nodes}
        adj["input"] = []

        for edge in self.edges:
            if edge.target in indegree:
                indegree[edge.target] += 1
            if edge.source in adj:
                adj[edge.source].append(edge.target)

        queue = ["input"]
        order: List[str] = []
        while queue:
            curr = queue.pop(0)
            for neighbor in adj.get(curr, []):
                if neighbor in indegree:
                    indegree[neighbor] -= 1
                    if indegree[neighbor] == 0:
                        queue.append(neighbor)
                        if neighbor != "output":
                            order.append(neighbor)
        return order

    def _collect_input_shapes(self, node_id: str, shapes: Dict[str, List[int]]) -> List[List[int]]:
        collected: List[List[int]] = []
        for edge in self.edges:
            if edge.target == node_id and edge.source in shapes:
                collected.append(shapes[edge.source])
        return collected

    def _merge_shapes(self, input_shapes: List[List[int]], node: Node, strict: bool = True) -> List[int]:
        if len(input_shapes) <= 1:
            return input_shapes[0]

        if node.type == "Add":
            reference = input_shapes[0]
            for candidate in input_shapes[1:]:
                if not self._shapes_compatible(candidate, reference):
                    if strict:
                        raise ValueError(f"Add node {node.id} requires compatible input shapes")
            return list(reference)

        if node.type == "Concat":
            concat_dim = int(node.params.get("dim", -1))
            rank = len(input_shapes[0])
            concat_dim = concat_dim if concat_dim >= 0 else rank + concat_dim
            if concat_dim < 0 or concat_dim >= rank:
                if strict:
                    raise ValueError(f"Concat node {node.id} has invalid dim {node.params.get('dim', -1)}")
                return list(input_shapes[0])

            reference = list(input_shapes[0])
            total = reference[concat_dim]
            for candidate in input_shapes[1:]:
                if len(candidate) != rank:
                    if strict:
                        raise ValueError(f"Concat node {node.id} requires matching ranks")
                    continue
                for index, (ref_dim, cand_dim) in enumerate(zip(reference, candidate)):
                    if index == concat_dim:
                        continue
                    if ref_dim != -1 and cand_dim != -1 and ref_dim != cand_dim:
                        if strict:
                            raise ValueError(f"Concat node {node.id} requires matching non-concat dimensions")
                if total == -1 or candidate[concat_dim] == -1:
                    total = -1
                else:
                    total += candidate[concat_dim]

            output = list(reference)
            output[concat_dim] = total
            return output

        if strict:
            raise ValueError(f"Node {node.id} received multiple inputs but type {node.type} is not an explicit merge op")
        return list(input_shapes[0])

    def _infer_output_shape(self, node: Node, in_shape: List[int]) -> List[int]:
        params = node.params
        batch_prefix = in_shape[:-1] if len(in_shape) > 1 else [in_shape[0] if in_shape else -1]

        def spatial_values(value, rank: int, default: int = 0) -> List[int]:
            if isinstance(value, str):
                return [value] * rank
            if isinstance(value, (list, tuple)):
                values = list(value)
                if len(values) == 1:
                    values *= rank
                return values[:rank]
            return [default if value is None else value] * rank

        def conv_dim(size, kernel, stride, padding, dilation, *, transpose=False, output_padding=0, ceil_mode=False):
            if not isinstance(size, int) or size <= 0:
                return -1
            if isinstance(padding, str):
                if padding.lower() == "same" and not transpose:
                    return size
                padding = 0
            kernel, stride, padding, dilation = int(kernel), int(stride), int(padding), int(dilation)
            if transpose:
                return (size - 1) * stride - 2 * padding + dilation * (kernel - 1) + int(output_padding) + 1
            numerator = size + 2 * padding - dilation * (kernel - 1) - 1
            if ceil_mode:
                return int(-(-numerator // stride) + 1)
            return int(numerator // stride + 1)

        def normalize_dim(dim: int, rank: int) -> int:
            return dim if dim >= 0 else rank + dim + 1

        def normalize_dims(value, rank: int) -> List[int]:
            if value is None:
                return []
            if isinstance(value, (list, tuple)):
                return sorted({normalize_dim(int(dim), rank) for dim in value})
            return [normalize_dim(int(value), rank)]

        def remove_dims(shape: List[int], dims: List[int], keepdim: bool) -> List[int]:
            if not dims:
                dims = list(range(1, len(shape)))
            result = []
            for index, size in enumerate(shape):
                if index in dims:
                    if keepdim:
                        result.append(1)
                    continue
                result.append(size)
            return result or ([shape[0]] if shape else [-1])

        def permute_shape(shape: List[int], dims: List[int]) -> List[int]:
            if len(dims) != len(shape):
                return list(shape)
            if sorted(dims) != list(range(len(shape))):
                return list(shape)
            return [shape[index] for index in dims]

        def flatten_shape(shape: List[int], start_dim: int, end_dim: int) -> List[int]:
            rank = len(shape)
            if rank == 0:
                return shape

            start = normalize_dim(start_dim, rank)
            end = normalize_dim(end_dim, rank)
            start = max(0, min(start, rank - 1))
            end = max(0, min(end, rank - 1))
            if start > end:
                start, end = end, start

            prefix = list(shape[:start])
            suffix = list(shape[end + 1:])
            segment = shape[start:end + 1]
            if all(isinstance(dim, int) and dim > 0 for dim in segment):
                flattened = 1
                for dim in segment:
                    flattened *= dim
            else:
                flattened = -1
            return prefix + [flattened] + suffix

        if node.type == "Unsqueeze":
            dim = params.get("dim", 0)
            rank = len(in_shape)
            insert_at = normalize_dim(dim, rank)
            insert_at = max(0, min(insert_at, rank))
            return list(in_shape[:insert_at]) + [1] + list(in_shape[insert_at:])

        if node.type == "Squeeze":
            dims = normalize_dims(params.get("dim"), len(in_shape))
            if not dims:
                return [dim for dim in in_shape if dim != 1] or [-1]
            return [size for index, size in enumerate(in_shape) if index not in dims] or [-1]

        if node.type == "ReduceMean":
            dims = normalize_dims(params.get("dim", params.get("axes")), len(in_shape))
            keepdim = bool(params.get("keepdim", False))
            return remove_dims(list(in_shape), dims, keepdim)

        if node.type == "Transpose":
            dim0 = params.get("dim0")
            dim1 = params.get("dim1")
            if dim0 is None or dim1 is None:
                return list(in_shape)
            rank = len(in_shape)
            d0 = normalize_dim(int(dim0), rank)
            d1 = normalize_dim(int(dim1), rank)
            if d0 >= rank or d1 >= rank:
                return list(in_shape)
            output = list(in_shape)
            output[d0], output[d1] = output[d1], output[d0]
            return output

        if node.type == "Permute":
            dims = params.get("dims")
            if not isinstance(dims, (list, tuple)):
                return list(in_shape)
            normalized = [normalize_dim(int(dim), len(in_shape)) for dim in dims]
            return permute_shape(list(in_shape), normalized)

        if node.type == "Add":
            return list(in_shape)

        if node.type == "Concat":
            return list(in_shape)

        if node.type == "Linear":
            return batch_prefix + [params.get("out_features", in_shape[-1])]
        if node.type == "Conv1d":
            out_channels = params.get("out_channels", in_shape[1] if len(in_shape) > 1 else in_shape[-1])
            length = in_shape[2] if len(in_shape) > 2 else -1
            kernel = spatial_values(params.get("kernel_size", 1), 1, 1)[0]
            stride = spatial_values(params.get("stride", 1), 1, 1)[0]
            padding = spatial_values(params.get("padding", 0), 1, 0)[0]
            dilation = spatial_values(params.get("dilation", 1), 1, 1)[0]
            return [in_shape[0] if in_shape else -1, out_channels, conv_dim(length, kernel, stride, padding, dilation)]
        if node.type == "Conv2d":
            out_channels = params.get("out_channels", in_shape[1] if len(in_shape) > 1 else in_shape[-1])
            height = in_shape[2] if len(in_shape) > 2 else -1
            width = in_shape[3] if len(in_shape) > 3 else -1
            kernel = spatial_values(params.get("kernel_size", 1), 2, 1)
            stride = spatial_values(params.get("stride", 1), 2, 1)
            padding = spatial_values(params.get("padding", 0), 2, 0)
            dilation = spatial_values(params.get("dilation", 1), 2, 1)
            return [
                in_shape[0] if in_shape else -1,
                out_channels,
                conv_dim(height, kernel[0], stride[0], padding[0], dilation[0]),
                conv_dim(width, kernel[1], stride[1], padding[1], dilation[1]),
            ]
        if node.type in ("LSTM", "GRU"):
            hidden = params.get("hidden_size", in_shape[-1])
            batch_first = bool(params.get("batch_first", True))
            batch = (in_shape[0] if batch_first else in_shape[1]) if len(in_shape) >= 2 else -1
            output_mode = (getattr(node, "execution", {}) or {}).get("output_mode", "auto")
            if output_mode == "sequence" and len(in_shape) >= 3:
                directions = 2 if params.get("bidirectional", False) else 1
                if batch_first:
                    return [batch, in_shape[1], hidden * directions]
                return [in_shape[0], in_shape[1], hidden * directions]
            directions = 2 if params.get("bidirectional", False) else 1
            return [batch, hidden * directions]
        if node.type in ("Input", "Output", "Identity", "Add", "Concat", "LayerNorm", "TransformerEncoderLayer"):
            return list(in_shape)
        if node.type == "Reshape":
            target_shape = params.get("target_shape")
            if isinstance(target_shape, (list, tuple)) and target_shape:
                batch = in_shape[0] if in_shape else -1
                return [batch, *list(target_shape)]
            return list(in_shape)
        if node.type == "Embedding":
            embedding_dim = params.get("embedding_dim", in_shape[-1])
            return list(in_shape) + [embedding_dim]
        if node.type in ("ConvTranspose1d", "ConvTranspose2d"):
            spatial_rank = 1 if node.type.endswith("1d") else 2
            out_channels = params.get("out_channels", in_shape[1] if len(in_shape) > 1 else -1)
            kernel = spatial_values(params.get("kernel_size", 1), spatial_rank, 1)
            stride = spatial_values(params.get("stride", 1), spatial_rank, 1)
            padding = spatial_values(params.get("padding", 0), spatial_rank, 0)
            dilation = spatial_values(params.get("dilation", 1), spatial_rank, 1)
            output_padding = spatial_values(params.get("output_padding", 0), spatial_rank, 0)
            spatial = list(in_shape[2:2 + spatial_rank])
            while len(spatial) < spatial_rank:
                spatial.append(-1)
            output_spatial = [
                conv_dim(spatial[i], kernel[i], stride[i], padding[i], dilation[i],
                         transpose=True, output_padding=output_padding[i])
                for i in range(spatial_rank)
            ]
            return [in_shape[0] if in_shape else -1, out_channels, *output_spatial]
        if node.type in ("MaxPool1d", "AvgPool1d", "MaxPool2d", "AvgPool2d"):
            spatial_rank = 1 if node.type.endswith("1d") else 2
            kernel = spatial_values(params.get("kernel_size", 1), spatial_rank, 1)
            stride = spatial_values(params.get("stride", params.get("kernel_size", 1)), spatial_rank, 1)
            padding = spatial_values(params.get("padding", 0), spatial_rank, 0)
            dilation = spatial_values(params.get("dilation", 1), spatial_rank, 1)
            spatial = list(in_shape[2:2 + spatial_rank])
            while len(spatial) < spatial_rank:
                spatial.append(-1)
            output_spatial = [
                conv_dim(spatial[i], kernel[i], stride[i], padding[i], dilation[i],
                         ceil_mode=bool(params.get("ceil_mode", False)))
                for i in range(spatial_rank)
            ]
            return [in_shape[0] if in_shape else -1, in_shape[1] if len(in_shape) > 1 else -1, *output_spatial]
        if node.type == "Flatten":
            start_dim = params.get("start_dim", 1)
            end_dim = params.get("end_dim", -1)
            return flatten_shape(list(in_shape), start_dim, end_dim)
        return list(in_shape)

    @staticmethod
    def _shapes_compatible(actual: List[int], expected: List[int]) -> bool:
        if len(actual) != len(expected):
            return False
        for a, e in zip(actual, expected):
            if e == -1 or a == -1:
                continue
            if a != e:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
            "tensor_contracts": {k: asdict(v) for k, v in self.tensor_contracts.items()},
            "merge_rules": self.merge_rules,
            "propagation_rules": self.propagation_rules,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArchitectureDescriptor":
        nodes = [Node(**n) for n in data.get("nodes", [])]
        edges = [Edge(**e) for e in data.get("edges", [])]
        contracts = {
            k: TensorContract(**v)
            for k, v in data.get("tensor_contracts", {}).items()
        }
        return cls(
            model_name=data["model_name"],
            input_shape=data["input_shape"],
            output_shape=data["output_shape"],
            nodes=nodes,
            edges=edges,
            tensor_contracts=contracts,
            merge_rules=data.get("merge_rules", {}),
            propagation_rules=data.get("propagation_rules", "deterministic_forward"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ArchitectureDescriptor":
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def default_feedforward(cls, input_dim: int, output_dim: int, hidden: int = 64) -> "ArchitectureDescriptor":
        mid = max(16, hidden // 2)
        return cls.from_dict({
            "model_name": "FeedforwardRegressor",
            "input_shape": [-1, input_dim],
            "output_shape": [-1, output_dim],
            "nodes": [
                {"id": "dense1", "type": "Linear", "params": {"in_features": input_dim, "out_features": hidden}},
                {"id": "relu1", "type": "ReLU", "params": {}},
                {"id": "dense2", "type": "Linear", "params": {"in_features": hidden, "out_features": mid}},
                {"id": "relu2", "type": "ReLU", "params": {}},
                {"id": "out_linear", "type": "Linear", "params": {"in_features": mid, "out_features": output_dim}},
            ],
            "edges": [
                {"source": "input", "target": "dense1"},
                {"source": "dense1", "target": "relu1"},
                {"source": "relu1", "target": "dense2"},
                {"source": "dense2", "target": "relu2"},
                {"source": "relu2", "target": "out_linear"},
                {"source": "out_linear", "target": "output"},
            ],
        })
