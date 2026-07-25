import copy
import json
import random
import os
import tempfile
import logging
import math
import traceback
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import time
from Core.ArchitectureDescriptor import ArchitectureDescriptor, Node, Edge
from Core.DescriptorModelBuilder import DescriptorModelBuilder
from Core.WeightCompatibilityEngine import WeightCompatibilityEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def _sanitize_float(value: float) -> Any:
    """Replace inf/nan with JSON-compliant values."""
    if isinstance(value, str):
        return value
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if math.isnan(value):
        return "nan"
    return value


@dataclass
class GenerationStats:
    """Track per-generation statistics for diagnostics."""
    generation: int
    tier1_survivors: int = 0
    tier2_survivors: int = 0
    tier1_failures: int = 0
    tier2_failures: int = 0
    finals_failures: int = 0
    mutation_attempts: int = 0
    mutation_successes: int = 0
    best_train_loss: float = float("inf")
    best_val_loss: float = float("inf")
    best_score: float = float("inf")
    avg_tier1_score: float = float("inf")
    structural_mutations: int = 0
    width_mutations: int = 0
    activation_mutations: int = 0
    hyperparam_mutations: int = 0
    layer_swap_mutations: int = 0
    shape_errors: int = 0
    nan_errors: int = 0
    other_errors: int = 0

    def mutation_success_rate(self) -> float:
        if self.mutation_attempts == 0:
            return 0.0
        return self.mutation_successes / self.mutation_attempts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "tier1_survivors": self.tier1_survivors,
            "tier2_survivors": self.tier2_survivors,
            "tier1_failures": self.tier1_failures,
            "tier2_failures": self.tier2_failures,
            "finals_failures": self.finals_failures,
            "mutation_attempts": self.mutation_attempts,
            "mutation_successes": self.mutation_successes,
            "mutation_success_rate": f"{self.mutation_success_rate():.1%}",
            "best_train_loss": _sanitize_float(round(self.best_train_loss, 6) if not math.isinf(self.best_train_loss) and not math.isnan(self.best_train_loss) else self.best_train_loss),
            "best_val_loss": _sanitize_float(round(self.best_val_loss, 6) if not math.isinf(self.best_val_loss) and not math.isnan(self.best_val_loss) else self.best_val_loss),
            "best_score": _sanitize_float(round(self.best_score, 6) if not math.isinf(self.best_score) and not math.isnan(self.best_score) else self.best_score),
            "avg_tier1_score": _sanitize_float(round(self.avg_tier1_score, 6) if not math.isinf(self.avg_tier1_score) and not math.isnan(self.avg_tier1_score) else self.avg_tier1_score),
            "structural_mutations": self.structural_mutations,
            "width_mutations": self.width_mutations,
            "activation_mutations": self.activation_mutations,
            "hyperparam_mutations": self.hyperparam_mutations,
            "layer_swap_mutations": self.layer_swap_mutations,
            "shape_errors": self.shape_errors,
            "nan_errors": self.nan_errors,
            "other_errors": self.other_errors,
        }


class MutationGrammar:
    MAX_NODES = 64
    MAX_DEPTH = 16
    WIDTH_FACTORS = [0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0]
    DROPOUT_RATES = [0.0, 0.1, 0.2, 0.3, 0.5]
    RECURRENT_TYPES = ["LSTM", "GRU"]
    ACTIVATION_TYPES = ["ReLU", "Tanh", "Sigmoid", "GELU", "SiLU"]
    NORM_TYPES = ["BatchNorm1d", "BatchNorm2d", "LayerNorm"]

    @staticmethod
    def scale_width(descriptor: ArchitectureDescriptor, target_node_id: str,
                    factor: float, event: Optional[dict] = None) -> bool:
        """Scale the output width of a Linear/Conv/LSTM node. normalize_inplace handles downstream propagation."""
        node = next(
            (n for n in descriptor.nodes if n.id == target_node_id),
            None)
        if not node:
            return False

        if node.type not in ["Linear", "Conv1d", "Conv2d", "LSTM", "GRU"]:
            return False

        param_to_scale = ""
        old_out = 0

        if node.type == "Linear":
            old_out = node.params.get("out_features", 0)
            param_to_scale = "out_features"
        elif "Conv" in node.type:
            old_out = node.params.get("out_channels", 0)
            param_to_scale = "out_channels"
        elif node.type in ["LSTM", "GRU"]:
            old_out = node.params.get("hidden_size", 0)
            param_to_scale = "hidden_size"

        if not old_out:
            return False

        new_out = max(1, int(old_out * factor))
        node.params[param_to_scale] = new_out

        descriptor.normalize_inplace()
        if event is not None:
            event.update({
                "op": "scale_width",
                "node": target_node_id,
                "node_type": node.type,
                "before": {param_to_scale: old_out},
                "after": {param_to_scale: new_out},
                "detail": f"{node.type} '{target_node_id}': {param_to_scale} {old_out} → {new_out} (×{factor})",
            })
        return True

    @staticmethod
    def mutate_activation(descriptor: ArchitectureDescriptor,
                          target_node_id: str, event: Optional[dict] = None) -> bool:
        """Randomly swap the activation type of an activation node."""
        node = next(
            (n for n in descriptor.nodes if n.id == target_node_id),
            None)
        if not node or node.type not in MutationGrammar.ACTIVATION_TYPES:
            return False
        choices = [
            t for t in MutationGrammar.ACTIVATION_TYPES if t != node.type]
        old_type = node.type
        node.type = random.choice(choices)
        # Rename the node without breaking edges.
        old_id = node.id
        new_id = f"{node.type.lower()}_{old_id.split('_')[-1]}"
        node.id = new_id
        for edge in descriptor.edges:
            if edge.source == old_id:
                edge.source = new_id
            if edge.target == old_id:
                edge.target = new_id
        logger.debug(
            f"Activation mutated: {old_type} -> {node.type} (id: {old_id} -> {new_id})")
        if event is not None:
            event.update({
                "op": "mutate_activation",
                "node": new_id,
                "node_type": node.type,
                "before": {"activation": old_type},
                "after": {"activation": node.type},
                "detail": f"Activation swapped: {old_type} → {node.type}",
            })
        return True

    @staticmethod
    def mutate_dropout_rate(descriptor: ArchitectureDescriptor,
                            target_node_id: str, event: Optional[dict] = None) -> bool:
        """Change the dropout probability of a Dropout node."""
        node = next(
            (n for n in descriptor.nodes if n.id == target_node_id),
            None)
        if not node or node.type != "Dropout":
            return False
        current_p = node.params.get("p", 0.2)
        delta = random.choice([-0.1, -0.05, 0.0, 0.05, 0.1])
        new_p = max(0.0, min(0.8, round(current_p + delta, 2)))
        node.params["p"] = new_p
        if event is not None:
            event.update({
                "op": "mutate_dropout_rate",
                "node": target_node_id,
                "node_type": "Dropout",
                "before": {"p": current_p},
                "after": {"p": new_p},
                "detail": f"Dropout '{target_node_id}': p {current_p} → {new_p}",
            })
        return True

    @staticmethod
    def swap_recurrent_type(descriptor: ArchitectureDescriptor,
                            target_node_id: str, event: Optional[dict] = None) -> bool:
        """Swap LSTM <-> GRU while preserving dimensions."""
        node = next(
            (n for n in descriptor.nodes if n.id == target_node_id),
            None)
        if not node or node.type not in MutationGrammar.RECURRENT_TYPES:
            return False

        old_type = node.type
        other_type = "GRU" if node.type == "LSTM" else "LSTM"
        # Keep dimensions compatible.
        input_size = node.params.get(
            "input_size", node.params.get(
                "in_features", 64))
        hidden_size = node.params.get("hidden_size", 64)
        batch_first = node.params.get("batch_first", True)
        num_layers = node.params.get("num_layers", 1)

        node.type = other_type
        node.params = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "batch_first": batch_first,
        }
        # Rename the node without breaking edges.
        old_id = node.id
        new_id = f"{node.type.lower()}_{old_id.split('_')[-1]}"
        node.id = new_id
        for edge in descriptor.edges:
            if edge.source == old_id:
                edge.source = new_id
            if edge.target == old_id:
                edge.target = new_id
        logger.debug(
            f"Recurrent swapped: {other_type} <- {old_type} (id: {old_id} -> {new_id})")
        if event is not None:
            event.update({
                "op": "swap_recurrent_type",
                "node": new_id,
                "node_type": other_type,
                "before": {"recurrent_type": old_type},
                "after": {"recurrent_type": other_type},
                "detail": f"Recurrent layer swapped: {old_type} → {other_type} (hidden_size={hidden_size})",
            })
        return True

    @staticmethod
    def mutate_attention_heads(descriptor: ArchitectureDescriptor,
                               target_node_id: str, event: Optional[dict] = None) -> bool:
        """Change num_heads in MultiheadAttention while keeping embed_dim divisible."""
        node = next(
            (n for n in descriptor.nodes if n.id == target_node_id),
            None)
        if not node or node.type != "MultiheadAttention":
            return False

        embed_dim = node.params.get("embed_dim", 64)
        current_heads = node.params.get("num_heads", 2)
        valid_heads = [
            h for h in [
                1, 2, 4, 8, 16] if embed_dim %
            h == 0 and h != current_heads]
        if not valid_heads:
            return False

        new_heads = random.choice(valid_heads)
        node.params["num_heads"] = new_heads
        if event is not None:
            event.update({
                "op": "mutate_attention_heads",
                "node": target_node_id,
                "node_type": "MultiheadAttention",
                "before": {"num_heads": current_heads},
                "after": {"num_heads": new_heads},
                "detail": f"Attention '{target_node_id}': num_heads {current_heads} → {new_heads}",
            })
        return True

    @staticmethod
    def add_norm_layer(descriptor: ArchitectureDescriptor,
                       after_node_id: str, event: Optional[dict] = None) -> bool:
        """Add a normalization layer after a node."""
        after_node = next(
            (n for n in descriptor.nodes if n.id == after_node_id), None)
        if not after_node:
            return False

        out_dim = MutationGrammar._get_node_output_dim(
            descriptor, after_node_id)
        if out_dim is None or out_dim <= 1:
            return False

        outgoing_edges = [
            e for e in descriptor.edges if e.source == after_node_id]
        if not outgoing_edges:
            return False

        if any(e.target == "output" for e in outgoing_edges):
            return False  # Don't insert norms right before output

        norm_type = random.choice(MutationGrammar.NORM_TYPES)
        new_node_id = f"{norm_type.lower()}_add_{random.randint(1000, 9999)}"

        if norm_type in ["BatchNorm1d", "BatchNorm2d"]:
            new_params = {"num_features": out_dim}
        elif norm_type == "LayerNorm":
            new_params = {"normalized_shape": out_dim}
        else:
            return False

        descriptor.nodes.append(
            Node(
                id=new_node_id,
                type=norm_type,
                params=new_params))

        original_targets = [edge.target for edge in outgoing_edges]
        descriptor.edges = [
            edge for edge in descriptor.edges if edge not in outgoing_edges]
        descriptor.edges.append(Edge(source=after_node_id, target=new_node_id))
        for original_target in original_targets:
            descriptor.edges.append(
                Edge(
                    source=new_node_id,
                    target=original_target))

        descriptor.normalize_inplace()
        if event is not None:
            event.update({
                "op": "add_norm_layer",
                "node": new_node_id,
                "node_type": norm_type,
                "before": {},
                "after": {"inserted_after": after_node_id, "params": new_params},
                "detail": f"Inserted {norm_type} after '{after_node_id}'",
            })
        return True

    @staticmethod
    def add_layer(descriptor: ArchitectureDescriptor, after_node_id: str,
                  new_node_type: str, event: Optional[dict] = None) -> bool:
        """Insert a new node immediately after after_node_id, rewiring the edge."""
        after_node = next(
            (n for n in descriptor.nodes if n.id == after_node_id), None)
        if not after_node:
            return False

        if new_node_type == after_node.type and new_node_type not in [
                "Linear", "Identity"]:
            return False

        outgoing_edges = [
            e for e in descriptor.edges if e.source == after_node_id]
        if not outgoing_edges:
            return False

        new_node_id = f"{
            new_node_type.lower()}_add_{
            random.randint(
                1000,
                9999)}"

        out_dim = MutationGrammar._get_node_output_dim(
            descriptor, after_node_id)
        if out_dim is None or out_dim <= 0:
            out_dim = 64

        if new_node_type == "Linear":
            out_features = max(
                1, int(out_dim * random.choice([0.5, 1.0, 1.5, 2.0])))
            new_params = {"in_features": out_dim, "out_features": out_features}
        elif new_node_type == "Dropout":
            new_params = {"p": random.choice([0.1, 0.2, 0.3, 0.5])}
        elif new_node_type == "LayerNorm":
            new_params = {"normalized_shape": out_dim}
        elif new_node_type in ["BatchNorm1d", "BatchNorm2d"]:
            new_params = {"num_features": out_dim}
        elif new_node_type == "Identity":
            new_params = {}
        elif new_node_type in MutationGrammar.ACTIVATION_TYPES:
            new_params = {}
        else:
            return False

        descriptor.nodes.append(
            Node(
                id=new_node_id,
                type=new_node_type,
                params=new_params))

        original_targets = [edge.target for edge in outgoing_edges]
        descriptor.edges = [
            edge for edge in descriptor.edges if edge not in outgoing_edges]

        descriptor.edges.append(Edge(source=after_node_id, target=new_node_id))
        for original_target in original_targets:
            descriptor.edges.append(
                Edge(
                    source=new_node_id,
                    target=original_target))

        descriptor.normalize_inplace()
        if event is not None:
            event.update({
                "op": "add_layer",
                "node": new_node_id,
                "node_type": new_node_type,
                "before": {},
                "after": {"inserted_after": after_node_id, "params": new_params},
                "detail": f"Inserted {new_node_type} after '{after_node_id}'",
            })
        return True

    @staticmethod
    def remove_layer(descriptor: ArchitectureDescriptor, target_node_id: str,
                     protected_types: Optional[Set[str]] = None, event: Optional[dict] = None) -> bool:
        """Remove a node and reconnect its parents directly to its children.

        Protected types (LSTM, GRU, MultiheadAttention) cannot be removed
        to prevent the algorithm from stripping useful backbone structure.
        """
        node = next(
            (n for n in descriptor.nodes if n.id == target_node_id),
            None)
        if not node:
            return False

        if node.id in ["input", "output"]:
            return False

        if protected_types is None:
            protected_types = {"LSTM", "GRU", "MultiheadAttention"}
        if node.type in protected_types:
            return False

        incoming = [e for e in descriptor.edges if e.target == target_node_id]
        outgoing = [e for e in descriptor.edges if e.source == target_node_id]

        if not incoming or not outgoing:
            return False

        if len(incoming) == 1 and node.type not in ["Add", "Concat"]:
            parent = incoming[0].source
            children = [e.target for e in outgoing]
            removed_type = node.type

            descriptor.nodes = [
                n for n in descriptor.nodes if n.id != target_node_id]
            descriptor.edges = [e for e in descriptor.edges if e.source !=
                                target_node_id and e.target != target_node_id]

            for child in children:
                descriptor.edges.append(Edge(source=parent, target=child))

            descriptor.normalize_inplace()
            if event is not None:
                event.update({
                    "op": "remove_layer",
                    "node": target_node_id,
                    "node_type": removed_type,
                    "before": {"params": node.params},
                    "after": {"reconnected": f"{parent} → {', '.join(children)}"},
                    "detail": f"Removed {removed_type} '{target_node_id}' ({parent} reconnected to {', '.join(children)})",
                })
            return True

        return False

    @staticmethod
    def add_skip_connection(
        descriptor: ArchitectureDescriptor,
        from_id: str,
        to_id: str,
        event: Optional[dict] = None,
    ) -> bool:
        """Add a skip connection from from_id -> to_id using Concat merge."""
        node_ids = {n.id for n in descriptor.nodes}
        if from_id not in node_ids or to_id not in node_ids:
            return False
        if from_id == to_id:
            return False

        levels = MutationGrammar._topological_levels(descriptor)
        if from_id not in levels or to_id not in levels:
            return False
        if levels[to_id] - levels[from_id] < 2:
            return False
        if any(edge.source == from_id and edge.target ==
               to_id for edge in descriptor.edges):
            return False

        def reachable(start: str, goal: str) -> bool:
            visited: Set[str] = set()
            stack = [start]
            while stack:
                curr = stack.pop()
                if curr == goal:
                    return True
                if curr in visited:
                    continue
                visited.add(curr)
                for edge in descriptor.edges:
                    if edge.source == curr:
                        stack.append(edge.target)
            return False

        if reachable(to_id, from_id):
            return False

        incoming_edges = [
            edge for edge in descriptor.edges if edge.target == to_id]
        if not incoming_edges:
            return False

        # Skip duplicate residual connections.
        existing_skips = [
            e for e in descriptor.edges if e.target == to_id and e.source != max(
                incoming_edges, key=lambda e: levels.get(
                    e.source, -1)).source]
        if len(existing_skips) >= 2:
            return False  # Already has skip connections, don't create a messy merge

        concat_id = f"concat_{random.randint(1000, 9999)}"

        num_inputs = len(incoming_edges) + 1
        descriptor.nodes.append(Node(
            id=concat_id,
            type="Concat",
            params={"dim": -1},
            io={"inputs": [f"input_{i}" for i in range(num_inputs)], "outputs": [
                "output"]}
        ))

        main_edge = max(incoming_edges, key=lambda e: levels.get(e.source, -1))
        main_edge.target = concat_id

        descriptor.edges.append(Edge(source=from_id, target=concat_id))
        descriptor.edges.append(Edge(source=concat_id, target=to_id))

        MutationGrammar._deduplicate_edges(descriptor)
        descriptor.normalize_inplace()
        if event is not None:
            event.update({
                "op": "add_skip_connection",
                "node": concat_id,
                "node_type": "Concat",
                "before": {},
                "after": {"from": from_id, "to": to_id},
                "detail": f"Added skip connection: '{from_id}' → '{to_id}' (via {concat_id})",
            })
        return True

    @staticmethod
    def _deduplicate_edges(descriptor: ArchitectureDescriptor) -> None:
        """Remove duplicate edges (same source, same target)."""
        seen = set()
        unique_edges = []
        for edge in descriptor.edges:
            key = (edge.source, edge.target)
            if key not in seen:
                seen.add(key)
                unique_edges.append(edge)
        descriptor.edges = unique_edges

    @staticmethod
    def _topological_levels(
            descriptor: ArchitectureDescriptor) -> Dict[str, int]:
        indegree = {node.id: 0 for node in descriptor.nodes}
        indegree["input"] = 0
        levels = {"input": 0}

        adjacency = {node.id: [] for node in descriptor.nodes}
        adjacency["input"] = []

        for edge in descriptor.edges:
            if edge.target in indegree:
                indegree[edge.target] += 1
            if edge.source in adjacency:
                adjacency[edge.source].append(edge.target)

        queue = ["input"]
        while queue:
            current = queue.pop(0)
            for neighbor in adjacency.get(current, []):
                if neighbor == "output":
                    continue
                levels[neighbor] = max(
                    levels.get(
                        neighbor, 0), levels.get(
                        current, 0) + 1)
                if neighbor in indegree:
                    indegree[neighbor] -= 1
                    if indegree[neighbor] == 0:
                        queue.append(neighbor)

        return levels

    @staticmethod
    def _graph_depth(descriptor: ArchitectureDescriptor) -> int:
        levels = MutationGrammar._topological_levels(descriptor)
        return max(levels.values(), default=0)

    @staticmethod
    def _within_limits(descriptor: ArchitectureDescriptor) -> bool:
        return len(descriptor.nodes) <= MutationGrammar.MAX_NODES and MutationGrammar._graph_depth(
            descriptor) <= MutationGrammar.MAX_DEPTH

    @staticmethod
    def apply_random_mutation(
        descriptor: ArchitectureDescriptor,
        stats: Optional[GenerationStats] = None,
        event_log: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Apply a single random mutation with proper rollback on failure.

        On success, a structured event describing exactly what changed is
        appended to `event_log` (tagged with `context`: generation, ids).
        """
        original_nodes = copy.deepcopy(descriptor.nodes)
        original_edges = copy.deepcopy(descriptor.edges)

        candidates = []
        width_nodes = [
            n.id for n in descriptor.nodes if n.type in [
                "Linear", "Conv1d", "LSTM", "GRU"]]
        layer_nodes = [
            n.id for n in descriptor.nodes if n.type in [
                "Linear", "Conv1d"]]
        activation_nodes = [
            n.id for n in descriptor.nodes if n.type in MutationGrammar.ACTIVATION_TYPES]
        dropout_nodes = [n.id for n in descriptor.nodes if n.type == "Dropout"]
        recurrent_nodes = [
            n.id for n in descriptor.nodes if n.type in MutationGrammar.RECURRENT_TYPES]
        attention_nodes = [
            n.id for n in descriptor.nodes if n.type == "MultiheadAttention"]

        protected_types = {"LSTM", "GRU", "MultiheadAttention"}
        removable = [n.id for n in descriptor.nodes if n.type not in ["Add", "Concat"]
                     and n.type not in protected_types
                     and n.id not in ["input", "output"]
                     and any(e.target == n.id for e in descriptor.edges)
                     and any(e.source == n.id for e in descriptor.edges)]

        if width_nodes:
            def _scale(desc=descriptor, nodes=list(width_nodes), event=None):
                return MutationGrammar.scale_width(
                    desc,
                    random.choice(nodes),
                    random.choice(MutationGrammar.WIDTH_FACTORS),
                    event=event,
                )
            candidates.append(("width", _scale))

        if layer_nodes:
            def _add_layer(desc=descriptor, nodes=list(
                    layer_nodes), event=None):
                return MutationGrammar.add_layer(
                    desc,
                    random.choice(nodes),
                    random.choice(["ReLU", "Dropout", "Linear", "LayerNorm"]),
                    event=event,
                )
            candidates.append(("structural", _add_layer))

        if activation_nodes:
            def _mutate_act(desc=descriptor, nodes=list(
                    activation_nodes), event=None):
                return MutationGrammar.mutate_activation(
                    desc, random.choice(nodes), event=event)
            candidates.append(("activation", _mutate_act))

        if dropout_nodes:
            def _mutate_dropout(desc=descriptor, nodes=list(
                    dropout_nodes), event=None):
                return MutationGrammar.mutate_dropout_rate(
                    desc, random.choice(nodes), event=event)
            candidates.append(("hyperparam", _mutate_dropout))

        if recurrent_nodes:
            def _swap_recurrent(desc=descriptor, nodes=list(
                    recurrent_nodes), event=None):
                return MutationGrammar.swap_recurrent_type(
                    desc, random.choice(nodes), event=event)
            candidates.append(("layer_swap", _swap_recurrent))

        if attention_nodes:
            def _mutate_heads(desc=descriptor, nodes=list(
                    attention_nodes), event=None):
                return MutationGrammar.mutate_attention_heads(
                    desc, random.choice(nodes), event=event)
            candidates.append(("hyperparam", _mutate_heads))

        if layer_nodes:
            def _add_norm(desc=descriptor, nodes=list(
                    layer_nodes), event=None):
                return MutationGrammar.add_norm_layer(
                    desc, random.choice(nodes), event=event)
            candidates.append(("structural", _add_norm))

        if removable:
            def _remove(desc=descriptor, nodes=list(removable), event=None):
                return MutationGrammar.remove_layer(
                    desc, random.choice(nodes), event=event)
            candidates.append(("structural", _remove))

        if len(width_nodes) >= 2:
            valid_pairs = []
            levels = MutationGrammar._topological_levels(descriptor)
            for from_id in width_nodes:
                for to_id in width_nodes:
                    if from_id == to_id:
                        continue
                    if from_id not in levels or to_id not in levels:
                        continue
                    if levels[to_id] - levels[from_id] < 2:
                        continue
                    if any(edge.source == from_id and edge.target ==
                           to_id for edge in descriptor.edges):
                        continue
                    valid_pairs.append((from_id, to_id))

            if valid_pairs:
                def _skip(desc=descriptor, pairs=list(
                        valid_pairs), event=None):
                    from_id, to_id = random.choice(pairs)
                    return MutationGrammar.add_skip_connection(
                        desc, from_id, to_id, event=event)
                candidates.append(("structural", _skip))

        random.shuffle(candidates)

        if stats:
            stats.mutation_attempts += len(candidates)

        for mut_type, mutator in candidates:
            event: Dict[str, Any] = {}
            try:
                if mutator(event=event):
                    descriptor.normalize_inplace()
                    if not MutationGrammar._within_limits(descriptor):
                        logger.debug(
                            f"Mutation {mut_type} failed: exceeded node/depth limits")
                        descriptor.nodes = copy.deepcopy(original_nodes)
                        descriptor.edges = copy.deepcopy(original_edges)
                        continue
                    descriptor.validate()
                    logger.debug(f"Mutation succeeded: {mut_type}")
                    if stats:
                        stats.mutation_successes += 1
                        if mut_type == "structural":
                            stats.structural_mutations += 1
                        elif mut_type == "width":
                            stats.width_mutations += 1
                        elif mut_type == "activation":
                            stats.activation_mutations += 1
                        elif mut_type == "hyperparam":
                            stats.hyperparam_mutations += 1
                        elif mut_type == "layer_swap":
                            stats.layer_swap_mutations += 1
                    if event_log is not None and event:
                        event["category"] = mut_type
                        if context:
                            event.update(context)
                        event_log.append(event)
                    return True
            except Exception as e:
                logger.debug(
                    f"Mutation {mut_type} failed: {
                        type(e).__name__}: {e}")
                descriptor.nodes = copy.deepcopy(original_nodes)
                descriptor.edges = copy.deepcopy(original_edges)
                continue

        descriptor.nodes = copy.deepcopy(original_nodes)
        descriptor.edges = copy.deepcopy(original_edges)
        return False

    @staticmethod
    def apply_random_mutations(
        descriptor: ArchitectureDescriptor,
        num_mutations: int = 3,
        stats: Optional[GenerationStats] = None,
        event_log: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Apply multiple random mutations. Returns number successfully applied."""
        applied = 0
        max_attempts = num_mutations * 5
        attempts = 0

        while applied < num_mutations and attempts < max_attempts:
            attempts += 1
            if MutationGrammar.apply_random_mutation(
                    descriptor, stats, event_log=event_log, context=context):
                applied += 1

        return applied


class TieredEvaluator:
    TIER1_EPOCHS = 3
    TIER2_EPOCHS = 5
    FINALS_EPOCHS = 15
    EARLY_STOPPING_PATIENCE = 2

    @staticmethod
    def evaluate(
        model: nn.Module,
        data_loader,
        criterion,
        optimizer,
        num_epochs: int,
        device: str = "cpu",
        val_loader=None,
        max_allowed_loss: float = float("inf"),
    ) -> Tuple[float, List[float], List[float]]:
        """Train model and return (final_train_loss, train_loss_history, val_loss_history)."""
        model.to(device)
        model.train()
        losses = []
        val_losses = []
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            batches = 0
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                outputs, batch_y = TieredEvaluator._align_shapes(
                    outputs, batch_y, criterion)
                loss = criterion(outputs, batch_y)

                if torch.isnan(loss) or torch.isinf(loss):
                    return float("inf"), losses, val_losses

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                batches += 1

            avg_loss = epoch_loss / max(batches, 1)
            losses.append(avg_loss)

            # Prune weak candidates after the first epoch.
            if num_epochs <= TieredEvaluator.TIER1_EPOCHS and epoch == 0 and max_allowed_loss < float("inf"):
                if avg_loss > max_allowed_loss:
                    return avg_loss, losses, val_losses

            should_validate = val_loader is not None and (num_epochs > TieredEvaluator.TIER1_EPOCHS or epoch == num_epochs - 1)
            if should_validate:
                val_loss = TieredEvaluator.validate(
                    model, val_loader, criterion, device)
                val_losses.append(val_loss)
                model.train()
                if epoch >= 1:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= TieredEvaluator.EARLY_STOPPING_PATIENCE:
                            break

        return (losses[-1] if losses else float("inf")), losses, val_losses

    @staticmethod
    def validate(
        model: nn.Module,
        data_loader,
        criterion,
        device: str = "cpu",
    ) -> float:
        model.to(device)
        model.eval()
        total_loss = 0.0
        batches = 0

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                outputs, batch_y = TieredEvaluator._align_shapes(
                    outputs, batch_y, criterion)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                batches += 1

        return total_loss / max(batches, 1)

    @staticmethod
    def _align_shapes(outputs: torch.Tensor, targets: torch.Tensor,
                      criterion: Optional[nn.Module] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if outputs.dim() == 3 and targets.dim() == 2:
            outputs = outputs[:, -1, :]
        if outputs.dim() > targets.dim() and targets.dim() == 1:
            outputs = outputs.squeeze(-1)
        elif outputs.dim() == 1 and targets.dim() == 2 and targets.shape[1] == 1:
            outputs = outputs.unsqueeze(-1)

        if isinstance(criterion, nn.CrossEntropyLoss):
            if targets.dim() > 1:
                targets = targets.squeeze(-1)
            targets = targets.long()
            if outputs.dim() > 2:
                outputs = outputs.reshape(outputs.shape[0], -1)
        elif isinstance(criterion, nn.BCEWithLogitsLoss):
            targets = targets.float()
        return outputs, targets

    @staticmethod
    def select_criterion(problem_type: str) -> nn.Module:
        if problem_type == "classification":
            return nn.CrossEntropyLoss()
        return nn.MSELoss()


class NeuroevolutionEngine:
    def __init__(self, initial_descriptor: ArchitectureDescriptor,
                 population_size: int = 30, statusCallback: Optional[callable] = None):
        self.initial_descriptor = initial_descriptor
        self.population_size = population_size
        self.statusCallback = statusCallback
        self.population: List[Tuple[ArchitectureDescriptor,
                                    Optional[Dict[str, torch.Tensor]]]] = []
        self._stagnation_counter = 0
        self._best_score_history: List[float] = []
        self.generation_stats: List[GenerationStats] = []
        self.original_descriptor_json: str = initial_descriptor.to_json()
        self._best_model_state: Optional[Dict[str, torch.Tensor]] = None
        self._best_descriptor: Optional[ArchitectureDescriptor] = None
        self._mutation_events: List[Dict[str, Any]] = []
        self._ancestry: Dict[int, Dict[str, Any]] = {}
        self._next_individual_id: int = 0
        self._root_desc: Optional[ArchitectureDescriptor] = None
        self._champion_training: Dict[str, Any] = {}
        self._start_time: Optional[float] = None
        self._active_callback: Optional[callable] = None


    @staticmethod
    def _get_meta(desc: ArchitectureDescriptor) -> Dict[str, Any]:
        meta = getattr(desc, "_evo_meta", None)
        return meta if isinstance(meta, dict) else {}

    @staticmethod
    def _set_meta(desc: ArchitectureDescriptor, meta: Dict[str, Any]) -> None:
        # Lineage metadata is optional.
        try:
            setattr(desc, "_evo_meta", meta)
        except Exception:
            pass

    def _mutate_and_track(self, child_desc: ArchitectureDescriptor, parent_desc: Optional[ArchitectureDescriptor],
                          generation: int, num_mutations: int, stats: Optional[GenerationStats] = None) -> int:
        parent_id = self._get_meta(parent_desc).get(
            "id") if parent_desc is not None else None
        self._next_individual_id += 1
        cid = self._next_individual_id
        self._set_meta(
            child_desc, {
                "id": cid, "parent": parent_id, "generation": generation})

        before = len(self._mutation_events)
        applied = MutationGrammar.apply_random_mutations(
            child_desc,
            num_mutations=num_mutations,
            stats=stats,
            event_log=self._mutation_events,
            context={
                "generation": generation,
                "individual_id": cid,
                "parent_id": parent_id},
        )
        self._ancestry[cid] = {
            "parent": parent_id,
            "generation": generation,
            "mutations": [e.get("detail", e.get("op", "mutation")) for e in self._mutation_events[before:]],
        }
        return applied

    def _champion_lineage(self) -> List[Dict[str, Any]]:
        """Walk ancestry from the champion back to the root, chronological order."""
        champ_id = self._get_meta(self._best_descriptor).get(
            "id") if self._best_descriptor is not None else None
        chain: List[Dict[str, Any]] = []
        visited = set()
        cid = champ_id
        while cid is not None and cid not in visited and cid in self._ancestry:
            visited.add(cid)
            entry = self._ancestry[cid]
            chain.append({
                "individual_id": cid,
                "generation": entry.get("generation"),
                "mutations": entry.get("mutations", []),
            })
            cid = entry.get("parent")
        chain.reverse()
        return chain

    def _emit(self, payload: Dict[str, Any]) -> None:
        """Fire a structured status event; a broken callback must never kill evolution."""
        cb = self._active_callback
        if cb is None:
            return
        try:
            cb(payload)
        except Exception:
            pass

    @staticmethod
    def _sanitize_history(hist: List[float]) -> List[Any]:
        out = []
        for v in hist:
            if isinstance(v, float):
                if math.isinf(v) or math.isnan(v):
                    out.append(_sanitize_float(v))
                else:
                    out.append(round(v, 6))
            else:
                out.append(v)
        return out


    def _collect_model_metrics(
        self,
        model: nn.Module,
        descriptor: ArchitectureDescriptor,
        data_loader,
        criterion,
        device: str,
        problem_type: str,
        max_samples: int = 256,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        try:
            result["param_count"] = int(self._model_complexity(model))
            size_bytes = sum(int(t.numel()) * int(t.element_size())
                             for t in model.state_dict().values())
            result["size_mb"] = round(size_bytes / (1024 * 1024), 4)
            result["node_count"] = len(descriptor.nodes)
            result["depth"] = MutationGrammar._graph_depth(descriptor)
        except Exception as e:
            logger.debug(f"Metric collection (structure) failed: {e}")

        try:
            model.to(device)
            model.eval()
            first_batch = None
            for batch_x, batch_y in data_loader:
                first_batch = (batch_x, batch_y)
                break
            if first_batch is not None:
                bx = first_batch[0].to(device)
                is_cuda = "cuda" in str(device)
                with torch.no_grad():
                    for _ in range(3):
                        model(bx)
                    if is_cuda:
                        torch.cuda.synchronize()
                    reps = 10
                    t0 = time.perf_counter()
                    for _ in range(reps):
                        model(bx)
                    if is_cuda:
                        torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - t0) / reps
                result["latency_ms_per_batch"] = round(elapsed * 1000, 3)
                batch_n = int(bx.shape[0]) if bx.dim() > 0 else 1
                result["latency_ms_per_sample"] = round(
                    elapsed * 1000 / max(batch_n, 1), 4)
        except Exception as e:
            logger.debug(f"Metric collection (latency) failed: {e}")

        try:
            val_loss = TieredEvaluator.validate(
                model, data_loader, criterion, device)
            result["val_loss"] = _sanitize_float(
                round(
                    val_loss,
                    6)) if isinstance(
                val_loss,
                float) else val_loss
        except Exception as e:
            logger.debug(f"Metric collection (val loss) failed: {e}")

        try:
            model.eval()
            y_true: List[Any] = []
            y_pred: List[Any] = []
            with torch.no_grad():
                for batch_x, batch_y in data_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    outputs, targets = TieredEvaluator._align_shapes(
                        outputs, batch_y, criterion)
                    if problem_type == "classification":
                        preds = outputs.argmax(dim=-1).reshape(-1).long()
                        targs = targets.reshape(-1).long()
                        y_pred.extend(int(v)
                                      for v in preds.detach().cpu().tolist())
                        y_true.extend(int(v)
                                      for v in targs.detach().cpu().tolist())
                    else:
                        preds = outputs.reshape(-1).float()
                        targs = targets.reshape(-1).float()
                        y_pred.extend(round(float(v), 6)
                                      for v in preds.detach().cpu().tolist())
                        y_true.extend(round(float(v), 6)
                                      for v in targs.detach().cpu().tolist())
                    if len(y_true) >= max_samples:
                        break

            y_true = y_true[:max_samples]
            y_pred = y_pred[:max_samples]
            n = len(y_true)

            if n > 0:
                if problem_type == "classification":
                    labels = sorted(set(y_true) | set(y_pred))
                    index = {label: i for i, label in enumerate(labels)}
                    matrix = [[0] * len(labels) for _ in labels]
                    correct = 0
                    for t, p in zip(y_true, y_pred):
                        matrix[index[t]][index[p]] += 1
                        if t == p:
                            correct += 1
                    result["accuracy"] = round(correct / n, 4)
                    result["confusion_matrix"] = {"labels": [
                        str(l) for l in labels], "matrix": matrix}
                    result["prediction_samples"] = {
                        "y_true": y_true, "y_pred": y_pred}
                else:
                    mse = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n
                    mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n
                    mean_t = sum(y_true) / n
                    ss_tot = sum((t - mean_t) ** 2 for t in y_true)
                    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
                    result["mse"] = round(mse, 6)
                    result["rmse"] = round(math.sqrt(mse), 6)
                    result["mae"] = round(mae, 6)
                    result["r2"] = round(
                        1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else None
                    result["prediction_samples"] = {
                        "y_true": y_true, "y_pred": y_pred}
        except Exception as e:
            logger.debug(f"Metric collection (predictions) failed: {e}")

        return result

    @staticmethod
    def _build_improvement(
            baseline: Optional[Dict[str, Any]], champion: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        baseline = baseline or {}
        champion = champion or {}

        def reduction_pct(before, after):
            if isinstance(before, (int, float)) and isinstance(after, (int, float)) \
                    and not isinstance(before, bool) and before != 0:
                return round((before - after) / abs(before) * 100.0, 2)
            return None

        improvement: Dict[str, Any] = {
            "val_loss_before": baseline.get("val_loss"),
            "val_loss_after": champion.get("val_loss"),
            "val_loss_change_pct": reduction_pct(baseline.get("val_loss"), champion.get("val_loss")),
            "param_count_before": baseline.get("param_count"),
            "param_count_after": champion.get("param_count"),
            "param_count_change_pct": reduction_pct(baseline.get("param_count"), champion.get("param_count")),
            "size_mb_before": baseline.get("size_mb"),
            "size_mb_after": champion.get("size_mb"),
            "latency_ms_per_sample_before": baseline.get("latency_ms_per_sample"),
            "latency_ms_per_sample_after": champion.get("latency_ms_per_sample"),
            "latency_change_pct": reduction_pct(baseline.get("latency_ms_per_sample"), champion.get("latency_ms_per_sample")),
        }

        if baseline.get("accuracy") is not None and champion.get(
                "accuracy") is not None:
            improvement["accuracy_before"] = baseline.get("accuracy")
            improvement["accuracy_after"] = champion.get("accuracy")
            improvement["accuracy_change_pp"] = round(
                (champion["accuracy"] - baseline["accuracy"]) * 100.0, 2)

        if baseline.get("r2") is not None and champion.get("r2") is not None:
            improvement["r2_before"] = baseline.get("r2")
            improvement["r2_after"] = champion.get("r2")

        v = improvement["val_loss_change_pct"]
        if v is None:
            verdict = "unknown"
        elif v > 1:
            verdict = "improved"
        elif v < -1:
            verdict = "regressed"
        else:
            verdict = "unchanged"
        improvement["verdict"] = verdict
        return improvement

    def initialize_population(
        self,
        parent_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self._mutation_events = []
        self._ancestry = {}
        self._next_individual_id = 0

        # Keep an untouched lineage root.
        original = copy.deepcopy(self.initial_descriptor)
        self._set_meta(original, {"id": 0, "parent": None, "generation": 1})
        self._ancestry[0] = {"parent": None, "generation": 1, "mutations": []}
        self._root_desc = original
        self.population = [(original, parent_state_dict)]

        light_mutant = copy.deepcopy(self.initial_descriptor)
        self._mutate_and_track(
            light_mutant,
            original,
            generation=1,
            num_mutations=random.randint(
                1,
                2))
        self.population.append((light_mutant, parent_state_dict))

        attempts = 0
        max_attempts = self.population_size * 10

        while len(
                self.population) < self.population_size and attempts < max_attempts:
            attempts += 1
            child = copy.deepcopy(self.initial_descriptor)
            if self._mutate_and_track(
                    child, original, generation=1, num_mutations=random.randint(2, 4)) > 0:
                self.population.append((child, parent_state_dict))

        while len(self.population) < self.population_size:
            child = copy.deepcopy(self.initial_descriptor)
            self._mutate_and_track(
                child,
                original,
                generation=1,
                num_mutations=random.randint(
                    1,
                    3))
            self.population.append((child, parent_state_dict))

    @staticmethod
    def _build_model_with_weights(
        descriptor: ArchitectureDescriptor,
        parent_state: Optional[Dict[str, torch.Tensor]],
    ) -> nn.Module:
        model = DescriptorModelBuilder.build(descriptor)
        if parent_state:
            WeightCompatibilityEngine.transfer_weights(parent_state, model)
        return model

    @staticmethod
    def _model_complexity(model: nn.Module) -> int:
        return sum(parameter.numel()
                   for parameter in model.parameters() if parameter.requires_grad)

    @staticmethod
    def _score(loss: float, model: nn.Module, complexity_penalty: float,
               descriptor: ArchitectureDescriptor = None) -> float:
        """Score with adaptive complexity penalty and structural preservation bonus."""
        raw_penalty = complexity_penalty * \
            NeuroevolutionEngine._model_complexity(model)

        adaptive_penalty = raw_penalty * max(1.0, abs(loss) * 10.0)

        score = loss + adaptive_penalty
        if descriptor is not None:
            has_recurrent = any(n.type in ["LSTM", "GRU"]
                                for n in descriptor.nodes)
            has_attention = any(
                n.type == "MultiheadAttention" for n in descriptor.nodes)
            has_activation = any(
                n.type in MutationGrammar.ACTIVATION_TYPES for n in descriptor.nodes)
            has_norm = any(
                n.type in MutationGrammar.NORM_TYPES for n in descriptor.nodes)

            structural_bonus = 0.0
            if has_recurrent:
                structural_bonus -= 0.005
            if has_attention:
                structural_bonus -= 0.005
            if has_activation:
                structural_bonus -= 0.002
            if has_norm:
                structural_bonus -= 0.002

            score += structural_bonus

        return score

    def _is_stagnant(self, window: int = 5,
                     relative_threshold: float = 0.02) -> bool:
        """Detect stagnation using relative change threshold."""
        effective_window = min(
            window, max(
                1, len(
                    self._best_score_history) - 1))
        if effective_window < 2:
            return False
        recent = self._best_score_history[-effective_window:]
        for i in range(1, len(recent)):
            if abs(recent[i] - recent[i - 1]) / \
                    max(abs(recent[i - 1]), 1e-8) > relative_threshold:
                return False
        return True

    @staticmethod
    def _get_species(descriptor: ArchitectureDescriptor) -> Tuple:
        """Classify architecture by structural features, not just parameter count."""
        depth = MutationGrammar._graph_depth(descriptor)
        has_lstm = any(n.type == "LSTM" for n in descriptor.nodes)
        has_attention = any(
            n.type == "MultiheadAttention" for n in descriptor.nodes)
        has_conv = any("Conv" in n.type for n in descriptor.nodes)
        skip_count = sum(
            1 for n in descriptor.nodes if n.type in [
                "Add", "Concat"])
        return (depth, has_lstm, has_attention, has_conv, min(skip_count, 3))

    @staticmethod
    def _has_nan_weights(model: nn.Module) -> bool:
        """Check if any model parameter contains NaN or Inf."""
        for param in model.parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                return True
        return False

    def evolve(
        self,
        train_loader,
        val_loader=None,
        generations: int = 5,
        max_epochs: int = 5,
        device: str = "cpu",
        parent_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        criterion: Optional[nn.Module] = None,
        problem_type: str = "regression",
        complexity_penalty: float = 1e-7,
        statusCallback: Optional[callable] = None,
    ) -> Tuple[ArchitectureDescriptor, nn.Module, Dict[str, Any]]:
        """
        Run tiered neuroevolution and return (best_descriptor, best_model, diagnostics).
        """
        statusCallback = statusCallback or self.statusCallback or (
            lambda *_args, **_kwargs: None)
        self._active_callback = statusCallback
        self._start_time = time.time()
        self._champion_training = {}
        self.initialize_population(parent_state_dict)

        if criterion is None:
            criterion = TieredEvaluator.select_criterion(problem_type)

        eval_loader = val_loader if val_loader is not None else train_loader
        if val_loader is None:
            logger.warning(
                "No validation loader provided. Using training data for evaluation (may overfit).")

        def _progress(gen_index: int) -> float:
            return 40 + (50 / max(generations, 1)) * gen_index

        self._emit({
            "type": "phase",
            "phase": "evolution_start",
            "status": f"Evolving population of {len(self.population)} architectures over {generations} generations...",
            "generations": generations,
            "population_size": len(self.population),
            "progress": 41,
        })

        best_descriptor = copy.deepcopy(self.initial_descriptor)
        self._set_meta(
            best_descriptor, {
                "id": 0, "parent": None, "generation": 1})
        best_model = self._build_model_with_weights(
            best_descriptor, parent_state_dict)
        best_score = float("inf")
        best_train_loss = float("inf")
        best_val_loss = float("inf")

        self._best_model_state = parent_state_dict
        self._best_descriptor = copy.deepcopy(best_descriptor)

        for gen in range(generations):
            stats = GenerationStats(generation=gen + 1)

            self._emit({
                "type": "generation",
                "phase": "start",
                "generation": gen + 1,
                "generations": generations,
                "population": len(self.population),
                "status": f"Generation {gen + 1}/{generations}: screening {len(self.population)} candidates...",
                "progress": _progress(gen),
            })

            tier1_results = []

            def _eval_single_candidate(item):
                desc, parent_state = item
                try:
                    if str(device).lower() == "cpu":
                        torch.set_num_threads(1)
                    desc.validate()
                    model = self._build_model_with_weights(desc, parent_state)
                    model.to(device)

                    lr = self._suggest_lr(desc)
                    optimizer = torch.optim.Adam(
                        model.parameters(), lr=lr, weight_decay=1e-5)

                    prune_threshold = float("inf")
                    if not math.isinf(best_train_loss):
                        prune_threshold = max(best_train_loss * 4.0, 10.0)

                    train_loss, _, val_history = TieredEvaluator.evaluate(
                        model,
                        train_loader,
                        criterion,
                        optimizer,
                        num_epochs=TieredEvaluator.TIER1_EPOCHS,
                        device=device,
                        val_loader=val_loader,
                        max_allowed_loss=prune_threshold,
                    )

                    if train_loss == float("inf") or math.isnan(train_loss):
                        return None, "nan"

                    if self._has_nan_weights(model):
                        return None, "nan"

                    score_loss = val_history[-1] if (val_loader is not None and val_history) else train_loss
                    score = self._score(
                        score_loss, model, complexity_penalty, desc)
                    trained_state = model.state_dict()
                    return (desc, trained_state, train_loss, score, model, []), None
                except RuntimeError as e:
                    err_type = "shape" if ("shape" in str(e).lower() or "size" in str(e).lower()) else "other"
                    logger.debug(f"Tier 1 eval failed: {type(e).__name__}: {e}")
                    return None, err_type
                except Exception as e:
                    logger.debug(f"Tier 1 eval failed: {type(e).__name__}: {e}")
                    return None, "other"

            import concurrent.futures
            max_workers = 1
            if str(device).lower() == "cpu":
                max_workers = min(4, max(1, (os.cpu_count() or 4) // 2))

            if max_workers > 1 and len(self.population) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    eval_out = list(executor.map(_eval_single_candidate, self.population))
            else:
                eval_out = [_eval_single_candidate(item) for item in self.population]

            for res, err_type in eval_out:
                if res is not None:
                    tier1_results.append(res)
                elif err_type == "nan":
                    stats.nan_errors += 1
                elif err_type == "shape":
                    stats.shape_errors += 1
                    stats.tier1_failures += 1
                elif err_type == "other":
                    stats.other_errors += 1
                    stats.tier1_failures += 1

            if not tier1_results:
                logger.warning(f"Generation {gen + 1}: No survivors in Tier 1")
                self.generation_stats.append(stats)
                continue

            tier1_results.sort(key=lambda item: item[3])
            stats.tier1_survivors = len(tier1_results)
            stats.avg_tier1_score = sum(
                item[3] for item in tier1_results) / len(tier1_results)
            stats.best_train_loss = tier1_results[0][2]
            stats.best_score = tier1_results[0][3]

            self._emit({
                "type": "tier",
                "tier": 1,
                "generation": gen + 1,
                "survivors": stats.tier1_survivors,
                "failures": stats.tier1_failures,
                "status": f"Gen {gen + 1}: {stats.tier1_survivors}/{len(self.population)} passed screening",
                "progress": _progress(gen) + (50 / max(generations, 1)) * 0.33,
            })

            tier2_count = max(3, self.population_size // 3)

            def _eval_tier2_candidate(item):
                desc, trained_state, _, _, _, _ = item
                try:
                    if str(device).lower() == "cpu":
                        torch.set_num_threads(1)
                    tier2_model = self._build_model_with_weights(desc, trained_state)
                    tier2_model.to(device)

                    lr = self._suggest_lr(desc)
                    optimizer = torch.optim.Adam(
                        tier2_model.parameters(), lr=lr, weight_decay=1e-5)

                    train_loss, loss_history, val_history = TieredEvaluator.evaluate(
                        tier2_model,
                        train_loader,
                        criterion,
                        optimizer,
                        num_epochs=TieredEvaluator.TIER2_EPOCHS,
                        device=device,
                        val_loader=eval_loader,
                    )

                    if train_loss == float("inf") or math.isnan(train_loss):
                        return None, "nan"

                    if self._has_nan_weights(tier2_model):
                        return None, "nan"

                    score_loss = val_history[-1] if (val_loader is not None and val_history) else train_loss
                    score = self._score(
                        score_loss, tier2_model, complexity_penalty, desc)
                    tier2_trained_state = tier2_model.state_dict()
                    return (desc, tier2_trained_state, train_loss, score, tier2_model, loss_history), None
                except RuntimeError as e:
                    err_type = "shape" if ("shape" in str(e).lower() or "size" in str(e).lower()) else "other"
                    return None, err_type
                except Exception as e:
                    return None, "other"

            if max_workers > 1 and len(tier1_results[:tier2_count]) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    t2_out = list(executor.map(_eval_tier2_candidate, tier1_results[:tier2_count]))
            else:
                t2_out = [_eval_tier2_candidate(item) for item in tier1_results[:tier2_count]]

            tier2_results = []
            for res, err_type in t2_out:
                if res is not None:
                    tier2_results.append(res)
                elif err_type == "nan":
                    stats.nan_errors += 1
                elif err_type == "shape":
                    stats.shape_errors += 1
                    stats.tier2_failures += 1
                elif err_type == "other":
                    stats.other_errors += 1
                    stats.tier2_failures += 1

            tier2_results.sort(key=lambda item: item[3])
            stats.tier2_survivors = len(tier2_results)

            finalists = tier2_results[:2] if tier2_results else tier1_results[:2]

            self._emit({
                "type": "tier",
                "tier": 2,
                "generation": gen + 1,
                "survivors": stats.tier2_survivors,
                "finalists": len(finalists),
                "status": f"Gen {gen + 1}: training {len(finalists)} finalists...",
                "progress": _progress(gen) + (50 / max(generations, 1)) * 0.66,
            })

            finals_best_score = float("inf")
            for desc, trained_state, _, _, model, _ in finalists:
                try:
                    finals_model = self._build_model_with_weights(
                        desc, trained_state)
                    finals_model.to(device)

                    lr = self._suggest_lr(desc)
                    optimizer = torch.optim.Adam(
                        finals_model.parameters(), lr=lr, weight_decay=1e-5)

                    finals_epochs = min(10, max(max_epochs, TieredEvaluator.FINALS_EPOCHS))
                    train_loss, finals_train_history, finals_val_history = TieredEvaluator.evaluate(
                        finals_model,
                        train_loader,
                        criterion,
                        optimizer,
                        num_epochs=finals_epochs,
                        device=device,
                        val_loader=eval_loader,
                    )

                    val_loss = finals_val_history[-1] if (eval_loader is not None and finals_val_history) else train_loss

                    if train_loss == float("inf") or math.isnan(
                            train_loss) or val_loss == float("inf") or math.isnan(val_loss):
                        stats.nan_errors += 1
                        continue

                    if self._has_nan_weights(finals_model):
                        stats.nan_errors += 1
                        continue

                    effective_score = self._score(
                        val_loss, finals_model, complexity_penalty, desc)

                    if effective_score < finals_best_score:
                        finals_best_score = effective_score

                    if effective_score < best_score:
                        best_score = effective_score
                        best_descriptor = copy.deepcopy(desc)
                        best_model = finals_model
                        best_train_loss = train_loss
                        best_val_loss = val_loss
                        self._best_model_state = finals_model.state_dict()
                        self._best_descriptor = copy.deepcopy(desc)
                        self._champion_training = {
                            "train_loss_history": self._sanitize_history(finals_train_history),
                            "val_loss_history": self._sanitize_history(finals_val_history),
                            "epochs_trained": len(finals_train_history),
                            "early_stopped": len(finals_train_history) < finals_epochs,
                        }
                        self._emit({
                            "type": "new_best",
                            "generation": gen + 1,
                            "train_loss": round(train_loss, 6),
                            "val_loss": round(val_loss, 6),
                            "score": round(effective_score, 6),
                            "status": f"Gen {gen + 1} new best: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, score={effective_score:.6f}",
                            "progress": _progress(gen + 1),
                        })
                        logger.info(
                            f"Gen {
                                gen +
                                1} new best: train_loss={
                                train_loss:.6f}, val_loss={
                                val_loss:.6f}, score={
                                effective_score:.6f}")
                except RuntimeError as e:
                    if "shape" in str(e).lower() or "size" in str(e).lower():
                        stats.shape_errors += 1
                    else:
                        stats.other_errors += 1
                    logger.error(
                        f"Finals eval failed: {
                            type(e).__name__}: {e}")
                    logger.debug(traceback.format_exc())
                    stats.finals_failures += 1
                    continue
                except Exception as e:
                    self._emit({
                        "type": "warning",
                        "generation": gen + 1,
                        "status": f"Gen {gen + 1} finals eval failed: {type(e).__name__}: {e}",
                        "progress": _progress(gen + 1),
                    })
                    logger.debug(
                        f"Finals eval failed: {
                            type(e).__name__}: {e}")
                    stats.other_errors += 1
                    logger.error(
                        f"Finals eval failed: {
                            type(e).__name__}: {e}")
                    logger.debug(traceback.format_exc())
                    stats.finals_failures += 1
                    continue

            stats.best_train_loss = best_train_loss
            stats.best_val_loss = best_val_loss
            stats.best_score = best_score
            self._best_score_history.append(best_score)
            self.generation_stats.append(stats)

            self._emit({
                "type": "generation",
                "phase": "complete",
                "generation": gen + 1,
                "generations": generations,
                "best_score": _sanitize_float(round(best_score, 6)) if not math.isinf(best_score) else "inf",
                "best_val_loss": _sanitize_float(round(best_val_loss, 6)) if not math.isinf(best_val_loss) else "inf",
                "mutation_successes": stats.mutation_successes,
                "mutation_attempts": stats.mutation_attempts,
                "tier1_survivors": stats.tier1_survivors,
                "tier2_survivors": stats.tier2_survivors,
                "status": f"Generation {gen + 1}/{generations} complete",
                "progress": _progress(gen + 1),
            })

            elite_count = max(3, self.population_size // 5)
            elite = tier2_results[:elite_count] if tier2_results else tier1_results[:elite_count]

            new_population = []

            # Carry the best elites forward unchanged.
            preserved_elite_count = min(2, len(elite))
            for i in range(preserved_elite_count):
                elite_desc = copy.deepcopy(elite[i][0])
                elite_sd = elite[i][4].state_dict()
                new_population.append((elite_desc, elite_sd))
                logger.debug(
                    f"Preserved elite {
                        i +
                        1} unmutated (score={
                        elite[i][3]:.6f})")

            for i in range(preserved_elite_count, len(elite)):
                elite_desc = copy.deepcopy(elite[i][0])
                elite_sd = elite[i][4].state_dict()
                num_mutations = random.randint(1, 3)  # Reduced from 2-5
                self._mutate_and_track(elite_desc, elite[i][0], generation=gen + 2,
                                       num_mutations=num_mutations, stats=stats)
                new_population.append((elite_desc, elite_sd))

            attempts = 0
            max_attempts = self.population_size * 10
            while len(
                    new_population) < self.population_size and attempts < max_attempts:
                attempts += 1
                elite_choice = random.choice(elite)
                parent_desc = elite_choice[0]
                parent_sd = elite_choice[4].state_dict()
                child = copy.deepcopy(parent_desc)
                num_mutations = random.randint(2, 5)
                if self._mutate_and_track(child, parent_desc, generation=gen + 2,
                                          num_mutations=num_mutations, stats=stats) > 0:
                    new_population.append((child, parent_sd))

            species_best = {}
            for item in tier2_results:
                species = self._get_species(item[0])
                if species not in species_best or item[3] < species_best[species][3]:
                    species_best[species] = item

            for species, item in species_best.items():
                if len(new_population) >= self.population_size:
                    break
                desc_json = item[0].to_json()
                if not any(p[0].to_json() ==
                           desc_json for p in new_population):
                    # Deepcopy retains lineage metadata.
                    new_population.append(
                        (copy.deepcopy(item[0]), item[4].state_dict()))

            if self._is_stagnant(window=5, relative_threshold=0.02):
                logger.info(
                    f"Stagnation detected at generation {
                        gen + 1}. Injecting aggressive diversity.")
                self._emit({
                    "type": "stagnation",
                    "generation": gen + 1,
                    "status": f"Stagnation detected at generation {gen + 1} — injecting aggressive diversity",
                    "progress": _progress(gen + 1),
                })
                inject_count = max(5, self.population_size // 3)

                for _ in range(inject_count // 2):
                    if len(new_population) >= self.population_size:
                        break
                    if self._best_model_state is not None and self._best_descriptor is not None:
                        fresh = copy.deepcopy(self._best_descriptor)
                        self._mutate_and_track(fresh, self._best_descriptor, generation=gen + 2,
                                               num_mutations=random.randint(4, 8), stats=stats)
                        new_population.append((fresh, self._best_model_state))

                # Restart from the baseline after stagnation.
                for _ in range(inject_count // 2):
                    if len(new_population) >= self.population_size:
                        break
                    fresh = copy.deepcopy(self.initial_descriptor)
                    self._mutate_and_track(fresh, self._root_desc, generation=gen + 2,
                                           num_mutations=random.randint(2, 5), stats=stats)
                    new_population.append((fresh, parent_state_dict))

            while len(new_population) < self.population_size:
                elite_choice = random.choice(elite)
                parent_desc = elite_choice[0]
                parent_sd = elite_choice[4].state_dict()
                child = copy.deepcopy(parent_desc)
                self._mutate_and_track(child, parent_desc, generation=gen + 2,
                                       num_mutations=random.randint(1, 3), stats=stats)
                new_population.append((child, parent_sd))

            self.population = new_population
            logger.info(
                f"Generation {gen + 1} complete. "
                f"Best Score: {_sanitize_float(best_score)} | "
                f"Mutations: {
                    stats.mutation_successes}/{
                    stats.mutation_attempts} succeeded | "
                f"Structural: {
                    stats.structural_mutations}, Width: {
                    stats.width_mutations}, "
                f"Act: {
                    stats.activation_mutations}, Hyper: {
                    stats.hyperparam_mutations}, Swap: {
                    stats.layer_swap_mutations} | "
                f"Failures: T1={
                    stats.tier1_failures}, T2={
                    stats.tier2_failures}, F={
                    stats.finals_failures}"
            )

        self._emit({
            "type": "phase",
            "phase": "measuring",
            "status": "Measuring baseline vs optimized model...",
            "progress": 88,
        })

        baseline_metrics = None
        try:
            baseline_model = self._build_model_with_weights(
                copy.deepcopy(self.initial_descriptor), parent_state_dict)
            baseline_metrics = self._collect_model_metrics(
                baseline_model, self.initial_descriptor, eval_loader, criterion, device, problem_type
            )
        except Exception as e:
            logger.warning(f"Baseline metric collection failed: {e}")

        champion_metrics = None
        try:
            champion_metrics = self._collect_model_metrics(
                best_model, best_descriptor, eval_loader, criterion, device, problem_type
            )
        except Exception as e:
            logger.warning(f"Champion metric collection failed: {e}")

        improvement = self._build_improvement(
            baseline_metrics, champion_metrics)

        elapsed_seconds = round(
            time.time() - self._start_time,
            2) if self._start_time else None

        mutation_timeline = self._mutation_events
        timeline_truncated = False
        if len(mutation_timeline) > 2000:
            mutation_timeline = mutation_timeline[:2000]
            timeline_truncated = True

        diagnostics = {
            "original_descriptor": json.loads(self.original_descriptor_json),
            "generation_stats": [s.to_dict() for s in self.generation_stats],
            "best_train_loss": _sanitize_float(round(best_train_loss, 6) if not math.isinf(best_train_loss) and not math.isnan(best_train_loss) else best_train_loss),
            "best_val_loss": _sanitize_float(round(best_val_loss, 6) if not math.isinf(best_val_loss) and not math.isnan(best_val_loss) else best_val_loss),
            "best_score": _sanitize_float(round(best_score, 6) if not math.isinf(best_score) and not math.isnan(best_score) else best_score),
            "total_mutation_attempts": sum(s.mutation_attempts for s in self.generation_stats),
            "total_mutation_successes": sum(s.mutation_successes for s in self.generation_stats),
            "overall_mutation_success_rate": f"{sum(s.mutation_successes for s in self.generation_stats) / max(1, sum(s.mutation_attempts for s in self.generation_stats)):.1%}",
            "used_validation_set": val_loader is not None,
            "stagnation_events": sum(1 for i in range(2, len(self._best_score_history))
                                     if self._is_stagnant_at(i, window=5)),
            "total_shape_errors": sum(s.shape_errors for s in self.generation_stats),
            "total_nan_errors": sum(s.nan_errors for s in self.generation_stats),
            "total_other_errors": sum(s.other_errors for s in self.generation_stats),
            "population_size": self.population_size,
            "generations_run": generations,
            "elapsed_seconds": elapsed_seconds,
            "baseline": baseline_metrics,
            "champion": champion_metrics,
            "improvement": improvement,
            "champion_training": self._champion_training or None,
            "mutation_timeline": mutation_timeline,
            "mutation_timeline_truncated": timeline_truncated,
            "champion_lineage": self._champion_lineage(),
            "champion_individual_id": self._get_meta(self._best_descriptor).get("id"),
        }

        return best_descriptor, best_model, diagnostics

    def _is_stagnant_at(self, index: int, window: int = 5) -> bool:
        effective_window = min(window, max(1, index))
        if effective_window < 2:
            return False
        recent = self._best_score_history[index - effective_window:index]
        for i in range(1, len(recent)):
            if abs(recent[i] - recent[i - 1]) / \
                    max(abs(recent[i - 1]), 1e-8) > 0.02:
                return False
        return True

    @staticmethod
    def _suggest_lr(descriptor: ArchitectureDescriptor) -> float:
        """Suggest learning rate based on architecture depth."""
        depth = MutationGrammar._graph_depth(descriptor)
        if depth <= 4:
            return 1e-3
        elif depth <= 8:
            return 5e-4
        else:
            return 1e-4
