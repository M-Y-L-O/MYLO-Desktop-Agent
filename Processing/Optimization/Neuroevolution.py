import copy
import json
import random
import os
import tempfile
from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from Core.ArchitectureDescriptor import ArchitectureDescriptor, Node, Edge
from Core.DescriptorModelBuilder import DescriptorModelBuilder
from Core.WeightCompatibilityEngine import WeightCompatibilityEngine


class MutationGrammar:
    MAX_NODES = 64
    MAX_DEPTH = 16

    @staticmethod
    def scale_width(descriptor: ArchitectureDescriptor, target_node_id: str, factor: float) -> bool:
        """Scale the output width of a Linear/Conv/LSTM node and auto-propagate to downstream nodes."""
        node = next((n for n in descriptor.nodes if n.id == target_node_id), None)
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

        return True

    @staticmethod
    def mutate_activation(descriptor: ArchitectureDescriptor, target_node_id: str) -> bool:
        """Randomly swap the activation type of an activation node."""
        node = next((n for n in descriptor.nodes if n.id == target_node_id), None)
        if not node or node.type not in ["ReLU", "Tanh", "Sigmoid"]:
            return False
        choices = [t for t in ["ReLU", "Tanh", "Sigmoid"] if t != node.type]
        node.type = random.choice(choices)
        return True

    @staticmethod
    def add_layer(descriptor: ArchitectureDescriptor, after_node_id: str, new_node_type: str) -> bool:
        """Insert a new node immediately after after_node_id, rewiring the edge."""
        after_node = next((n for n in descriptor.nodes if n.id == after_node_id), None)
        if not after_node:
            return False

        if new_node_type == after_node.type and new_node_type != "Linear":
            return False

        outgoing_edges = [e for e in descriptor.edges if e.source == after_node_id]
        if not outgoing_edges:
            return False

        new_node_id = f"{new_node_type.lower()}_add_{random.randint(1000, 9999)}"
        out_dim = (
            after_node.params.get("out_features")
            or after_node.params.get("out_channels")
            or after_node.params.get("hidden_size")
            or after_node.params.get("normalized_shape")
            or 64
        )
        if isinstance(out_dim, (list, tuple)):
            out_dim = out_dim[-1] if out_dim else 64

        if new_node_type == "Linear":
            if out_dim is None:
                return False
            new_params = {"in_features": out_dim, "out_features": max(1, int(out_dim // 2))}
        elif new_node_type == "Dropout":
            new_params = {"p": 0.1}
        elif new_node_type == "LayerNorm":
            new_params = {"normalized_shape": out_dim}
        elif new_node_type == "Identity":
            new_params = {}
        elif new_node_type in ["ReLU", "Tanh", "Sigmoid"]:
            new_params = {}
        else:
            return False

        descriptor.nodes.append(Node(id=new_node_id, type=new_node_type, params=new_params))

        original_targets = [edge.target for edge in outgoing_edges]
        descriptor.edges = [edge for edge in descriptor.edges if edge not in outgoing_edges]

        descriptor.edges.append(Edge(source=after_node_id, target=new_node_id))
        for original_target in original_targets:
            descriptor.edges.append(Edge(source=new_node_id, target=original_target))

        descriptor.normalize_inplace()
        return True

    @staticmethod
    def add_skip_connection(
        descriptor: ArchitectureDescriptor,
        from_id: str,
        to_id: str,
    ) -> bool:
        
        # add a skip (residual) connection from from_id → to_id.
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
        if any(edge.source == from_id and edge.target == to_id for edge in descriptor.edges):
            return False

        # Guard against creating cycles: to_id must not be an ancestor of from_id
        # (simple reachability check in existing graph)
        def reachable(start: str, goal: str) -> bool:
            visited: set = set()
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
            return False  # Would create a cycle

        incoming_edges = [edge for edge in descriptor.edges if edge.target == to_id]
        if not incoming_edges:
            return False

        concat_id = f"concat_{random.randint(1000, 9999)}"
        descriptor.nodes.append(Node(id=concat_id, type="Concat", params={"dim": -1}))

        for edge in incoming_edges:
            edge.target = concat_id

        descriptor.edges.append(Edge(source=from_id, target=concat_id))
        descriptor.edges.append(Edge(source=concat_id, target=to_id))

        descriptor.normalize_inplace()

        return True

    @staticmethod
    def _topological_levels(descriptor: ArchitectureDescriptor) -> Dict[str, int]:
        indegree = {node.id: 0 for node in descriptor.nodes}
        adjacency = {node.id: [] for node in descriptor.nodes}
        adjacency["input"] = []
        levels = {"input": 0}

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
                levels[neighbor] = max(levels.get(neighbor, 0), levels.get(current, 0) + 1)
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
        return len(descriptor.nodes) <= MutationGrammar.MAX_NODES and MutationGrammar._graph_depth(descriptor) <= MutationGrammar.MAX_DEPTH

    @staticmethod
    def apply_random_mutation(descriptor: ArchitectureDescriptor) -> bool:
        
        candidates = []
        width_nodes = [n.id for n in descriptor.nodes if n.type in ["Linear", "Conv1d", "LSTM", "GRU"]]
        layer_nodes = [n.id for n in descriptor.nodes if n.type in ["Linear", "Conv1d"]]
        activation_nodes = [n.id for n in descriptor.nodes if n.type in ["ReLU", "Tanh", "Sigmoid"]]

        # Use default-argument capture to avoid late-binding closure bug
        if width_nodes:
            def _scale(desc=descriptor, nodes=list(width_nodes)):
                return MutationGrammar.scale_width(
                    desc,
                    random.choice(nodes),
                    random.choice([0.5, 0.8, 1.2, 1.5, 2.0]),
                )
            candidates.append(_scale)

        if layer_nodes:
            def _add_layer(desc=descriptor, nodes=list(layer_nodes)):
                return MutationGrammar.add_layer(
                    desc,
                    random.choice(nodes),
                    random.choice(["ReLU", "Dropout"]),
                )
            candidates.append(_add_layer)

        if activation_nodes:
            def _mutate_act(desc=descriptor, nodes=list(activation_nodes)):
                return MutationGrammar.mutate_activation(desc, random.choice(nodes))
            candidates.append(_mutate_act)

        # Skip connection: pick two non-adjacent nodes
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
                    if any(edge.source == from_id and edge.target == to_id for edge in descriptor.edges):
                        continue
                    valid_pairs.append((from_id, to_id))

            if valid_pairs:
                def _skip(desc=descriptor, pairs=list(valid_pairs)):
                    from_id, to_id = random.choice(pairs)
                    return MutationGrammar.add_skip_connection(desc, from_id, to_id)

                candidates.append(_skip)

        random.shuffle(candidates)
        for mutator in candidates:
            try:
                if mutator():
                    descriptor.normalize_inplace()
                    if not MutationGrammar._within_limits(descriptor):
                        continue
                    descriptor.validate()
                    return True
            except (ValueError, Exception):
                continue
        return False


class TieredEvaluator:
    TIER1_EPOCHS = 1
    TIER2_EPOCHS = 10

    @staticmethod
    def evaluate(
        model: nn.Module,
        data_loader,
        criterion,
        optimizer,
        num_epochs: int,
        device: str = "cpu",
    ) -> float:
        model.to(device)
        model.train()
        losses = []

        for _ in range(num_epochs):
            epoch_loss = 0.0
            batches = 0
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                outputs, batch_y = TieredEvaluator._align_shapes(outputs, batch_y, criterion)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batches += 1
            losses.append(epoch_loss / max(batches, 1))

        return losses[-1] if losses else float("inf")

    @staticmethod
    def validate(
        model: nn.Module,
        data_loader,
        criterion,
        device: str = "cpu",
    ) -> float:
        #evaluare
        model.to(device)
        model.eval()
        total_loss = 0.0
        batches = 0

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                outputs, batch_y = TieredEvaluator._align_shapes(outputs, batch_y, criterion)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                batches += 1

        return total_loss / max(batches, 1)

    @staticmethod
    def _align_shapes(outputs: torch.Tensor, targets: torch.Tensor, criterion: Optional[nn.Module] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
    def __init__(self, initial_descriptor: ArchitectureDescriptor, population_size: int = 20, statusCallback: Optional[callable] = None):
        self.initial_descriptor = initial_descriptor
        self.population_size = population_size
        self.statusCallback = statusCallback
        self.population: List[Tuple[ArchitectureDescriptor, Optional[Dict[str, torch.Tensor]]]] = []

    def initialize_population(
        self,
        parent_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        #populare
        self.population = [(copy.deepcopy(self.initial_descriptor), parent_state_dict)]
        attempts = 0
        max_attempts = self.population_size * 10

        while len(self.population) < self.population_size and attempts < max_attempts:
            attempts += 1
            child = copy.deepcopy(self.initial_descriptor)
            if MutationGrammar.apply_random_mutation(child):
                self.population.append((child, parent_state_dict))

        # copii nemodificate daca nu s-au generat destule mutatii
        while len(self.population) < self.population_size:
            self.population.append((copy.deepcopy(self.initial_descriptor), parent_state_dict))

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
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    @staticmethod
    def _score(loss: float, model: nn.Module, complexity_penalty: float) -> float:
        return loss + (complexity_penalty * NeuroevolutionEngine._model_complexity(model))

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
    ) -> Tuple[ArchitectureDescriptor, nn.Module]:
        """
        Run tiered neuroevolution and return (best_descriptor, best_model).

        Tier 1: quick 1-epoch screen of the full population.
        Tier 2: 10-epoch evaluation of the top quartile.
        Finals: full max_epochs training of the top 2.

        Args:
            criterion: Loss function. Defaults to MSELoss (regression).
        """
        self.initialize_population(parent_state_dict)

        if criterion is None:
            criterion = TieredEvaluator.select_criterion(problem_type)

        eval_loader = val_loader or train_loader
        best_descriptor = copy.deepcopy(self.initial_descriptor)
        best_model = self._build_model_with_weights(best_descriptor, parent_state_dict)
        best_score = float("inf")

        for gen in range(generations):
            # ── Tier 1: 1-epoch screen ──────────────────────────────────────
            tier1_results = []
            for desc, parent_state in self.population:
                try:
                    desc.validate()
                    model = self._build_model_with_weights(desc, parent_state)
                    model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    loss = TieredEvaluator.evaluate(
                        model,
                        train_loader,
                        criterion,
                        optimizer,
                        num_epochs=TieredEvaluator.TIER1_EPOCHS,
                        device=device,
                    )
                    score = self._score(loss, model, complexity_penalty)
                    tier1_results.append((desc, parent_state, loss, score, model))
                except Exception:
                    continue

            if not tier1_results:
                continue

            tier1_results.sort(key=lambda item: item[3])

            # ── Tier 2: 10-epoch eval of top quartile ───────────────────────
            tier2_count = max(2, self.population_size // 10)
            tier2_results = []

            for desc, parent_state, _, _, model in tier1_results[:tier2_count]:
                try:
                    model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    loss = TieredEvaluator.evaluate(
                        model,
                        eval_loader,
                        criterion,
                        optimizer,
                        num_epochs=TieredEvaluator.TIER2_EPOCHS,
                        device=device,
                    )
                    score = self._score(loss, model, complexity_penalty)
                    tier2_results.append((desc, parent_state, loss, score, model))
                except Exception:
                    continue

            tier2_results.sort(key=lambda item: item[3])
            finalists = tier2_results[:2] if tier2_results else tier1_results[:2]

            # ── Finals: full training of top 2 ──────────────────────────────
            for desc, parent_state, _, _, model in finalists:
                try:
                    model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    loss = TieredEvaluator.evaluate(
                        model,
                        eval_loader,
                        criterion,
                        optimizer,
                        num_epochs=max(max_epochs, TieredEvaluator.TIER2_EPOCHS),
                        device=device,
                    )
                    val_loss = TieredEvaluator.validate(model, eval_loader, criterion, device=device)
                    effective_loss = val_loss if val_loader else loss
                    effective_score = self._score(effective_loss, model, complexity_penalty)

                    if effective_score < best_score:
                        best_score = effective_score
                        best_descriptor = copy.deepcopy(desc)
                        best_model = model
                except Exception:
                    continue

            # ── Reproducere: elites + mutated offspring ────────────────────────
            elite_count = max(2, self.population_size // 10)
            elite = tier1_results[:elite_count]
            new_population = [(copy.deepcopy(item[0]), item[4].state_dict()) for item in elite]

            attempts = 0
            max_attempts = self.population_size * 10
            while len(new_population) < self.population_size and attempts < max_attempts:
                attempts += 1
                elite_choice = random.choice(elite)
                parent_desc = elite_choice[0]
                parent_sd = elite_choice[4].state_dict()
                child = copy.deepcopy(parent_desc)
                if MutationGrammar.apply_random_mutation(child):
                    new_population.append((child, parent_sd))

            # Padding
            while len(new_population) < self.population_size:
                new_population.append((copy.deepcopy(self.initial_descriptor), parent_state_dict))

            self.population = new_population
            if self.statusCallback:
                self.statusCallback({"status": f"Generation {gen + 1} complete. Best Score: {best_score:.6f}", "progress": 20 + ((gen+1) * 0.7)})
        return best_descriptor, best_model
