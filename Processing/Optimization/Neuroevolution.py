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

        # Auto-propagate to downstream nodes
        downstream_targets = [e.target for e in descriptor.edges if e.source == target_node_id]
        for tgt_id in downstream_targets:
            tgt_node = next((n for n in descriptor.nodes if n.id == tgt_id), None)
            if not tgt_node:
                continue

            if tgt_node.type == "Linear":
                tgt_node.params["in_features"] = new_out
            elif "Conv" in tgt_node.type:
                tgt_node.params["in_channels"] = new_out
            elif tgt_node.type in ["LSTM", "GRU"]:
                tgt_node.params["input_size"] = new_out

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

        target_edge = next((e for e in descriptor.edges if e.source == after_node_id), None)
        if not target_edge:
            return False

        new_node_id = f"{new_node_type.lower()}_add_{random.randint(1000, 9999)}"
        out_dim = 64
        if after_node.type == "Linear":
            out_dim = after_node.params.get("out_features", 64)

        if new_node_type == "Linear":
            new_params = {"in_features": out_dim, "out_features": out_dim}
        elif new_node_type == "Dropout":
            new_params = {"p": 0.1}
        elif new_node_type in ["ReLU", "Tanh", "Sigmoid"]:
            new_params = {}
        else:
            return False

        descriptor.nodes.append(Node(id=new_node_id, type=new_node_type, params=new_params))
        original_target = target_edge.target
        target_edge.target = new_node_id
        descriptor.edges.append(Edge(source=new_node_id, target=original_target))
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

        # Determine sizes to decide if a projection is needed
        from_node = next(n for n in descriptor.nodes if n.id == from_id)
        to_node = next(n for n in descriptor.nodes if n.id == to_id)

        from_out = None
        to_in = None

        if from_node.type == "Linear":
            from_out = from_node.params.get("out_features")
        elif from_node.type in ("LSTM", "GRU"):
            from_out = from_node.params.get("hidden_size")

        if to_node.type == "Linear":
            to_in = to_node.params.get("in_features")

        needs_projection = (
            from_out is not None
            and to_in is not None
            and from_out != to_in
        )

        if needs_projection:
            proj_id = f"proj_skip_{random.randint(1000, 9999)}"
            proj_node = Node(
                id=proj_id,
                type="Linear",
                params={"in_features": from_out, "out_features": to_in},
            )
            descriptor.nodes.append(proj_node)
            descriptor.edges.append(Edge(source=from_id, target=proj_id))
            descriptor.edges.append(Edge(source=proj_id, target=to_id))
        else:
            descriptor.edges.append(Edge(source=from_id, target=to_id))

        # Mark to_id as a concat-merge node
        descriptor.merge_rules[to_id] = "concat"

        # Update to_node's in_features to account for concatenated input
        if to_node.type == "Linear" and to_in is not None and from_out is not None:
            effective_skip = to_in if not needs_projection else to_in
            to_node.params["in_features"] = to_in + effective_skip

        return True

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
                    random.choice(["ReLU", "Linear", "Dropout"]),
                )
            candidates.append(_add_layer)

        if activation_nodes:
            def _mutate_act(desc=descriptor, nodes=list(activation_nodes)):
                return MutationGrammar.mutate_activation(desc, random.choice(nodes))
            candidates.append(_mutate_act)

        # Skip connection: pick two non-adjacent nodes
        if len(width_nodes) >= 2:
            def _skip(desc=descriptor, nodes=list(width_nodes)):
                from_id = random.choice(nodes)
                to_id = random.choice(nodes)
                if from_id == to_id:
                    return False
                return MutationGrammar.add_skip_connection(desc, from_id, to_id)
            candidates.append(_skip)

        random.shuffle(candidates)
        for mutator in candidates:
            try:
                if mutator():
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
                outputs, batch_y = TieredEvaluator._align_shapes(outputs, batch_y)
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
                outputs, batch_y = TieredEvaluator._align_shapes(outputs, batch_y)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                batches += 1

        return total_loss / max(batches, 1)

    @staticmethod
    def _align_shapes(outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if outputs.dim() == 3 and targets.dim() == 2:
            outputs = outputs[:, -1, :]
        if outputs.dim() > targets.dim() and targets.dim() == 1:
            outputs = outputs.squeeze(-1)
        elif outputs.dim() == 1 and targets.dim() == 2 and targets.shape[1] == 1:
            outputs = outputs.unsqueeze(-1)
        return outputs, targets


class NeuroevolutionEngine:
    def __init__(self, initial_descriptor: ArchitectureDescriptor, population_size: int = 20):
        self.initial_descriptor = initial_descriptor
        self.population_size = population_size
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

    def evolve(
        self,
        train_loader,
        val_loader=None,
        generations: int = 5,
        max_epochs: int = 5,
        device: str = "cpu",
        parent_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        criterion: Optional[nn.Module] = None,
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
            criterion = nn.MSELoss()

        eval_loader = val_loader or train_loader
        best_descriptor = copy.deepcopy(self.initial_descriptor)
        best_model = self._build_model_with_weights(best_descriptor, parent_state_dict)
        best_loss = float("inf")

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
                    tier1_results.append((desc, parent_state, loss, model))
                except Exception:
                    continue

            if not tier1_results:
                continue

            tier1_results.sort(key=lambda item: item[2])

            # ── Tier 2: 10-epoch eval of top quartile ───────────────────────
            tier2_count = max(2, self.population_size // 4)
            tier2_results = []

            for desc, parent_state, _, model in tier1_results[:tier2_count]:
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
                    tier2_results.append((desc, parent_state, loss, model))
                except Exception:
                    continue

            tier2_results.sort(key=lambda item: item[2])
            finalists = tier2_results[:2] if tier2_results else tier1_results[:2]

            # ── Finals: full training of top 2 ──────────────────────────────
            for desc, parent_state, _, model in finalists:
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

                    if effective_loss < best_loss:
                        best_loss = effective_loss
                        best_descriptor = copy.deepcopy(desc)
                        best_model = model
                except Exception:
                    continue

            # ── Reproducere: elites + mutated offspring ────────────────────────
            elite = tier1_results[:max(2, self.population_size // 4)]
            new_population = [(copy.deepcopy(item[0]), item[3].state_dict()) for item in elite]

            attempts = 0
            max_attempts = self.population_size * 10
            while len(new_population) < self.population_size and attempts < max_attempts:
                attempts += 1
                parent_desc = random.choice(elite)[0]
                parent_sd = random.choice(elite)[3].state_dict()
                child = copy.deepcopy(parent_desc)
                if MutationGrammar.apply_random_mutation(child):
                    new_population.append((child, parent_sd))

            # Padding
            while len(new_population) < self.population_size:
                new_population.append((copy.deepcopy(self.initial_descriptor), parent_state_dict))

            self.population = new_population
            print(f"Generation {gen + 1} complete. Best Loss: {best_loss:.6f}")

        return best_descriptor, best_model
