import torch
import torch.nn as nn
import copy
from typing import Dict, Any, List
from .ArchitectureDescriptor import ArchitectureDescriptor
from .NodeRegistry import NodeRegistry

class DynamicGraphModule(nn.Module):
    def __init__(self, descriptor: ArchitectureDescriptor):
        super().__init__()
        self.descriptor = descriptor
        self.node_modules = nn.ModuleDict()
        
        # Build modules using NodeRegistry
        for node in descriptor.nodes:
            # Check if this node type is a valid pytorch module
            try:
                mod = NodeRegistry.get_module(node.type, node.params)
                self.node_modules[node.id] = mod
            except ValueError:
                pass

        self.sorted_nodes = self._topological_sort()

    def _topological_sort(self) -> List[str]:
        # Simple topological sort
        indegree = {node.id: 0 for node in self.descriptor.nodes}
        indegree["input"] = 0
        indegree["output"] = 0
        
        adj = {node.id: [] for node in self.descriptor.nodes}
        adj["input"] = []

        for edge in self.descriptor.edges:
            if edge.target in indegree:
                indegree[edge.target] += 1
            if edge.source in adj:
                adj[edge.source].append(edge.target)

        queue = [n for n, deg in indegree.items() if deg == 0]
        sorted_order = []

        while queue:
            curr = queue.pop(0)
            if curr not in ("input", "output"):
                sorted_order.append(curr)
            if curr in adj:
                for neighbor in adj[curr]:
                    if neighbor in indegree:
                        indegree[neighbor] -= 1
                        if indegree[neighbor] == 0:
                            queue.append(neighbor)
        
        return sorted_order

    def _downstream_expects_sequence(self, node_id: str) -> bool:
        sequence_nodes = {"LSTM", "GRU", "RNN", "MultiheadAttention"}
        downstream_ids = [edge.target for edge in self.descriptor.edges if edge.source == node_id]
        for downstream_id in downstream_ids:
            downstream_node = next((n for n in self.descriptor.nodes if n.id == downstream_id), None)
            if downstream_node and downstream_node.type in sequence_nodes:
                return True
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tensors flowing through edges
        outputs = {"input": x}
        hidden_states = {}

        for node_id in self.sorted_nodes:
            # Collect inputs to this node; skip edges whose source hasn't been computed yet
            node_inputs = []
            for edge in self.descriptor.edges:
                if edge.target == node_id:
                    if edge.source in outputs:
                        node_inputs.append(outputs[edge.source])

            if not node_inputs:
                continue

            node_def = next((n for n in self.descriptor.nodes if n.id == node_id), None)
            if not node_def:
                continue

            # Merge rule: concat along last dim when multiple inputs arrive
            if len(node_inputs) == 1:
                in_tensor = node_inputs[0]
            elif node_def.type == "Add":
                in_tensor = torch.stack(node_inputs).sum(dim=0)
            elif node_def.type == "Concat":
                concat_dim = int(node_def.params.get("dim", -1))
                in_tensor = torch.cat(node_inputs, dim=concat_dim)
            else:
                raise ValueError(
                    f"Node {node_id} received {len(node_inputs)} inputs but type {node_def.type} is not an explicit merge op"
                )

            # Node execution
            execution_op = NodeRegistry.get_execution_semantic(node_def.type)

            if node_id in self.node_modules:
                mod = self.node_modules[node_id]

                if execution_op == "recurrent_lstm":
                    out, hidden = mod(in_tensor)
                    hidden_states[node_id] = hidden
                    if out.dim() == 3 and not self._downstream_expects_sequence(node_id):
                        outputs[node_id] = out[:, -1, :]
                    else:
                        outputs[node_id] = out
                elif execution_op == "recurrent_gru":
                    out, hidden = mod(in_tensor)
                    hidden_states[node_id] = hidden
                    if out.dim() == 3 and not self._downstream_expects_sequence(node_id):
                        outputs[node_id] = out[:, -1, :]
                    else:
                        outputs[node_id] = out
                else:
                    outputs[node_id] = mod(in_tensor)
            else:
                # Virtual nodes (e.g. merge/reshape) pass through
                outputs[node_id] = in_tensor

        # Find final output
        final_output_sources = [e.source for e in self.descriptor.edges if e.target == "output"]
        for src in final_output_sources:
            if src in outputs:
                return outputs[src]

        # Fallback
        for node_id in reversed(self.sorted_nodes):
            if node_id in outputs:
                return outputs[node_id]

        return x

class DescriptorModelBuilder:
    @staticmethod
    def build(descriptor: ArchitectureDescriptor) -> "DynamicGraphModule":
        normalized_descriptor = copy.deepcopy(descriptor)
        normalized_descriptor.normalize_inplace()
        normalized_descriptor.validate()
        return DynamicGraphModule(normalized_descriptor)