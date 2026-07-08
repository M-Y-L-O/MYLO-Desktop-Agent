import torch
import torch.nn as nn
from typing import Dict, Any

class WeightCompatibilityEngine:
    @staticmethod
    def transfer_weights(source_state_dict: Dict[str, torch.Tensor], target_module: nn.Module) -> Dict[str, Any]:
        """
        Attempts a best-effort partial weight transfer from a legacy/mutated state_dict 
        to a new DynamicGraphModule built from ArchitectureDescriptor.
        """
        target_state_dict = target_module.state_dict()
        matched_keys = []
        unmatched_source = []
        unmatched_target = list(target_state_dict.keys())

        new_state_dict = {}

        for src_key, src_tensor in source_state_dict.items():
            # Map legacy keys.
            target_key = WeightCompatibilityEngine._resolve_key_mapping(src_key, target_state_dict)
            
            if target_key in target_state_dict:
                tgt_tensor = target_state_dict[target_key]
                
                # Check shape compatibility
                if src_tensor.shape == tgt_tensor.shape:
                    new_state_dict[target_key] = src_tensor
                    matched_keys.append(target_key)
                    if target_key in unmatched_target:
                        unmatched_target.remove(target_key)
                else:
                    # Smart partial transfer for structural changes
                    transferred = WeightCompatibilityEngine._partial_shape_transfer(src_tensor, tgt_tensor)
                    if transferred is not None:
                        new_state_dict[target_key] = transferred
                        matched_keys.append(target_key)
                        if target_key in unmatched_target:
                            unmatched_target.remove(target_key)
                    else:
                        unmatched_source.append(src_key)
            else:
                unmatched_source.append(src_key)

        # Initialize remaining target keys with their original initialization
        for tk in unmatched_target:
            new_state_dict[tk] = target_state_dict[tk]

        # Load standard dict strict=False
        target_module.load_state_dict(new_state_dict, strict=False)

        return {
            "matched_keys": matched_keys,
            "unmatched_source": unmatched_source,
            "unmatched_target": unmatched_target
        }

    @staticmethod
    def _resolve_key_mapping(src_key: str, target_state_dict: Dict[str, torch.Tensor]) -> str:
        """
        Heuristic key resolver to handle naming differences between 
        old fragile architectures and new named Node architectures.
        """
        # Node modules use 'node_modules.<node_id>.<param>'
        # Try finding suffix matches
        suffix = src_key.split('.')[-1]
        for tk in target_state_dict.keys():
            if tk.endswith(suffix) and tk.split('.')[-2] in src_key:
                return tk
                
        # Direct match check
        if f"node_modules.{src_key}" in target_state_dict:
            return f"node_modules.{src_key}"
            
        return src_key

    @staticmethod
    def _partial_shape_transfer(src_tensor: torch.Tensor, tgt_tensor: torch.Tensor) -> torch.Tensor:
        """
        Transfers weights safely up to the minimum common dimensions.
        Useful when an optimizer scale_width mutation happened.
        """
        tgt = tgt_tensor.clone()
        if len(src_tensor.shape) == len(tgt_tensor.shape):
            slices = tuple(slice(0, min(s, t)) for s, t in zip(src_tensor.shape, tgt_tensor.shape))
            tgt[slices] = src_tensor[slices]
            return tgt
            
        return None
