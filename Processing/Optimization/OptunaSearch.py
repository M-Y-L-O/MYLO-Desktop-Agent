from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import optuna
    from optuna.pruners import (
        HyperbandPruner,
        MedianPruner,
        NopPruner,
        SuccessiveHalvingPruner,
        ThresholdPruner,
    )
    from optuna.samplers import (
        CmaEsSampler,
        GPSampler,
        QMCSampler,
        RandomSampler,
        TPESampler,
    )
    OPTUNA_AVAILABLE = True
except ImportError:  # pragma: no cover - we guard usage
    OPTUNA_AVAILABLE = False

# Reuse everything the previous code already imported successfully
from Core.ArchitectureDescriptor import ArchitectureDescriptor, Node, Edge
from Core.DescriptorModelBuilder import DescriptorModelBuilder
from Core.WeightCompatibilityEngine import WeightCompatibilityEngine
from Processing.Optimization.Neuroevolution import (
    MutationGrammar,
    TieredEvaluator,
    _sanitize_float,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Trial-level diagnostics
# -----------------------------------------------------------------------------
@dataclass
class TrialRecord:
    """Per-trial statistics, surfaced through the final diagnostics dict."""

    number: int
    score: float
    train_loss: float
    val_loss: float
    epochs_completed: int
    pruned: bool
    architecture_params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "number": self.number,
            "score": _sanitize_float(self.score),
            "train_loss": _sanitize_float(self.train_loss),
            "val_loss": _sanitize_float(self.val_loss),
            "epochs_completed": self.epochs_completed,
            "pruned": self.pruned,
            "architecture_params": self.architecture_params,
        }


# -----------------------------------------------------------------------------
# ArchitectureEncoder kept for diagnostics/back-compat only.
# The Optuna path does NOT need it for the surrogate (Optuna handles
# parameterization internally), but the previous code's diagnostics
# referenced an "original_descriptor" field, so we keep the import path
# working without forcing sklearn at runtime.
# -----------------------------------------------------------------------------
def _safe_encode_architecture(descriptor: ArchitectureDescriptor) -> Optional[np.ndarray]:
    """Best-effort architecture fingerprint for diagnostics. Returns None on failure."""
    try:
        # Lightweight feature fingerprint: node-type histogram + width stats.
        # Avoid sklearn dependency; the previous ArchitectureEncoder required it.
        type_counts: Dict[str, int] = {}
        widths: List[int] = []
        for n in descriptor.nodes:
            type_counts[n.type] = type_counts.get(n.type, 0) + 1
            if n.type == "Linear":
                widths.append(int(n.params.get("out_features", 0)))
            elif "Conv" in n.type:
                widths.append(int(n.params.get("out_channels", 0)))
            elif n.type in ("LSTM", "GRU"):
                widths.append(int(n.params.get("hidden_size", 0)))

        features = [
            float(len(descriptor.nodes)),
            float(len(descriptor.edges)),
            float(sum(type_counts.values())),
            float(np.mean(widths)) if widths else 0.0,
            float(np.std(widths)) if len(widths) > 1 else 0.0,
            float(max(widths)) if widths else 0.0,
            float(min(widths)) if widths else 0.0,
        ]
        return np.array(features, dtype=np.float32)
    except Exception:  # pragma: no cover - diagnostic only
        return None


# -----------------------------------------------------------------------------
# Structured search space
# -----------------------------------------------------------------------------
class ArchitectureSearchSpace:
    """
    Maps Optuna Trial hyperparameter suggestions to mutations on an
    ArchitectureDescriptor.

    The search space is intentionally structured (not raw node-by-node
    random mutations) because:

    - It is far smaller (~10^6 combinations vs ~10^20 for random mutation),
      so TPE converges quickly.
    - Each dimension has a clear, well-understood effect on the model
      (width, depth, activation, regularization), which TPE can model.
    - We can apply the same mutations deterministically, making trials
      reproducible from their params.

    Each Optuna trial produces a single (descriptor, lr, weight_decay,
    optimizer_choice) tuple. The architecture dimensions are conditional
    where it makes sense (e.g. skip connections only meaningful if depth
    allows them), which Optuna handles natively.
    """

    WIDTH_CHOICES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    DEPTH_RANGE = (-2, 2)  # -2..+2 extra hidden layers
    ACTIVATION_CHOICES = ["relu", "gelu", "silu", "tanh", "leaky_relu", "elu"]
    DROPOUT_RANGE = (0.0, 0.5)
    LR_LOG_RANGE = (1e-4, 1e-2)
    WD_LOG_RANGE = (1e-6, 1e-3)
    OPTIMIZER_CHOICES = ["adam", "adamw", "sgd"]
    NORM_CHOICES = ["batch", "layer", "none"]

    @classmethod
    def suggest(cls, trial: "optuna.Trial", base: ArchitectureDescriptor) -> Tuple[
        ArchitectureDescriptor, Dict[str, Any]
    ]:
        """Sample a new descriptor + training hparams from the search space."""
        descriptor = copy.deepcopy(base)
        params: Dict[str, Any] = {}

        # 1) Width scaling
        params["width_multiplier"] = trial.suggest_categorical(
            "width_multiplier", cls.WIDTH_CHOICES
        )
        cls._apply_width(descriptor, params["width_multiplier"])

        # 2) Depth delta
        params["depth_delta"] = trial.suggest_int(
            "depth_delta", cls.DEPTH_RANGE[0], cls.DEPTH_RANGE[1]
        )
        cls._apply_depth(descriptor, params["depth_delta"])

        # 3) Activation
        params["activation"] = trial.suggest_categorical(
            "activation", cls.ACTIVATION_CHOICES
        )
        cls._apply_activation(descriptor, params["activation"])

        # 4) Dropout
        params["dropout_rate"] = trial.suggest_float(
            "dropout_rate", cls.DROPOUT_RANGE[0], cls.DROPOUT_RANGE[1], step=0.05
        )
        cls._apply_dropout(descriptor, params["dropout_rate"])

        # 5) Normalization
        params["normalization"] = trial.suggest_categorical(
            "normalization", cls.NORM_CHOICES
        )
        cls._apply_normalization(descriptor, params["normalization"])

        # 6) Skip / residual connections (only if depth allows)
        params["use_skip"] = trial.suggest_categorical("use_skip", [True, False])
        if params["use_skip"] and params["depth_delta"] >= -1:
            cls._maybe_add_skip(descriptor)
        elif not params["use_skip"]:
            cls._remove_skip_nodes(descriptor)

        # 7) Training hyperparameters (per-trial, so each trial has its own budget)
        params["learning_rate"] = trial.suggest_float(
            "learning_rate", cls.LR_LOG_RANGE[0], cls.LR_LOG_RANGE[1], log=True
        )
        params["weight_decay"] = trial.suggest_float(
            "weight_decay", cls.WD_LOG_RANGE[0], cls.WD_LOG_RANGE[1], log=True
        )
        params["optimizer"] = trial.suggest_categorical(
            "optimizer", cls.OPTIMIZER_CHOICES
        )

        # 8) Optional: warm-start LR for transfer-learning scenarios
        # Only meaningful if the parent has weights, but we always suggest it;
        # warm-start factor just rescales the LR.
        params["warm_start_factor"] = trial.suggest_float(
            "warm_start_factor", 0.1, 1.0, step=0.1
        )

        return descriptor, params

    # ---- mutation helpers --------------------------------------------------
    @staticmethod
    def _apply_width(descriptor: ArchitectureDescriptor, mult: float) -> None:
        """Scale channel / feature sizes by `mult` (clamped to >= 8)."""
        for node in descriptor.nodes:
            if node.type == "Linear" and "out_features" in node.params:
                node.params["out_features"] = max(8, int(round(node.params["out_features"] * mult)))
            elif "Conv" in node.type and "out_channels" in node.params:
                node.params["out_channels"] = max(8, int(round(node.params["out_channels"] * mult)))
            elif node.type in ("LSTM", "GRU") and "hidden_size" in node.params:
                node.params["hidden_size"] = max(8, int(round(node.params["hidden_size"] * mult)))
            # Update downstream in_features for next Linear layer connected
            # by an edge (topology-aware width propagation)
            if node.type == "Linear" and "out_features" in node.params:
                for edge in descriptor.edges:
                    if edge.source == node.id:
                        for downstream in descriptor.nodes:
                            if downstream.id == edge.target and downstream.type == "Linear":
                                downstream.params["in_features"] = node.params["out_features"]

    @staticmethod
    def _apply_depth(descriptor: ArchitectureDescriptor, delta: int) -> None:
        """Add or remove hidden layers (only between input and output, only Linear/Conv)."""
        if delta == 0:
            return

        # Identify a "template" hidden node we can duplicate or remove.
        hidden_candidates = [
            n for n in descriptor.nodes
            if n.id not in ("input", "output")
            and n.type in ("Linear", "Conv1d", "Conv2d")
        ]
        if not hidden_candidates:
            return

        if delta > 0:
            # Insert new layer(s) before the last hidden layer (or output edge)
            # Find the last hidden node (deepest by topological order assumption)
            insertion_anchor = hidden_candidates[-1]
            for i in range(delta):
                new_node = copy.deepcopy(insertion_anchor)
                new_node.id = f"{insertion_anchor.id}_optuna_dup_{i}"
                # Slightly perturb to avoid identical duplicates
                if "out_features" in new_node.params:
                    new_node.params["out_features"] = max(8, int(new_node.params["out_features"] * 0.9))
                elif "out_channels" in new_node.params:
                    new_node.params["out_channels"] = max(8, int(new_node.params["out_channels"] * 0.9))
                descriptor.nodes.append(new_node)
        else:
            # Remove `|delta|` of the deepest hidden layers
            removable = sorted(
                hidden_candidates,
                key=lambda n: ArchitectureSearchSpace._node_depth(n, descriptor),
                reverse=True,
            )[: -delta]
            removable_ids = {n.id for n in removable}
            descriptor.nodes = [n for n in descriptor.nodes if n.id not in removable_ids]
            descriptor.edges = [
                e for e in descriptor.edges
                if e.source not in removable_ids and e.target not in removable_ids
            ]

    @staticmethod
    def _node_depth(node: Node, descriptor: ArchitectureDescriptor) -> int:
        """Heuristic depth: count of incoming edges from input/output side."""
        # Cheap proxy: longer id strings tend to be deeper, plus look at edges
        incoming = sum(1 for e in descriptor.edges if e.target == node.id)
        return incoming

    @staticmethod
    def _apply_activation(descriptor: ArchitectureDescriptor, activation: str) -> None:
        """Replace all activation nodes with the chosen activation type."""
        # Mapping: "leaky_relu" -> "LeakyReLU" etc.
        canon = {
            "relu": "ReLU",
            "gelu": "GELU",
            "silu": "SiLU",
            "tanh": "Tanh",
            "leaky_relu": "LeakyReLU",
            "elu": "ELU",
        }.get(activation, "ReLU")
        for node in descriptor.nodes:
            if node.type in ("ReLU", "GELU", "SiLU", "Tanh", "Sigmoid", "LeakyReLU", "ELU"):
                node.type = canon
                if node.type == "LeakyReLU" and "negative_slope" not in node.params:
                    node.params["negative_slope"] = 0.01

    @staticmethod
    def _apply_dropout(descriptor: ArchitectureDescriptor, rate: float) -> None:
        """Set dropout rate on all Dropout nodes; insert one if none exists."""
        dropout_nodes = [n for n in descriptor.nodes if n.type == "Dropout"]
        if not dropout_nodes:
            # Insert a single Dropout node after the last activation
            activations = [
                n for n in descriptor.nodes
                if n.type in ("ReLU", "GELU", "SiLU", "Tanh", "Sigmoid", "LeakyReLU", "ELU")
            ]
            if activations and rate > 0.0:
                new_dropout = Node(
                    id=f"optuna_dropout_{int(time.time() * 1000) % 100000}",
                    type="Dropout",
                    params={"p": float(rate)},
                )
                descriptor.nodes.append(new_dropout)
            return
        for d in dropout_nodes:
            d.params["p"] = float(rate)

    @staticmethod
    def _apply_normalization(descriptor: ArchitectureDescriptor, choice: str) -> None:
        """Insert or replace normalization layers."""
        # First, remove any existing norm nodes AND their connected edges
        norm_types = ("BatchNorm1d", "BatchNorm2d", "LayerNorm")
        removed_ids = {n.id for n in descriptor.nodes if n.type in norm_types}
        descriptor.nodes = [n for n in descriptor.nodes if n.type not in norm_types]
        descriptor.edges = [
            e for e in descriptor.edges
            if e.source not in removed_ids and e.target not in removed_ids
        ]

        if choice == "none":
            return

        norm_type = "LayerNorm" if choice == "layer" else "BatchNorm1d"
        if choice == "batch":
            has_conv = any("Conv" in n.type for n in descriptor.nodes)
            norm_type = "BatchNorm2d" if has_conv else "BatchNorm1d"

        # Insert one norm node per linear/conv layer (bounded to avoid explosion)
        insert_targets = [
            n for n in descriptor.nodes
            if n.type in ("Linear", "Conv1d", "Conv2d", "LSTM", "GRU")
        ]
        for n in insert_targets[:3]:
            new_norm = Node(
                id=f"{n.id}_norm_optuna",
                type=norm_type,
                params={},
            )
            descriptor.nodes.append(new_norm)

    @staticmethod
    def _maybe_add_skip(descriptor: ArchitectureDescriptor) -> None:
        """Add a single residual Add node if none exists."""
        has_add = any(n.type == "Add" for n in descriptor.nodes)
        if has_add:
            return
        # Find two Linear/Conv nodes to bridge
        linears = [n for n in descriptor.nodes if n.type in ("Linear", "Conv1d", "Conv2d")]
        if len(linears) < 2:
            return
        add_node = Node(
            id=f"optuna_skip_{int(time.time() * 1000) % 100000}",
            type="Add",
            params={},
        )
        descriptor.nodes.append(add_node)

    @staticmethod
    def _remove_skip_nodes(descriptor: ArchitectureDescriptor) -> None:
        """Remove Add/Concat skip-connection nodes."""
        skip_types = ("Add", "Concat")
        skip_ids = {n.id for n in descriptor.nodes if n.type in skip_types}
        descriptor.nodes = [n for n in descriptor.nodes if n.id not in skip_ids]
        descriptor.edges = [
            e for e in descriptor.edges
            if e.source not in skip_ids and e.target not in skip_ids
        ]


# -----------------------------------------------------------------------------
# Score function (preserved from previous Bayesian code for back-compat)
# -----------------------------------------------------------------------------
def _model_complexity(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _has_nan_weights(model: nn.Module) -> bool:
    for param in model.parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            return True
    return False


def _score_with_penalty(
    loss: float,
    model: nn.Module,
    complexity_penalty: float,
    descriptor: ArchitectureDescriptor,
) -> float:
    """
    Score = loss + adaptive complexity penalty + small structural priors.

    The structural bonuses (recurrent/attention/norm/activation) prefer
    "well-formed" architectures over pathological ones. They are tiny
    (sub-1% of typical loss values) but help the sampler break ties.
    """
    if math.isnan(loss) or math.isinf(loss):
        return float("inf")

    raw_penalty = complexity_penalty * _model_complexity(model)
    adaptive_penalty = raw_penalty * max(1.0, abs(loss) * 10.0)
    score = loss + adaptive_penalty

    has_recurrent = any(n.type in ("LSTM", "GRU") for n in descriptor.nodes)
    has_attention = any(n.type == "MultiheadAttention" for n in descriptor.nodes)
    has_activation = any(
        n.type in getattr(MutationGrammar, "ACTIVATION_TYPES", set())
        for n in descriptor.nodes
    )
    has_norm = any(
        n.type in getattr(MutationGrammar, "NORM_TYPES", set())
        for n in descriptor.nodes
    )

    if has_recurrent:
        score -= 0.005
    if has_attention:
        score -= 0.005
    if has_activation:
        score -= 0.002
    if has_norm:
        score -= 0.002

    return score


# -----------------------------------------------------------------------------
# Main engine
# -----------------------------------------------------------------------------
class OptunaSearchEngine:
    """
    Optuna-based architecture search. Drop-in replacement for BayesianSearchEngine.

    Usage:
        engine = OptunaSearchEngine(
            initial_descriptor=descriptor,
            n_trials=40,
            epochs_per_trial=10,
        )
        best_desc, best_model, diagnostics = engine.search(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=10,
            device="cpu",
            parent_state_dict=parent_state_dict,
            problem_type="regression",
            complexity_penalty=1e-5,
        )

    The same call signature as BayesianSearchEngine.search().
    """

    def __init__(
        self,
        initial_descriptor: ArchitectureDescriptor,
        n_trials: int = 40,
        epochs_per_trial: int = 10,
        min_resource: int = 2,
        reduction_factor: int = 3,
        sampler_type: str = "tpe",
        pruner_type: str = "hyperband",
        n_startup_trials: int = 8,
        seed: int = 42,
        statusCallback: Optional[Callable] = None,
    ) -> None:
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is not installed. Run `pip install optuna` to use OptunaSearchEngine."
            )

        self.initial_descriptor = initial_descriptor
        self.n_trials = max(1, int(n_trials))
        self.epochs_per_trial = max(1, int(epochs_per_trial))
        self.statusCallback = statusCallback or (lambda *_a, **_kw: None)

        # --- Sampler --------------------------------------------------------
        if sampler_type == "tpe":
            self.sampler: optuna.samplers.BaseSampler = TPESampler(
                n_startup_trials=n_startup_trials,
                n_ei_candidates=24,
                seed=seed,
                multivariate=True,
                group=True,
            )
        elif sampler_type == "cmaes":
            self.sampler = CmaEsSampler(
                seed=seed,
                n_startup_trials=n_startup_trials,
            )
        elif sampler_type == "gp":
            self.sampler = GPSampler(seed=seed, n_startup_trials=n_startup_trials)
        elif sampler_type == "qmt":
            self.sampler = QMCSampler(seed=seed, n_startup_trials=n_startup_trials)
        elif sampler_type == "random":
            self.sampler = RandomSampler(seed=seed)
        else:
            raise ValueError(f"Unknown sampler_type: {sampler_type}")

        # --- Pruner ---------------------------------------------------------
        if pruner_type == "hyperband":
            self.pruner: optuna.pruners.BasePruner = HyperbandPruner(
                min_resource=min_resource,
                max_resource=self.epochs_per_trial,
                reduction_factor=reduction_factor,
            )
        elif pruner_type == "median":
            self.pruner = MedianPruner(
                n_startup_trials=n_startup_trials,
                n_warmup_steps=min_resource,
            )
        elif pruner_type == "sh":
            self.pruner = SuccessiveHalvingPruner(
                min_resource=min_resource,
                reduction_factor=reduction_factor,
            )
        elif pruner_type == "none":
            self.pruner = NopPruner()
        else:
            raise ValueError(f"Unknown pruner_type: {pruner_type}")

        # --- Study ----------------------------------------------------------
        # Suppress Optuna's verbose logging -- the surrounding app has its own
        # status callback. Keep warnings/errors.
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self.study = optuna.create_study(
            direction="minimize",
            sampler=self.sampler,
            pruner=self.pruner,
        )

        # --- Bookkeeping ----------------------------------------------------
        self.trial_records: List[TrialRecord] = []
        self.pruned_count = 0
        self.completed_count = 0
        self.failed_count = 0
        self.evaluated_descriptors: List[ArchitectureDescriptor] = []
        self.evaluated_scores: List[float] = []

        self.best_descriptor = copy.deepcopy(initial_descriptor)
        self.best_score = float("inf")
        self.best_train_loss = float("inf")
        self.best_val_loss = float("inf")
        self.best_model_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_params: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _build_model_with_weights(
        descriptor: ArchitectureDescriptor,
        parent_state: Optional[Dict[str, torch.Tensor]],
    ) -> nn.Module:
        model = DescriptorModelBuilder.build(descriptor)
        if parent_state:
            try:
                WeightCompatibilityEngine.transfer_weights(parent_state, model)
            except Exception as e:  # weight transfer is best-effort
                logger.debug(f"Weight transfer failed: {e}")
        return model

    @staticmethod
    def _make_optimizer(
        name: str, params, lr: float, weight_decay: float
    ) -> torch.optim.Optimizer:
        if name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        if name == "sgd":
            return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    # -------------------------------------------------------- per-epoch train
    def _train_one_epoch(
        self,
        model: nn.Module,
        loader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
    ) -> float:
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            # Support both (x, y) and (x, y, ...) shapes
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch["x"], batch["y"]
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            if torch.isnan(loss) or torch.isinf(loss):
                return float("inf")
            loss.backward()
            # Gradient clipping for stability with deeper / wider nets
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
        return total_loss / max(1, n_batches)

    def _train_with_pruning(
        self,
        trial: "optuna.Trial",
        model: nn.Module,
        train_loader,
        val_loader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        max_epochs: int,
        device: str,
        complexity_penalty: float,
        descriptor: ArchitectureDescriptor,
    ) -> Tuple[float, float, int]:
        """
        Train for up to max_epochs, reporting intermediate val_loss to Optuna
        for pruning. Returns (best_score, best_val_loss, epochs_completed).
        """
        eval_loader = val_loader if val_loader is not None else train_loader
        best_val = float("inf")
        best_train = float("inf")
        best_score = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        epochs_completed = 0

        for epoch in range(int(max_epochs)):
            train_loss = self._train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            if math.isnan(train_loss) or math.isinf(train_loss):
                # Bail early on numerical failure
                return float("inf"), float("inf"), epoch

            val_loss = TieredEvaluator.validate(model, eval_loader, criterion, device)
            if math.isnan(val_loss) or math.isinf(val_loss):
                return float("inf"), float("inf"), epoch

            # Restore train mode for next epoch
            model.train()

            score = _score_with_penalty(val_loss, model, complexity_penalty, descriptor)
            epochs_completed = epoch + 1

            # Report to Optuna for pruning. We use the *raw* val_loss (not
            # the score) as the pruning signal because val_loss is on a
            # comparable scale across architectures, whereas the complexity
            # penalty makes the score architecture-dependent.
            trial.report(val_loss, step=epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

            if val_loss < best_val:
                best_val = val_loss
                best_train = train_loss
                best_score = score
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }

        # Restore best weights observed during training
        if best_state is not None:
            model.load_state_dict(best_state)

        return best_score, best_val, best_train, epochs_completed

    # ---------------------------------------------------------------- objective
    def _objective(
        self,
        trial: "optuna.Trial",
        train_loader,
        val_loader,
        criterion: nn.Module,
        max_epochs: int,
        device: str,
        parent_state_dict: Optional[Dict[str, torch.Tensor]],
        problem_type: str,
        complexity_penalty: float,
    ) -> float:
        # 1) Sample architecture + training hparams
        descriptor, params = ArchitectureSearchSpace.suggest(trial, self.initial_descriptor)
        try:
            descriptor.validate()
        except Exception as e:
            # Structured mutations produced an invalid descriptor; fall back
            # to MutationGrammar so the trial is not wasted.
            logger.debug(
                f"Structured descriptor invalid ({e}); falling back to MutationGrammar"
            )
            descriptor = copy.deepcopy(self.initial_descriptor)
            n_mutations = int(params.get("depth_delta", 1)) + 2
            try:
                if MutationGrammar.apply_random_mutations(
                    descriptor, num_mutations=max(1, n_mutations)
                ) == 0:
                    raise optuna.TrialPruned()
                descriptor.validate()
            except optuna.TrialPruned:
                raise
            except Exception as e2:
                logger.debug(f"Fallback descriptor invalid, pruning trial: {e2}")
                raise optuna.TrialPruned()

        # 2) Build the model
        try:
            model = self._build_model_with_weights(descriptor, parent_state_dict)
        except Exception as e:
            logger.debug(f"Model build failed, pruning trial: {e}")
            raise optuna.TrialPruned()

        # Warm-start LR scaling (smaller LR for fine-tuning from parent)
        lr = params["learning_rate"] * params["warm_start_factor"]
        optimizer = self._make_optimizer(
            params["optimizer"], model.parameters(), lr, params["weight_decay"]
        )
        model.to(device)

        # 3) Train with pruning
        try:
            score, val_loss, best_train, epochs_completed = self._train_with_pruning(
                trial,
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                max_epochs=max_epochs,
                device=device,
                complexity_penalty=complexity_penalty,
                descriptor=descriptor,
            )
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.debug(f"Trial failed: {type(e).__name__}: {e}")
            raise optuna.TrialPruned()

        if _has_nan_weights(model):
            raise optuna.TrialPruned()

        # 4) Bookkeeping
        # Compute train_loss at the loaded best-val weights for diagnostics
        train_loss = TieredEvaluator.validate(
            model, train_loader, criterion, device
        )
        if math.isnan(train_loss) or math.isinf(train_loss):
            train_loss = best_train  # fallback to in-loop best
        record = TrialRecord(
            number=trial.number,
            score=score,
            train_loss=train_loss,
            val_loss=val_loss,
            epochs_completed=epochs_completed,
            pruned=False,
            architecture_params=params,
        )
        self.trial_records.append(record)
        self.evaluated_descriptors.append(descriptor)
        self.evaluated_scores.append(score)
        self.completed_count += 1

        if score < self.best_score:
            self.best_score = score
            self.best_train_loss = train_loss
            self.best_val_loss = val_loss
            self.best_descriptor = copy.deepcopy(descriptor)
            self.best_model_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
            self.best_params = copy.deepcopy(params)
            logger.info(
                f"Trial {trial.number} new best: "
                f"val_loss={val_loss:.6f}, score={score:.6f}, "
                f"epochs={epochs_completed}, params={params}"
            )

        return score

    # ----------------------------------------------------------------- search
    def search(
        self,
        train_loader,
        val_loader=None,
        max_epochs: int = 10,
        device: str = "cpu",
        parent_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        criterion: Optional[nn.Module] = None,
        problem_type: str = "regression",
        complexity_penalty: float = 1e-5,
    ) -> Tuple[ArchitectureDescriptor, nn.Module, Dict[str, Any]]:
        """Run Optuna study and return (best_descriptor, best_model, diagnostics)."""

        if criterion is None:
            criterion = TieredEvaluator.select_criterion(problem_type)

        eval_loader = val_loader if val_loader is not None else train_loader
        if val_loader is None:
            logger.warning(
                "No validation loader provided. Using training data for evaluation "
                "(may overfit; pruning will be less reliable)."
            )

        logger.info(
            f"Starting Optuna search: n_trials={self.n_trials}, "
            f"epochs_per_trial={self.epochs_per_trial}, "
            f"sampler={type(self.sampler).__name__}, "
            f"pruner={type(self.pruner).__name__}"
        )

        start_time = time.time()

        # Wrap the objective so we can inject context
        def _wrapped(trial: "optuna.Trial") -> float:
            self.statusCallback({
                "status": f"Optuna trial {trial.number + 1}/{self.n_trials}",
                "progress": int(40 + 50 * (trial.number / max(1, self.n_trials))),
            })
            return self._objective(
                trial,
                train_loader=train_loader,
                val_loader=eval_loader,
                criterion=criterion,
                max_epochs=min(self.epochs_per_trial, int(max_epochs)),
                device=device,
                parent_state_dict=parent_state_dict,
                problem_type=problem_type,
                complexity_penalty=complexity_penalty,
            )

        # Custom callback to count pruned/failed trials for diagnostics
        def _state_callback(study: "optuna.Study", trial_: "optuna.trial.FrozenTrial") -> None:
            if trial_.state == optuna.trial.TrialState.PRUNED:
                self.pruned_count += 1
            elif trial_.state == optuna.trial.TrialState.FAIL:
                self.failed_count += 1
            elif trial_.state == optuna.trial.TrialState.COMPLETE:
                self.completed_count += 1

        # Suppress optuna-internal exception spam for pruned trials
        try:
            self.study.optimize(
                _wrapped,
                n_trials=self.n_trials,
                callbacks=[_state_callback],
                show_progress_bar=False,
            )
        except KeyboardInterrupt:
            logger.warning("Optuna search interrupted by user")

        elapsed = time.time() - start_time

        # Count trials by final state (more reliable than callback counters)
        from optuna.trial import TrialState as _TS
        self.completed_count = sum(
            1 for t in self.study.trials if t.state == _TS.COMPLETE
        )
        self.pruned_count = sum(
            1 for t in self.study.trials if t.state == _TS.PRUNED
        )
        self.failed_count = sum(
            1 for t in self.study.trials if t.state == _TS.FAIL
        )

        # Build final model with best weights
        best_model = self._build_model_with_weights(
            self.best_descriptor, self.best_model_state
        )

        # Diagnostics in the same shape as BayesianSearchEngine for back-compat
        # Note: we omit original_descriptor because Optuna does not need a
        # surrogate-encoded feature vector; the trial records carry enough info.
        diagnostics: Dict[str, Any] = {
            "strategy_used": "optuna_search",
            "n_trials_requested": self.n_trials,
            "n_trials_completed": self.completed_count,
            "n_trials_pruned": self.pruned_count,
            "n_trials_failed": self.failed_count,
            "prune_rate": (
                self.pruned_count / max(1, self.n_trials)
            ),
            "complete_rate": (
                self.completed_count / max(1, self.n_trials)
            ),
            "sampler": type(self.sampler).__name__,
            "pruner": type(self.pruner).__name__,
            "elapsed_seconds": round(elapsed, 2),
            "total_evaluations": len(self.evaluated_descriptors),
            "used_validation_set": val_loader is not None,
            "best_train_loss": _sanitize_float(
                round(self.best_train_loss, 6)
                if not math.isinf(self.best_train_loss) and not math.isnan(self.best_train_loss)
                else self.best_train_loss
            ),
            "best_val_loss": _sanitize_float(
                round(self.best_val_loss, 6)
                if not math.isinf(self.best_val_loss) and not math.isnan(self.best_val_loss)
                else self.best_val_loss
            ),
            "best_score": _sanitize_float(
                round(self.best_score, 6)
                if not math.isinf(self.best_score) and not math.isnan(self.best_score)
                else self.best_score
            ),
            "best_params": self.best_params,
            "trial_records": [r.to_dict() for r in self.trial_records],
        }

        self.statusCallback({"status": "Optuna search complete", "progress": 90})

        return self.best_descriptor, best_model, diagnostics
