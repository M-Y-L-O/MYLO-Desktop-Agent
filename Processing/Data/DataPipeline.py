from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class DescriptorDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        sequence_length: Optional[int] = None,
    ):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        self.problem_type = "regression"

        if sequence_length and sequence_length > 1:
            self._build_sequences(sequence_length)

    def _build_sequences(self, sequence_length: int) -> None:
        usable = len(self.features) - sequence_length + 1
        if usable <= 0:
            raise ValueError(
                f"Not enough rows ({len(self.features)}) for sequence_length={sequence_length}"
            )

        self.features = torch.stack(
            [self.features[index:index + sequence_length] for index in range(usable)]
        )
        self.targets = torch.stack(
            [self.targets[index + sequence_length - 1] for index in range(usable)]
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


@dataclass
class DataPipelineResult:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: Optional[DataLoader]
    scalers: Dict[str, Any]
    input_shape: List[int]
    output_shape: List[int]
    feature_names: List[str]
    split_metadata: Dict[str, Any]

    def fresh_loaders(self, seed: int) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        train_loader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=True,
            generator=generator,
        )
        validation_loader = DataLoader(
            self.val_loader.dataset,
            batch_size=self.val_loader.batch_size,
            shuffle=False,
        )
        test_loader = (
            DataLoader(
                self.test_loader.dataset,
                batch_size=self.test_loader.batch_size,
                shuffle=False,
            )
            if self.test_loader is not None
            else None
        )
        return train_loader, validation_loader, test_loader

    def preprocessing_metadata(self) -> Dict[str, Any]:
        scaler_x = self.scalers.get("x")
        scaler_y = self.scalers.get("y")
        target_encoder = self.scalers.get("target_encoder")
        return {
            "feature_names": list(self.feature_names),
            "feature_encoder": self.scalers.get("feature_encoder"),
            "input_scaler": _scaler_metadata(scaler_x),
            "target_scaler": _scaler_metadata(scaler_y),
            "target_classes": (
                list(target_encoder.get("classes", [])) if target_encoder else None
            ),
            "split": self.split_metadata,
        }


class DataPipeline:
    @staticmethod
    def prepare_data(
        source: Union[str, pd.DataFrame],
        feature_cols: List[str],
        target_cols: List[str],
        problem_type: str = "regression",
        batch_size: int = 32,
        val_split: float = 0.2,
        test_split: float = 0.1,
        sequence_length: Optional[int] = None,
        random_state: int = 42,
        encoding: str = "none",
    ) -> DataPipelineResult:
        """Build deterministic train/validation/test loaders without fit-time leakage."""
        df = pd.read_csv(source) if isinstance(source, str) else source.copy()
        feature_cols = _unique_columns(feature_cols, "feature_cols")
        target_cols = _unique_columns(target_cols, "target_cols")
        problem_type = str(problem_type).strip().lower()
        encoding = str(encoding or "none").strip().lower()

        if problem_type not in {"regression", "classification"}:
            raise ValueError("problem_type must be 'regression' or 'classification'")
        if encoding not in {"none", "label", "one-hot"}:
            raise ValueError("encoding must be 'none', 'label', or 'one-hot'")
        if problem_type == "classification" and len(target_cols) != 1:
            raise ValueError("Classification currently supports exactly one target column.")
        if not feature_cols or not target_cols:
            raise ValueError("At least one feature and one target column are required.")
        overlap = sorted(set(feature_cols) & set(target_cols))
        if overlap:
            raise ValueError(f"Columns cannot be both features and targets: {overlap}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0.0 < val_split < 1.0:
            raise ValueError("val_split must be between 0 and 1")
        if not 0.0 <= test_split < 1.0:
            raise ValueError("test_split must be between 0 (inclusive) and 1")
        if val_split + test_split >= 0.8:
            raise ValueError("val_split + test_split must leave at least 20% for training")

        missing = [column for column in feature_cols + target_cols if column not in df]
        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")

        selected = df[feature_cols + target_cols].copy()
        selected["__mylo_row_id__"] = range(len(selected))
        if problem_type == "regression":
            for column in target_cols:
                converted = pd.to_numeric(selected[column], errors="coerce")
                invalid = selected[column].notna() & converted.isna()
                if invalid.any():
                    raise ValueError(f"Regression target '{column}' contains non-numeric values.")
                selected[column] = converted

        target_missing = selected[target_cols].isna().any(axis=1)
        dropped_target_rows = int(target_missing.sum())
        selected = selected.loc[~target_missing].reset_index(drop=True)
        minimum_rows = 3 if test_split == 0 else 5
        if len(selected) < minimum_rows:
            raise ValueError(
                f"Only {len(selected)} rows have usable targets; at least {minimum_rows} are required."
            )

        chronological = bool(sequence_length and sequence_length > 1)
        train_df, val_df, test_df = _split_rows(
            selected,
            target_cols,
            problem_type=problem_type,
            val_split=val_split,
            test_split=test_split,
            random_state=random_state,
            chronological=chronological,
        )

        transformed, feature_names, feature_encoder = _transform_features(
            train_df,
            val_df,
            test_df,
            feature_cols,
            encoding,
        )
        train_x_raw, val_x_raw, test_x_raw = transformed
        scaler_x = StandardScaler()
        train_x = scaler_x.fit_transform(train_x_raw)
        val_x = scaler_x.transform(val_x_raw)
        test_x = scaler_x.transform(test_x_raw) if test_x_raw is not None else None

        scaler_y = None
        target_encoder = None
        if problem_type == "regression":
            scaler_y = StandardScaler()
            train_y = scaler_y.fit_transform(train_df[target_cols].to_numpy(dtype="float64"))
            val_y = scaler_y.transform(val_df[target_cols].to_numpy(dtype="float64"))
            test_y = (
                scaler_y.transform(test_df[target_cols].to_numpy(dtype="float64"))
                if test_df is not None
                else None
            )
        else:
            target_column = target_cols[0]
            classes = pd.Index(pd.unique(train_df[target_column])).tolist()
            if len(classes) < 2:
                raise ValueError("The training partition must contain at least two classes.")
            class_to_index = {value: index for index, value in enumerate(classes)}
            train_y = _encode_targets(train_df[target_column], class_to_index, "training")
            val_y = _encode_targets(val_df[target_column], class_to_index, "validation")
            test_y = (
                _encode_targets(test_df[target_column], class_to_index, "test")
                if test_df is not None
                else None
            )
            target_encoder = {
                "classes": classes,
                "class_to_index": class_to_index,
            }

        target_dtype = torch.float32 if problem_type == "regression" else torch.long
        train_dataset = DescriptorDataset(
            torch.tensor(train_x, dtype=torch.float32),
            torch.tensor(train_y, dtype=target_dtype),
            sequence_length,
        )
        val_dataset = DescriptorDataset(
            torch.tensor(val_x, dtype=torch.float32),
            torch.tensor(val_y, dtype=target_dtype),
            sequence_length,
        )
        test_dataset = (
            DescriptorDataset(
                torch.tensor(test_x, dtype=torch.float32),
                torch.tensor(test_y, dtype=target_dtype),
                sequence_length,
            )
            if test_x is not None and test_y is not None
            else None
        )

        generator = torch.Generator()
        generator.manual_seed(int(random_state))
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = (
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            if test_dataset is not None
            else None
        )

        if chronological:
            input_shape = [-1, int(sequence_length), len(feature_names)]
        else:
            input_shape = [-1, len(feature_names)]
        output_width = len(target_encoder["classes"]) if target_encoder else len(target_cols)
        output_shape = [-1, output_width]
        split_metadata = _split_metadata(
            train_df,
            val_df,
            test_df,
            train_dataset,
            val_dataset,
            test_dataset,
            random_state=random_state,
            val_split=val_split,
            test_split=test_split,
            strategy="chronological" if chronological else (
                "stratified_random" if problem_type == "classification" else "random"
            ),
            dropped_target_rows=dropped_target_rows,
        )

        return DataPipelineResult(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            scalers={
                "x": scaler_x,
                "y": scaler_y,
                "target_encoder": target_encoder,
                "feature_encoder": feature_encoder,
            },
            input_shape=input_shape,
            output_shape=output_shape,
            feature_names=feature_names,
            split_metadata=split_metadata,
        )


def _unique_columns(columns: Sequence[str], name: str) -> List[str]:
    normalized = [str(column).strip() for column in columns if str(column).strip()]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} contains duplicate columns")
    return normalized


def _split_rows(
    df: pd.DataFrame,
    target_cols: List[str],
    *,
    problem_type: str,
    val_split: float,
    test_split: float,
    random_state: int,
    chronological: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    if chronological:
        train_end = int(len(df) * (1.0 - val_split - test_split))
        val_end = int(len(df) * (1.0 - test_split))
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:] if test_split > 0 else None
    else:
        stratify = df[target_cols[0]] if problem_type == "classification" else None
        try:
            if test_split > 0:
                remaining, test_df = train_test_split(
                    df,
                    test_size=test_split,
                    random_state=random_state,
                    shuffle=True,
                    stratify=stratify,
                )
            else:
                remaining, test_df = df, None
            adjusted_val = val_split / (1.0 - test_split)
            remaining_stratify = (
                remaining[target_cols[0]] if problem_type == "classification" else None
            )
            train_df, val_df = train_test_split(
                remaining,
                test_size=adjusted_val,
                random_state=random_state + 1,
                shuffle=True,
                stratify=remaining_stratify,
            )
        except ValueError as exc:
            if problem_type == "classification":
                raise ValueError(
                    "Could not create stratified train/validation/test partitions. "
                    "Each class needs enough examples for every partition."
                ) from exc
            raise

    partitions = {"training": train_df, "validation": val_df}
    if test_df is not None:
        partitions["test"] = test_df
    for name, partition in partitions.items():
        if partition.empty:
            raise ValueError(f"The {name} partition is empty; use more data or smaller splits.")
    return train_df.copy(), val_df.copy(), test_df.copy() if test_df is not None else None


def _transform_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    feature_cols: List[str],
    encoding: str,
) -> Tuple[Tuple[Any, Any, Optional[Any]], List[str], Dict[str, Any]]:
    frames = [train_df, val_df] + ([test_df] if test_df is not None else [])
    transformed = [pd.DataFrame(index=frame.index) for frame in frames]
    metadata: Dict[str, Any] = {"encoding": encoding, "columns": {}}
    feature_names: List[str] = []

    for column in feature_cols:
        if is_numeric_dtype(train_df[column]):
            train_values = pd.to_numeric(train_df[column], errors="coerce").replace(
                [float("inf"), float("-inf")], float("nan")
            )
            fill_value = float(train_values.mean()) if train_values.notna().any() else 0.0
            for output, frame in zip(transformed, frames):
                output[column] = pd.to_numeric(frame[column], errors="coerce").replace(
                    [float("inf"), float("-inf")], float("nan")
                ).fillna(fill_value).astype("float64")
            feature_names.append(column)
            metadata["columns"][column] = {"kind": "numeric", "fill_value": fill_value}
            continue

        if encoding == "none":
            raise ValueError(
                f"Feature '{column}' is categorical. Select label or one-hot encoding."
            )
        train_values = train_df[column].fillna("_MISSING_").astype(str)
        categories = pd.Index(pd.unique(train_values)).tolist()
        if encoding == "label":
            mapping = {value: index for index, value in enumerate(categories)}
            for output, frame in zip(transformed, frames):
                output[column] = (
                    frame[column].fillna("_MISSING_").astype(str).map(mapping).fillna(-1).astype("float64")
                )
            feature_names.append(column)
            metadata["columns"][column] = {
                "kind": "label",
                "categories": categories,
                "unknown_value": -1,
            }
            continue

        if len(categories) > 100:
            raise ValueError(
                f"Feature '{column}' has {len(categories)} training categories; one-hot encoding is capped at 100."
            )
        encoded_names = [f"{column}__{index}" for index in range(len(categories))]
        for output, frame in zip(transformed, frames):
            values = frame[column].fillna("_MISSING_").astype(str)
            for name, category in zip(encoded_names, categories):
                output[name] = (values == category).astype("float64")
        feature_names.extend(encoded_names)
        metadata["columns"][column] = {
            "kind": "one-hot",
            "categories": categories,
            "output_columns": encoded_names,
            "unknown_policy": "all_zero",
        }

    arrays = tuple(frame[feature_names].to_numpy(dtype="float64") for frame in transformed)
    test_array = arrays[2] if test_df is not None else None
    return (arrays[0], arrays[1], test_array), feature_names, metadata


def _encode_targets(series: pd.Series, mapping: Dict[Any, int], partition: str) -> Any:
    encoded = series.map(mapping)
    if encoded.isna().any():
        unknown = pd.unique(series[encoded.isna()]).tolist()
        raise ValueError(
            f"The {partition} partition contains classes absent from training: {unknown}"
        )
    return encoded.to_numpy(dtype="int64")


def _split_metadata(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    train_dataset: DescriptorDataset,
    val_dataset: DescriptorDataset,
    test_dataset: Optional[DescriptorDataset],
    **config: Any,
) -> Dict[str, Any]:
    partitions = {
        "train": (train_df, train_dataset),
        "validation": (val_df, val_dataset),
    }
    if test_df is not None and test_dataset is not None:
        partitions["test"] = (test_df, test_dataset)
    details = {}
    for name, (frame, dataset) in partitions.items():
        row_ids = [int(value) for value in frame["__mylo_row_id__"].tolist()]
        details[name] = {
            "rows": len(frame),
            "examples": len(dataset),
            "row_fingerprint": hashlib.sha256(
                json.dumps(row_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
    return {
        **config,
        "partitions": details,
        "test_used_for_selection": False,
    }


def _scaler_metadata(scaler: Any) -> Optional[Dict[str, Any]]:
    if scaler is None:
        return None
    return {
        "type": type(scaler).__name__,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "variance": scaler.var_.tolist(),
        "features_seen": int(scaler.n_features_in_),
    }
