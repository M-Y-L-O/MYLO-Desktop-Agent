import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional, Dict, Any


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

    def _build_sequences(self, sequence_length: int):
        usable = len(self.features) - sequence_length + 1
        if usable <= 0:
            raise ValueError(
                f"Not enough rows ({len(self.features)}) for sequence_length={sequence_length}"
            )

        seq_features = []
        seq_targets = []
        for idx in range(usable):
            seq_features.append(self.features[idx: idx + sequence_length])
            seq_targets.append(self.targets[idx + sequence_length - 1])

        self.features = torch.stack(seq_features)
        self.targets = torch.stack(seq_targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class DataPipelineResult:
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scalers: Dict[str, Any],
        input_shape: List[int],
        output_shape: List[int],
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scalers = scalers
        self.input_shape = input_shape
        self.output_shape = output_shape


class DataPipeline:
    @staticmethod
    def prepare_data(
        csv_path: str,
        feature_cols: List[str],
        target_cols: List[str],
        problem_type: str = "regression",
        batch_size: int = 32,
        val_split: float = 0.2,
        sequence_length: Optional[int] = None,
        random_state: int = 42,
    ) -> DataPipelineResult:
        """
        Leakage-free preprocessing: split first, fit scalers on train only.
        """
        df = pd.read_csv(csv_path)

        missing_features = [c for c in feature_cols if c not in df.columns]
        missing_targets = [c for c in target_cols if c not in df.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns in CSV: {missing_features}")
        if missing_targets:
            raise ValueError(f"Missing target columns in CSV: {missing_targets}")

        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean(numeric_only=True)).fillna(0.0)
        df[target_cols] = df[target_cols].fillna(df[target_cols].mean(numeric_only=True)).fillna(0.0)

        if sequence_length is not None:
            # Chronological split to preserve temporal continuity for sequence data
            split_idx = int(len(df) * (1 - val_split))
            train_df = df.iloc[:split_idx]
            val_df = df.iloc[split_idx:]
        else:
            train_df, val_df = train_test_split(
                df,
                test_size=val_split,
                random_state=random_state,
            )

        scaler_x = StandardScaler()
        train_x = scaler_x.fit_transform(train_df[feature_cols].values)
        val_x = scaler_x.transform(val_df[feature_cols].values)

        scaler_y = None
        if problem_type == "regression":
            scaler_y = StandardScaler()
            train_y = scaler_y.fit_transform(train_df[target_cols].values)
            val_y = scaler_y.transform(val_df[target_cols].values)
        else:
            train_y = train_df[target_cols].values
            val_y = val_df[target_cols].values

        train_features = torch.tensor(train_x, dtype=torch.float32)
        val_features = torch.tensor(val_x, dtype=torch.float32)
        target_dtype = torch.float32 if problem_type == "regression" else torch.long
        train_targets = torch.tensor(train_y, dtype=target_dtype)
        val_targets = torch.tensor(val_y, dtype=target_dtype)

        train_dataset = DescriptorDataset(train_features, train_targets, sequence_length)
        val_dataset = DescriptorDataset(val_features, val_targets, sequence_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        if sequence_length and sequence_length > 1:
            input_shape = [-1, sequence_length, len(feature_cols)]
        else:
            input_shape = [-1, len(feature_cols)]
        output_shape = [-1, len(target_cols)]

        return DataPipelineResult(
            train_loader=train_loader,
            val_loader=val_loader,
            scalers={"x": scaler_x, "y": scaler_y},
            input_shape=input_shape,
            output_shape=output_shape,
        )
