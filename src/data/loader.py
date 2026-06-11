import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


class TimeSeriesDataset(Dataset):
    """Dataset for sliding window time series sequences."""

    def __init__(self, sequences: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def load_nab_data(file_path: str) -> pd.DataFrame:
    """Load a NAB dataset CSV file."""
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def normalize(values: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    """Normalize values to [0, 1] using MinMaxScaler."""
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(values.reshape(-1, 1)).flatten()
    return normalized, scaler


def create_sequences(values: np.ndarray, sequence_length: int) -> np.ndarray:
    """Create sliding window sequences from a 1D array."""
    sequences = []
    for i in range(len(values) - sequence_length + 1):
        sequences.append(values[i : i + sequence_length])
    return np.array(sequences)


def get_dataloaders(
    file_path: str,
    sequence_length: int,
    train_split: float,
    batch_size: int,
    normalize_data: bool = True,
) -> tuple[DataLoader, DataLoader, MinMaxScaler]:
    """End-to-end: load CSV, normalize, create sequences, split, return dataloaders."""
    df = load_nab_data(file_path)
    values = df["value"].values.astype(np.float32)

    scaler = None
    if normalize_data:
        values, scaler = normalize(values)

    sequences = create_sequences(values, sequence_length)

    split_idx = int(len(sequences) * train_split)
    train_seqs = sequences[:split_idx]
    val_seqs = sequences[split_idx:]

    train_loader = DataLoader(
        TimeSeriesDataset(train_seqs), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TimeSeriesDataset(val_seqs), batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, scaler
