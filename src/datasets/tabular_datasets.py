"""Tabular dataset loaders for UCI Adult and Covertype."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.datasets.image_datasets import _safe_num_workers


def get_adult(
    data_dir: str = "./data",
    batch_size: int = 256,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """
    Load UCI Adult (Census Income) dataset from OpenML.

    Binary classification: income >50K vs <=50K.
    48,842 instances, 14 features.

    Returns:
        (train_loader, val_loader, test_loader, num_train_samples, input_dim)
    """
    data = fetch_openml("adult", version=2, as_frame=True, data_home=data_dir, parser="auto")
    X = data.data
    y = data.target

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # One-hot encode categoricals, keep numerics
    X_processed = []
    for col in X.columns:
        if X[col].dtype == "object" or X[col].dtype.name == "category":
            # Convert to object to avoid Categorical fillna issues
            col_data = X[col].astype(object).fillna("missing")
            le_col = LabelEncoder()
            encoded = le_col.fit_transform(col_data)
            dummies = np.eye(len(le_col.classes_))[encoded]
            X_processed.append(dummies)
        else:
            vals = X[col].fillna(0).values.astype(np.float32).reshape(-1, 1)
            X_processed.append(vals)

    X_np = np.hstack(X_processed).astype(np.float32)
    y_np = y_encoded.astype(np.int64)

    # Stratified split: 70/15/15
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_np, y_np, test_size=0.15, random_state=seed, stratify=y_np
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.176, random_state=seed, stratify=y_trainval  # 0.176 ≈ 0.15/0.85
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    input_dim = X_train.shape[1]

    # Create dataloaders
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    num_workers = _safe_num_workers(num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, len(X_train), input_dim


def get_covertype(
    data_dir: str = "./data",
    batch_size: int = 512,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """
    Load Forest Covertype dataset from scikit-learn.

    Multi-class classification: 7 cover types.
    581,012 instances, 54 features.

    Returns:
        (train_loader, val_loader, test_loader, num_train_samples, input_dim)
    """
    data = fetch_covtype(data_home=data_dir)
    X_np = data.data.astype(np.float32)
    y_np = (data.target - 1).astype(np.int64)  # Classes 1-7 → 0-6

    # Stratified split: 70/15/15
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_np, y_np, test_size=0.15, random_state=seed, stratify=y_np
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.176, random_state=seed, stratify=y_trainval
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    input_dim = X_train.shape[1]

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    num_workers = _safe_num_workers(num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, len(X_train), input_dim
