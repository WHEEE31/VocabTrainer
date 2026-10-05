"""
Neural Network Ensemble Pipeline with Optuna Hyperparameter Tuning, PCA, and Logging.
"""

import os
import json
import math
import joblib
import numpy as np
import pandas as pd
import optuna
from typing import List, Optional
import tqdm

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = "C:\\Users\\super\\Documents\\projects\\VOCABULATOR"
FEATURES_PATH = os.path.join(ROOT, "assets", "database", "WORDSfeatures.csv")
LABELING_PATH = os.path.join(ROOT, "assets", "database", "WORDSlabeling.csv")

# ---------------- Data prep functions ----------------

def prep_features(features: pd.DataFrame) -> pd.DataFrame:
    emb = np.vstack(features["embedding"].map(lambda s: np.fromstring(s, sep=",")))
    emb_df = pd.DataFrame(emb, columns=[f"embed_{i}" for i in range(emb.shape[1])])
    features = pd.concat([features.drop(columns=["embedding"]), emb_df], axis=1)
    return features

def load_and_merge_data(features_path: str, labels_path: str) -> pd.DataFrame:
    features = pd.read_csv(features_path, keep_default_na=False, na_values=[])
    labels = pd.read_csv(labels_path, keep_default_na=False, na_values=[])
    features = prep_features(features)
    merged = labels.merge(features, on="word", how="inner")
    return merged

# ---------------- Dataset & Model ----------------

class TrainingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int], dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ---------------- Training helpers ----------------

def train_nn_model(X_train, y_train, X_val, y_val, params, model_path, log_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleRegressor(
        input_dim=X_train.shape[1],
        hidden=params['hidden_layers'],
        dropout=params['dropout']
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    loss_fn = nn.SmoothL1Loss()
    train_loader = DataLoader(TrainingDataset(X_train, y_train), batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(TrainingDataset(X_val, y_val), batch_size=params['batch_size'], shuffle=False)

    with open(log_csv, 'w') as f:
        f.write('epoch,train_loss,val_loss\n')

    best_val = float('inf')
    patience_counter = 0
    for epoch in tqdm.tqdm(range(params['epochs']), desc="Epochs", leave=False):
        model.train()
        train_loss_sum = 0
        for xb, yb in tqdm.tqdm(train_loader, desc="Training batches", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += loss.item() * xb.size(0)
        train_loss = train_loss_sum / len(train_loader.dataset)

        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for xb, yb in tqdm.tqdm(val_loader, desc="Validation batches", leave=False):
                xb, yb = xb.to(device), yb.to(device)
                val_loss_sum += loss_fn(model(xb), yb).item() * xb.size(0)
        val_loss = val_loss_sum / len(val_loader.dataset)

        with open(log_csv, 'a') as f:
            f.write(f"{epoch},{train_loss},{val_loss}\n")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= params['patience']:
                break

    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

# ---------------- Optuna tuning ----------------

def objective(trial, X_train, y_train):
    hidden_layers = [trial.suggest_int('h1', 64, 512), trial.suggest_int('h2', 32, 256)]
    params = {
        'hidden_layers': hidden_layers,
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
        'epochs': 50,
        'patience': 5
    }
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in tqdm.tqdm(kf.split(X_train), total=kf.get_n_splits(), desc="Optuna folds"):
        scaler = StandardScaler().fit(X_train[train_idx])
        X_tr, X_val = scaler.transform(X_train[train_idx]), scaler.transform(X_train[val_idx])
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        y_mean, y_std = y_tr.mean(), y_tr.std()+1e-8
        y_tr = (y_tr - y_mean)/y_std
        y_val = (y_val - y_mean)/y_std
        model_path = f"src/database/experiments/optuna/tmp_model_{trial.number}.pt"
        _ = train_nn_model(X_tr, y_tr, X_val, y_val, params, model_path, 'src/database/experiments/optuna/tmp_log.csv')
        with torch.no_grad():
            preds = _.forward(torch.tensor(X_val, dtype=torch.float32)).numpy()*y_std + y_mean
        scores.append(mean_absolute_error(y_train[val_idx], preds))
    return np.mean(scores)

# ---------------- PCA ----------------

def predict_pipeline(new_features_df: pd.DataFrame, out_dir: str, ensemble_size: int = 3) -> np.ndarray:
    """
    Predict AoA values for new data using the saved PCA, scalers, and ensemble models.
    Args:
        new_features_df: DataFrame containing features including 'embedding' column.
        out_dir: Directory where trained models and scalers were saved.
        ensemble_size: Number of ensemble models per fold.
    Returns:
        preds: np.ndarray of predictions.
    """
    # Prep features the same way as training
    new_features_df = prep_features(new_features_df)
    feature_cols = [c for c in new_features_df.columns if c != 'word']
    X = new_features_df[feature_cols].values

    # Apply PCA if it exists
    pca_path = os.path.join(out_dir, 'pca.pkl')
    if os.path.exists(pca_path):
        pca = joblib.load(pca_path)
        X = pca.transform(X)

    # Collect fold scalers and model predictions
    preds_all_folds = []
    fold_num = 1
    while os.path.exists(os.path.join(out_dir, f'scaler_fold{fold_num}.pkl')):
        scaler = joblib.load(os.path.join(out_dir, f'scaler_fold{fold_num}.pkl'))
        X_scaled = scaler.transform(X)
        # Load fold-specific y_mean, y_std (now always saved during training)
        y_stats_path = os.path.join(out_dir, f'y_stats_fold{fold_num}.json')
        if not os.path.exists(y_stats_path):
            raise FileNotFoundError(f"Missing y_stats for fold {fold_num} in {out_dir}")
        y_stats = json.load(open(y_stats_path))
        y_mean, y_std = y_stats['mean'], y_stats['std']

        fold_preds = np.zeros(len(X_scaled))
        for ens in range(ensemble_size):
            model_path = os.path.join(out_dir, f'fold{fold_num}_ens{ens}.pt')
            if not os.path.exists(model_path):
                continue
            params = json.load(open(os.path.join(out_dir, 'best_params.json')))
            model = SimpleRegressor(input_dim=X_scaled.shape[1], hidden=params['hidden_layers'], dropout=params['dropout'])
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            with torch.no_grad():
                pred_norm = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()
            fold_preds += pred_norm * y_std + y_mean
        fold_preds /= max(1, ensemble_size)
        preds_all_folds.append(fold_preds)
        fold_num += 1

    if preds_all_folds:
        preds = np.mean(np.vstack(preds_all_folds), axis=0)
    else:
        preds = np.zeros(len(X))
    return preds

# Drop this function after the other functions in the file, before or after run_pipeline as desired.
# Ensure in run_pipeline, after computing y_mean and y_std for a fold, you save them:
# json.dump({'mean': y_mean, 'std': y_std}, open(os.path.join(out_dir, f'y_stats_fold{fold}.json'), 'w'))

# ---------------- Main pipeline ----------------

def run_pipeline(features_path: str, labels_path: str, n_splits: int = 5, ensemble_size: int = 3,
                 pca_components: Optional[int] = None, out_dir: str = 'src/database/experiments/nn_ensemble'):
    os.makedirs(out_dir, exist_ok=True)
    df = load_and_merge_data(features_path, labels_path)
    feature_cols = df.columns[2:]
    X = df[feature_cols].values
    y = df['rtg'].values

    # PCA on embeddings if requested
    pca = None
    if pca_components is not None and pca_components < X.shape[1]:
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(X)
        joblib.dump(pca, os.path.join(out_dir, 'pca.pkl'))

    # Hyperparameter tuning
    best_params_path = os.path.join(out_dir, 'best_params.json')
    if os.path.exists(best_params_path):
        print("Found existing hyperparameter file. Skipping Optuna tuning...")
        best_params = json.load(open(best_params_path))
    else:
        # Hyperparameter tuning
        print("Starting Optuna hyperparameter tuning...")
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda t: objective(t, X, y), n_trials=20)
        best_params = study.best_params
        best_params['hidden_layers'] = [best_params.pop('h1'), best_params.pop('h2')]
        json.dump(best_params, open(best_params_path, 'w'))

    # Merge with fixed defaults to ensure all keys are present
    defaults = {'epochs': 50, 'patience': 5}
    for k, v in defaults.items():
        best_params.setdefault(k, v)

    # K-Fold ensemble training
    print("Training ensemble models...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df))
    for fold, (train_idx, val_idx) in enumerate(tqdm.tqdm(kf.split(X), total=n_splits, desc="Folds"), start=1):
        print(f"Fold {fold}/{n_splits}")
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train_raw, y_val_raw = y[train_idx], y[val_idx]
        scaler = StandardScaler().fit(X_train_raw)
        joblib.dump(scaler, os.path.join(out_dir, f'scaler_fold{fold}.pkl'))
        X_train = scaler.transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        y_mean, y_std = y_train_raw.mean(), y_train_raw.std()+1e-8
        y_train = (y_train_raw - y_mean)/y_std
        y_val = (y_val_raw - y_mean)/y_std
        
        json.dump({'mean': y_mean, 'std': y_std}, open(os.path.join(out_dir, f'y_stats_fold{fold}.json'), 'w'))

        fold_preds = np.zeros(len(val_idx))
        for ens in tqdm.tqdm(range(ensemble_size), desc=f"Ensemble models (fold {fold})", leave=False):
            print(f"  Ensemble model {ens+1}/{ensemble_size}")
            torch.manual_seed(42 + ens)
            model_path = os.path.join(out_dir, f"fold{fold}_ens{ens}.pt")
            log_csv = os.path.join(out_dir, f"fold{fold}_ens{ens}_log.csv")
            model = train_nn_model(X_train, y_train, X_val, y_val, best_params, model_path, log_csv)
            with torch.no_grad():
                pred_norm = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
            fold_preds += pred_norm*y_std + y_mean
        fold_preds /= ensemble_size
        oof_preds[val_idx] = fold_preds

    metrics = {
        'MAE': mean_absolute_error(y, oof_preds),
        'RMSE': math.sqrt(mean_squared_error(y, oof_preds)),
        'within_0.5': np.mean(np.abs(y-oof_preds) <= 0.5),
        'within_1.0': np.mean(np.abs(y-oof_preds) <= 1.0)
    }
    json.dump(metrics, open(os.path.join(out_dir, 'metrics.json'), 'w'))
    print(metrics)

run_pipeline(
    features_path=FEATURES_PATH,
    labels_path=LABELING_PATH,
    n_splits=5,
    ensemble_size=3,
    pca_components=None,    # 300, or None to skip PCA
    out_dir="src/database/experiments/nn_ensemble"
)

'''
import pandas as pd
from difficulty_pipeline import predict_pipeline

new_features_df = pd.read_csv("path/to/new_features.csv")
predictions = predict_pipeline(new_features_df, out_dir="experiments/nn_ensemble", ensemble_size=3)
print(predictions)
'''