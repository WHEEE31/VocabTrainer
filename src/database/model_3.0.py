import os
import json
import math
import joblib
import tqdm
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# Model with residuals
# ----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out += residual
        return torch.relu(out)

class DifficultyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=4, dropout=0.2, output_dim=1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers.append(ResidualBlock(hidden_dim, dropout))
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.backbone(x)
        return self.out(x)

# ----------------------------
# Data prep
# ----------------------------
def prep_features(features: pd.DataFrame, pca_components: Optional[int] = 200):
    emb = np.vstack(features["embedding"].map(lambda s: np.fromstring(s, sep=",")))
    if pca_components and pca_components < emb.shape[1]:
        pca = PCA(n_components=pca_components)
        emb = pca.fit_transform(emb)
        joblib.dump(pca, "pca_embeddings.pkl")
    emb_df = pd.DataFrame(emb, columns=[f"embed_{i}" for i in range(emb.shape[1])])
    num_df = features.drop(columns=["embedding", "word"])
    return pd.concat([num_df.reset_index(drop=True), emb_df], axis=1)

def bin_labels(y, bin_size):
    bins = np.floor(y / bin_size).astype(int)
    midpoints = (bins + 0.5) * bin_size
    return bins, midpoints

# ----------------------------
# Training loop
# ----------------------------
def train_model(model, train_X, train_y, val_X, val_y, epochs=50, lr=1e-3, batch_size=128, patience=5, classification=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if classification:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_X = torch.tensor(train_X, dtype=torch.float32).to(device)
    val_X = torch.tensor(val_X, dtype=torch.float32).to(device)
    if classification:
        train_y = torch.tensor(train_y, dtype=torch.long).to(device)
        val_y = torch.tensor(val_y, dtype=torch.long).to(device)
    else:
        train_y = torch.tensor(train_y, dtype=torch.float32).unsqueeze(1).to(device)
        val_y = torch.tensor(val_y, dtype=torch.float32).unsqueeze(1).to(device)

    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(train_X.size(0))
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = train_X[idx], train_y[idx]
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(val_X)
            val_loss = criterion(val_preds, val_y).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model

# ----------------------------
# Main pipeline
# ----------------------------
def run_pipeline(features_path, labels_path, n_splits=5, ensemble_size=3, ordinal_bins: Optional[float] = None):
    features = pd.read_csv(features_path, keep_default_na=False, na_values=[])
    labels = pd.read_csv(labels_path, keep_default_na=False, na_values=[])
    X_all = prep_features(features)
    df = labels.merge(X_all, left_on="word", right_index=True, how="inner")

    feature_cols = [c for c in df.columns if c not in ("word", "rtg")]
    X = df[feature_cols].values
    y = df["rtg"].values

    classification = ordinal_bins is not None
    if classification:
        y_classes, midpoints = bin_labels(y, ordinal_bins)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df))
    ensemble_models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        scaler = StandardScaler().fit(X_train_raw)
        joblib.dump(scaler, f"scaler_fold{fold}.pkl")
        X_train, X_val = scaler.transform(X_train_raw), scaler.transform(X_val_raw)

        if classification:
            y_train, y_val = y_classes[train_idx], y_classes[val_idx]
        else:
            y_mean, y_std = y[train_idx].mean(), y[train_idx].std()+1e-8
            y_train, y_val = (y[train_idx]-y_mean)/y_std, (y[val_idx]-y_mean)/y_std
            json.dump({'mean': y_mean, 'std': y_std}, open(f'y_stats_fold{fold}.json', 'w'))

        fold_models = []
        for ens in range(ensemble_size):
            model = DifficultyNet(input_dim=X.shape[1], hidden_dim=256, num_layers=4, dropout=0.2, output_dim=(len(np.unique(y_classes)) if classification else 1))
            model = train_model(model, X_train, y_train, X_val, y_val, classification=classification)
            model_path = f"fold{fold}_ens{ens}.pt"
            torch.save(model.state_dict(), model_path)
            fold_models.append(model_path)
            ensemble_models.append(model_path)

            with torch.no_grad():
                preds = model(torch.tensor(X_val, dtype=torch.float32))
                if classification:
                    preds = torch.softmax(preds, dim=1).cpu().numpy()
                    pred_vals = np.sum(preds * midpoints, axis=1)
                else:
                    preds = preds.numpy().flatten()
                    pred_vals = preds * y_std + y_mean
            oof_preds[val_idx] += pred_vals / ensemble_size

    # OOF metrics
    oof_metrics = {
        'MAE': mean_absolute_error(y, oof_preds),
        'RMSE': math.sqrt(mean_squared_error(y, oof_preds)),
        'within_0.5': np.mean(np.abs(y - oof_preds) <= 0.5),
        'within_1.0': np.mean(np.abs(y - oof_preds) <= 1.0)
    }
    json.dump(oof_metrics, open('oof_metrics.json', 'w'))

    # Full dataset ensemble metrics
    full_preds = np.zeros(len(df))
    for model_path in ensemble_models:
        model = DifficultyNet(input_dim=X.shape[1], hidden_dim=256, num_layers=4, dropout=0.2, output_dim=(len(np.unique(y_classes)) if classification else 1))
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(StandardScaler().fit_transform(X), dtype=torch.float32))
            if classification:
                preds = torch.softmax(preds, dim=1).cpu().numpy()
                pred_vals = np.sum(preds * midpoints, axis=1)
            else:
                preds = preds.numpy().flatten()
                pred_vals = preds * y_std + y_mean
        full_preds += pred_vals / len(ensemble_models)

    ensemble_metrics = {
        'MAE': mean_absolute_error(y, full_preds),
        'RMSE': math.sqrt(mean_squared_error(y, full_preds)),
        'within_0.5': np.mean(np.abs(y - full_preds) <= 0.5),
        'within_1.0': np.mean(np.abs(y - full_preds) <= 1.0)
    }
    json.dump(ensemble_metrics, open('ensemble_metrics.json', 'w'))

    print("OOF Metrics:", oof_metrics)
    print("Ensemble Metrics:", ensemble_metrics)


ROOT = "C:\\Users\\super\\Documents\\projects\\VOCABULATOR"
FEATURES_PATH = os.path.join(ROOT, "assets", "database", "WORDSfeatures.csv")
LABELING_PATH = os.path.join(ROOT, "assets", "database", "WORDSlabeling.csv")

run_pipeline(FEATURES_PATH, LABELING_PATH, n_splits=5, ensemble_size=3, ordinal_bins=0.5)
