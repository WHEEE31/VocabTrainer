# this is for the model that will be used to determine the difficulty of the wrods

import os
import csv
import sys
import copy
import time
import tqdm
import torch
import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Subset, DataLoader

ROOT = "C:\\Users\\super\\Documents\\projects\\VOCABULATOR"
FEATURES_PATH = os.path.join(ROOT, "assets", "database", "WORDSfeatures.csv")
LABELING_PATH = os.path.join(ROOT, "assets", "database", "WORDSlabeling.csv")
DIFFICULTIES_PATH = os.path.join(ROOT, "assets", "database", "WORDSdifficulties.csv")
MODEL_PATH = os.path.join(ROOT, "src", "database", "difficulty_model.pt")
LOG_PATH = os.path.join(ROOT, "src", "database", "debugging", "traininglog.csv")

class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, df, feature_cols, target_col):
        self.words = df["word"].to_numpy()
        self.X = torch.tensor(df[feature_cols].to_numpy(), dtype=torch.float32)
        self.y = torch.tensor(df[target_col].to_numpy(), dtype=torch.float32)

    def __getitem__(self, idx):
        return self.words[idx], self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)


class ExecutionDataset(torch.utils.data.Dataset):
    def __init__(self, df, feature_cols):
        self.words = df["word"].to_numpy()
        self.X = torch.tensor(df[feature_cols].to_numpy(), dtype=torch.float32)

    def __getitem__(self, idx):
        return self.words[idx], self.X[idx]

    def __len__(self):
        return len(self.X)

class DifficultyRegressor(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)
    
#################################################################################################################
#################################################################################################################
####################################          E X P E R I M E N T S          ####################################
#################################################################################################################
#################################################################################################################

from sklearn.model_selection import KFold
import joblib
import lightgbm as lgb

def train_kfold_and_ensemble(df, feature_cols, y_std, y_mean, x_scaler, k=5):

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    models = []
    val_preds = numpy.zeros(len(df))  # stores predictions for each outer fold's validation set
    all_idx = numpy.arange(len(df))

    for fold, (outer_train_idx, outer_val_idx) in enumerate(kf.split(df)):
        print(f"===== Fold {fold+1}/{k} =====")

        # Outer split into train/val DataFrames
        train_df = df.iloc[outer_train_idx].reset_index(drop=True)
        val_df   = df.iloc[outer_val_idx].reset_index(drop=True)

        # Build datasets
        train_ds = TrainingDataset(train_df, feature_cols, target_col="rtg_norm")
        val_ds   = TrainingDataset(val_df, feature_cols, target_col="rtg_norm")

        # Inner split for early stopping
        inner_train_idx, inner_val_idx = train_test_split(
            list(range(len(train_ds))),
            test_size=0.2,
            random_state=42
        )
        train_subset = Subset(train_ds, inner_train_idx)
        earlystop_subset = Subset(train_ds, inner_val_idx)

        # Train
        model = DifficultyRegressor(input_dim=len(feature_cols))
        best_state = TRAIN(model, train_subset, earlystop_subset, limit=1)
        model.load_state_dict(best_state)
        model.eval()

        # Save model
        model_path = MODEL_PATH.replace(".pt", f"_fold{fold}.pt")
        torch.save(model.state_dict(), model_path)
        models.append(model_path)

        # Predict on outer fold's validation set
        preds = []
        with torch.no_grad():
            for i in range(len(val_ds)):
                _, x, _ = val_ds[i]
                p = model(x.unsqueeze(0)).item()
                preds.append(p)

        val_preds[outer_val_idx] = numpy.array(preds)

    # ---- Ensemble predictions over full dataset ----
    print("Ensembling fold models for full dataset predictions...")
    ensemble_preds = numpy.zeros(len(df))
    for model_path in models:
        mdl = DifficultyRegressor(input_dim=len(feature_cols))
        mdl.load_state_dict(torch.load(model_path, map_location='cpu'))
        mdl.eval()
        preds = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(df))):
                x = torch.tensor(df[feature_cols].iloc[i].to_numpy(), dtype=torch.float32)
                preds.append(mdl(x.unsqueeze(0)).item())
        ensemble_preds += numpy.array(preds)
    ensemble_preds /= len(models)

    # De-normalize
    ensemble_real = ensemble_preds * y_std + y_mean

    # Evaluate metrics
    y_true = df["rtg"].to_numpy()
    mae = numpy.mean(numpy.abs(ensemble_real - y_true))
    rmse = numpy.sqrt(numpy.mean((ensemble_real - y_true) ** 2))
    within_half = numpy.mean(numpy.abs(ensemble_real - y_true) <= 0.5)
    within_one = numpy.mean(numpy.abs(ensemble_real - y_true) <= 1.0)

    print(f"Ensemble MAE: {mae:.4f}, RMSE: {rmse:.4f}, "
          f"within0.5: {within_half*100:.2f}%, within1: {within_one*100:.2f}%")

    # Save scalers & metadata
    joblib.dump({
        "feature_cols": feature_cols,
        "x_mean": x_scaler.mean_,
        "x_scale": x_scaler.scale_,
        "y_mean": y_mean,
        "y_std": y_std,
        "model_paths": models
    }, MODEL_PATH.replace(".pt", "_meta.joblib"))

    return ensemble_real, df, models

def run_lightgbm(df, feature_cols):
    X = df[feature_cols].to_numpy()
    y = df["rtg"].to_numpy()

    # simple train/test
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "l2",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
    }

    model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=1000)

    preds = model.predict(X_val, num_iteration=model.best_iteration)
    mae = numpy.mean(numpy.abs(preds - y_val))
    rmse = numpy.sqrt(numpy.mean((preds - y_val) ** 2))
    within_half = numpy.mean(numpy.abs(preds - y_val) <= 0.5)
    within_one = numpy.mean(numpy.abs(preds - y_val) <= 1.0)
    print("LightGBM val MAE:", mae, "RMSE:", rmse, "within0.5:", within_half, "within1:", within_one)

    return model

def LIGHTGBM_EXPERIMENT():
    features = pandas.read_csv(FEATURES_PATH, keep_default_na=False, na_values=[])
    labels_df = pandas.read_csv(LABELING_PATH, keep_default_na=False, na_values=[])
    features_df = prep_features(features)

    training_data = labels_df.merge(features_df, on="word", how="inner")
    feature_cols = list(training_data.columns)[2:]

    x_scaler = StandardScaler().fit(training_data[feature_cols])
    training_data[feature_cols] = x_scaler.transform(training_data[feature_cols])                                               

    y_mean = training_data["rtg"].mean()
    y_std = training_data["rtg"].std(ddof=0) + 1e-8
    df = training_data.assign(
        rtg_norm = (training_data["rtg"] - y_mean) / y_std
    )

    run_lightgbm(df, feature_cols)

def KFOLD_EXPERIMENT():
    features = pandas.read_csv(FEATURES_PATH, keep_default_na=False, na_values=[])
    labels_df = pandas.read_csv(LABELING_PATH, keep_default_na=False, na_values=[])
    features_df = prep_features(features)

    training_data = labels_df.merge(features_df, on="word", how="inner")
    feature_cols = list(training_data.columns)[2:]

    x_scaler = StandardScaler().fit(training_data[feature_cols])
    training_data[feature_cols] = x_scaler.transform(training_data[feature_cols])                                               

    y_mean = training_data["rtg"].mean()
    y_std = training_data["rtg"].std(ddof=0) + 1e-8
    df = training_data.assign(
        rtg_norm = (training_data["rtg"] - y_mean) / y_std
    )

    train_kfold_and_ensemble(df, feature_cols, y_std, y_mean, x_scaler)

#################################################################################################################
#################################################################################################################
####################################          E X P E R I M E N T S          ####################################
#################################################################################################################
#################################################################################################################

def TRAIN(model, train_subset, val_subset, limit=1):

    NUM_EPOCHS = 250 * limit
    BATCH_SIZE = 64  # you can increase this now
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = torch.nn.SmoothL1Loss(beta=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6)
    model.to(device)

    best_state = None
    best_val = float("inf")
    patience = 8
    min_delta = 1e-4
    bad = 0

    with open(LOG_PATH, "w", newline = '', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames = {'epoch':'', 'train loss':'', 'val loss':''}.keys())
        writer.writeheader()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_sum, n_train = 0.0, 0

        for _, x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_sum += loss.item() * x.size(0)
            n_train += x.size(0)

        avg_train = train_sum / n_train

        # ---- validation ----
        model.eval()
        val_sum, n_val = 0.0, 0
        with torch.no_grad():
            for _, x, y in val_loader:
                x = x.to(device); y = y.to(device)
                out = model(x)
                val_sum += criterion(out, y).item() * x.size(0)
                n_val += x.size(0)
        avg_val = val_sum / n_val
        scheduler.step(avg_val)

        rows = {'epoch' : epoch, 'train loss' : avg_train, 'val loss' : avg_val}
        print(f"\rEpoch [{rows['epoch']}/{NUM_EPOCHS}] - Train loss: {rows['train loss']:.4f} | "
              f"Val loss: {rows['val loss']:.4f}", end = '')

        with open(LOG_PATH, "a", newline = '', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames = rows.keys())
            writer.writerows([rows])

        if (best_val - avg_val) > min_delta:
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return best_state

def EVAL(model, dataset, y_mean=None, y_std=None):
    model.eval()
    percent = [0, 0]
    within = [0, 0]
    total_abs_error = 0.0
    total_squared_error = 0.0

    with torch.no_grad():
        for i in tqdm.tqdm(range(len(dataset))):
            word, x, y_norm = dataset[i]
            pred_norm = model(x.unsqueeze(0)).item()
            pred = pred_norm * y_std + y_mean if (y_mean is not None and y_std is not None) else pred_norm
            y = y_norm * y_std + y_mean

            y_val = y.item()
            diff = abs(pred - y_val)

            # Accuracy by int rounding (not ideal, but useful visual)
            if abs(int(pred) - int(y_val)) <= 0.5:
                percent[0] += 1
            percent[1] += 1

            # Real-valued error bands
            if diff <= 0.5:
                within[0] += 1
            elif diff <= 1.0:
                within[1] += 1

            # Errors for MAE/RMSE
            total_abs_error += diff
            total_squared_error += diff ** 2

    n = percent[1]
    mae = total_abs_error / n
    rmse = (total_squared_error / n) ** 0.5

    print(f'Accuracy: {percent[0]/n*100:.2f}% [{percent[0]}/{n}]')
    print(f'Within 0.5: {within[0]/n*100:.2f}% [{within[0]}/{n}]')
    print(f'Within 1.0: {(within[0]+within[1])/n*100:.2f}% [{within[0]+within[1]}/{n}]')
    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')

def EXEC(model, dataset, y_mean=None, y_std=None):
    print('executing EXEC()')
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in tqdm.tqdm(range(len(dataset))):
            word, x = dataset[i]
            pred_norm = model(x.unsqueeze(0)).item()
            pred = pred_norm * y_std + y_mean if (y_mean is not None and y_std is not None) else pred_norm
            predictions.append({"word": word, "difficulty": int(pred)})

    with open(DIFFICULTIES_PATH, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)

def parse_embedding(string):
    return numpy.array([float(x) for x in string.split(",")])

def prep_features(features):
    emb = numpy.vstack(features["embedding"].map(lambda s: numpy.fromstring(s, sep=",")))
    emb_df = pandas.DataFrame(emb, columns=[f"embed_{i}" for i in range(emb.shape[1])])
    features = pandas.concat([features.drop(columns=["embedding"]), emb_df], axis=1)
    
    return features

def get_train_dataset(features_df, labels_df):
    training_data = labels_df.merge(features_df, on="word", how="inner")
    feature_cols = list(training_data.columns)[2:]

    x_scaler = StandardScaler().fit(training_data[feature_cols])
    training_data[feature_cols] = x_scaler.transform(training_data[feature_cols])                                               

    y_mean = training_data["rtg"].mean()
    y_std = training_data["rtg"].std(ddof=0) + 1e-8
    training_data = training_data.assign(
        rtg_norm = (training_data["rtg"] - y_mean) / y_std
    )

    return TrainingDataset(training_data, feature_cols, target_col="rtg_norm"), feature_cols, x_scaler.mean_, x_scaler.scale_, y_mean, y_std

def run():
    print('executing run()')
    features = pandas.read_csv(FEATURES_PATH, keep_default_na=False, na_values=[])
    labels = pandas.read_csv(LABELING_PATH, keep_default_na=False, na_values=[])
    features = prep_features(features)

    training_dataset, feature_cols, x_mean, x_scale, y_mean, y_std = get_train_dataset(features, labels)

    # split *indices* (not the DataFrame)
    indices = list(range(len(training_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_subset = Subset(training_dataset, train_idx)
    val_subset = Subset(training_dataset, val_idx)

    input_dim = training_dataset[0][1].shape[0]
    model = DifficultyRegressor(input_dim=input_dim)

    best_state = TRAIN(model, train_subset, val_subset)

    torch.save({
        "state_dict": best_state,
        "feature_cols": feature_cols,
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_std": y_std,
        "input_dim": input_dim,
        "train_dataset": training_dataset
    }, MODEL_PATH)

    EVAL(model, val_subset, y_mean=y_mean, y_std=y_std)

def evaluate():
    ckpt = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    dataset = ckpt["train_dataset"]

    model = DifficultyRegressor(input_dim=ckpt["input_dim"])
    model.load_state_dict(ckpt["state_dict"])

    EVAL(model, dataset, y_mean=ckpt["y_mean"], y_std=ckpt["y_std"])

def execute():
    features = pandas.read_csv(FEATURES_PATH, keep_default_na=False, na_values=[])
    features = prep_features(features)

    ckpt = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    feature_cols = ckpt["feature_cols"]

    x_mean = ckpt["x_mean"]
    x_scale = ckpt["x_scale"]
    features[feature_cols] = (features[feature_cols] - x_mean) / x_scale

    dataset = ExecutionDataset(features, feature_cols)

    model = DifficultyRegressor(input_dim=ckpt["input_dim"])
    model.load_state_dict(ckpt["state_dict"])

    EXEC(model, dataset, y_mean=ckpt["y_mean"], y_std=ckpt["y_std"])

#run()
#evaluate()

KFOLD_EXPERIMENT()