#!/usr/bin/env python
# coding: utf-8


# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import optuna
from hybrid_model import HybridModel


class Logger:
    def __init__(self, start_time):
        """
        Logger for tracking training process.

        Args:
            start_time (datetime): Starting timestamp for logging.
        """
        self.current_time = start_time

    def log(self, message, source):
        """
        Log a message with timestamp and source.

        Args:
            message (str): Message to log.
            source (str): Source module of the log.
        """
        timestamp = self.current_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [SOURCE: {source}] {message}")


class ChurnDataset(Dataset):
    def __init__(self, df, edge_index, node_features, num_days, ts_features):
        super().__init__()
        static_columns = ['age', 'gender', 'cpi', 'transaction_ratio', 'loyalty_score',
                          'weibull_lambda', 'weibull_k', 'event_activity']
        missing_cols = [col for col in static_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in df: {missing_cols}")

        self.static_features = df[static_columns].values
        self.node_features = node_features
        self.edge_index = edge_index
        self.y = df['churn'].values
        self.client_ids = df['client_id'].values
        self.client_id_to_idx = {cid: idx for idx, cid in enumerate(df['client_id'])}

        self.ts_data = []
        for feature in ts_features:
            cols = [f"{feature}_day{i+1}" for i in range(num_days)]
            missing_ts_cols = [col for col in cols if col not in df.columns]
            if missing_ts_cols:
                raise ValueError(f"Missing time series columns: {missing_ts_cols}")
            self.ts_data.append(df[cols].values)
        self.ts_data = np.stack(self.ts_data, axis=1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'static': torch.tensor(self.static_features[idx], dtype=torch.float),
            'ts': torch.tensor(self.ts_data[idx], dtype=torch.float),
            'edge_index': self.edge_index,
            'client_id': torch.tensor(self.client_id_to_idx[self.client_ids[idx]], dtype=torch.long),
            'y': torch.tensor(self.y[idx], dtype=torch.float)
        }


def compute_l2_regularization(model, l2_lambda=0.0001):
    """
    Compute L2 regularization term for all model parameters.

    Args:
        model (nn.Module): Model to compute regularization for.
        l2_lambda (float): L2 regularization coefficient.

    Returns:
        torch.Tensor: L2 regularization term.
    """
    l2_norm = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
    return l2_lambda * l2_norm




def objective(trial, train_df, val_df, edge_index, node_features, num_days, ts_features, device, logger):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    cnn_filters = trial.suggest_categorical('cnn_filters', [32, 64, 128])
    lstm_units = trial.suggest_categorical('lstm_units', [32, 64, 128])
    gnn_layers = trial.suggest_int('gnn_layers', 1, 3)

    model = HybridModel(
        node_feature_size=node_features.size(1),
        num_static_features=8,
        edge_index=edge_index.to(device),
        cnn_filters=cnn_filters,
        lstm_units=lstm_units,
        gnn_layers=gnn_layers
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor([len(train_df[train_df['churn'] == 0]) / max(1, len(train_df[train_df['churn'] == 1]))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_dataset = ChurnDataset(train_df, edge_index, node_features, num_days, ts_features)
    val_dataset = ChurnDataset(val_df, edge_index, node_features, num_days, ts_features)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_val_roc = 0
    for epoch in range(10):
        model.train()
        train_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            static, ts, edge_index_batch, y = data['static'].to(device), data['ts'].to(device), data['edge_index'].to(device), data['y'].to(device)
            output = model(ts, static, data['client_id'].to(device), node_features.to(device))
            loss = criterion(output.squeeze(), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for data in val_loader:
                static, ts, edge_index_batch, y = data['static'].to(device), data['ts'].to(device), data['edge_index'].to(device), data['y'].to(device)
                output = model(ts, static, data['client_id'].to(device), node_features.to(device))
                val_preds.extend(torch.sigmoid(output).squeeze().cpu().numpy())
                val_labels.extend(y.cpu().numpy())
        val_roc = roc_auc_score(val_labels, val_preds)
        if val_roc > best_val_roc:
            best_val_roc = val_roc

    return best_val_roc

def train_model(data_file, transactions_file, events_file, model_path, num_days=90, ts_features=4):
    logger = Logger(datetime.datetime.now())
    logger.log("Начало обучения модели...", "TrainModel")

    try:
        # Load and preprocess data
        from data_collector import DataCollector
        from preprocessor import Preprocessor
        from feature_engineering import FeatureEngineer

        dc = DataCollector(logger)
        df = dc.collect_data(data_file)
        logger.log("Данные загружены.", "TrainModel")

        # Preprocess data
        preprocessor = Preprocessor(logger=logger)
        df = preprocessor.preprocess(df)
        logger.log("Данные предобработаны.", "TrainModel")

        # Generate features
        fe = FeatureEngineer(logger=logger, num_days=num_days, ts_features=ts_features)
        time_series, static_features, edge_index, node_features = fe.generate_features(
            df, transactions_file, events_file
        )
        logger.log("Признаки сгенерированы.", "TrainModel")
        logger.log(f"Столбцы в df после generate_features: {df.columns.tolist()}", "TrainModel")

        # Split data
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        logger.log("Данные разделены на обучающую и валидационную выборки.", "TrainModel")
        logger.log(f"Столбцы в train_df: {train_df.columns.tolist()}", "TrainModel")
        logger.log(f"Столбцы в val_df: {val_df.columns.tolist()}", "TrainModel")

        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.log(f"Используемое устройство: {device}.", "TrainModel")

        # Optimize hyperparameters
        ts_features_list = ['bets', 'amount', 'wins', 'losses']
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: objective(trial, train_df, val_df, edge_index, node_features, num_days, ts_features_list,
                                    device, logger),
            n_trials=10
        )

        # Train final model with best parameters
        best_params = study.best_params
        model = HybridModel(
            node_feature_size=node_features.size(1),
            num_static_features=8,
            edge_index=edge_index.to(device),
            cnn_filters=best_params['cnn_filters'],
            lstm_units=best_params['lstm_units'],
            gnn_layers=best_params['gnn_layers']
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
        pos_weight = torch.tensor([len(train_df[train_df['churn'] == 0]) / max(1, len(train_df[train_df['churn'] == 1]))], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_dataset = ChurnDataset(train_df, edge_index, node_features, num_days, ts_features_list)
        val_dataset = ChurnDataset(val_df, edge_index, node_features, num_days, ts_features_list)
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'])

        best_val_roc = 0
        for epoch in range(50):
            model.train()
            train_loss = 0
            for data in train_loader:
                optimizer.zero_grad()
                static, ts, edge_index_batch, y = data['static'].to(device), data['ts'].to(device), data['edge_index'].to(device), data['y'].to(device)
                output = model(ts, static, data['client_id'].to(device), node_features.to(device))
                loss = criterion(output.squeeze(), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for data in val_loader:
                    static, ts, edge_index_batch, y = data['static'].to(device), data['ts'].to(device), data['edge_index'].to(device), data['y'].to(device)
                    output = model(ts, static, data['client_id'].to(device), node_features.to(device))
                    val_preds.extend(torch.sigmoid(output).squeeze().cpu().numpy())
                    val_labels.extend(y.cpu().numpy())
            val_roc = roc_auc_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, [1 if p > 0.5 else 0 for p in val_preds])
            logger.log(
                f"Эпоха {epoch + 1}, Train Loss: {train_loss:.4f}, Val ROC-AUC: {val_roc:.4f}, Val F1: {val_f1:.4f}",
                "TrainModel")

            if val_roc > best_val_roc:
                best_val_roc = val_roc
                torch.save(model.state_dict(), 'best_model.pth')

        torch.save(model.state_dict(), model_path)
        logger.log(f"Модель сохранена в {model_path}.", "TrainModel")

        # Save best hyperparameters
        with open('best_params.json', 'w') as f:
            json.dump(best_params, f)
        logger.log("Лучшие гиперпараметры сохранены в best_params.json.", "TrainModel")

    except Exception as e:
        logger.log(f"Ошибка при обучении модели: {str(e)}.", "TrainModel")
        raise RuntimeError(f"Model training failed: {str(e)}")

if __name__ == "__main__":
    train_model("client_data.csv", "transactions.csv", "events.csv", "final_model.pth")

# In[ ]:




