#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import datetime


class Logger:
    def __init__(self, start_time):
        """
        Logger for tracking model operations.

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


class HybridModel(nn.Module):
    def __init__(self, node_feature_size, num_static_features, edge_index,
                 cnn_filters=64, lstm_units=64, gnn_layers=2, dropout_rate=0.3, l2_lambda=0.0001):
        """
        Hybrid model combining CNN, LSTM, and GNN for churn prediction.

        Args:
            node_feature_size (int): Number of features per graph node.
            num_static_features (int): Number of static features.
            edge_index (torch.Tensor): Graph edge indices [2, num_edges].
            cnn_filters (int): Number of CNN filters.
            lstm_units (int): Number of LSTM units.
            gnn_layers (int): Number of GNN layers.
            dropout_rate (float): Dropout rate.
            l2_lambda (float): L2 regularization coefficient.

        Raises:
            ValueError: If input parameters are invalid.
        """
        super(HybridModel, self).__init__()
        self.logger = Logger(datetime.datetime.now())
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate

        # Validate inputs
        if not isinstance(edge_index, torch.Tensor) or edge_index.shape[0] != 2:
            self.logger.log("Ошибка: некорректный формат edge_index.", "HybridModel")
            raise ValueError("edge_index must be a tensor with shape [2, num_edges]")
        if cnn_filters <= 0 or lstm_units <= 0 or gnn_layers <= 0:
            self.logger.log("Ошибка: гиперпараметры должны быть положительными.", "HybridModel")
            raise ValueError("cnn_filters, lstm_units, and gnn_layers must be positive")

        self.edge_index = edge_index

        # CNN for time series
        self.cnn = nn.Conv1d(
            in_channels=4,
            out_channels=cnn_filters,
            kernel_size=5,
            padding=2
        )
        self.cnn_bn = nn.BatchNorm1d(cnn_filters)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_units,
            batch_first=True
        )

        # GNN
        self.gnn_layers = nn.ModuleList()
        gnn_input_size = node_feature_size
        for _ in range(gnn_layers):
            self.gnn_layers.append(GCNConv(gnn_input_size, 32))
            gnn_input_size = 32

        # Fully connected layers
        self.fc_static = nn.Linear(num_static_features, 32)
        self.fc_combined = nn.Linear(lstm_units + 32 + 32, 64)
        self.fc_output = nn.Linear(64, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        self.logger.log("Модель инициализирована.", "HybridModel")

    def forward(self, time_series, static_features, client_ids, node_features):
        """
        Forward pass of the hybrid model.

        Args:
            time_series (torch.Tensor): Time series data [batch_size, num_days, ts_features].
            static_features (torch.Tensor): Static features [batch_size, num_static_features].
            client_ids (torch.Tensor): Client IDs [batch_size].
            node_features (torch.Tensor): Node features [num_clients, node_feature_size].

        Returns:
            torch.Tensor: Model logits [batch_size].

        Raises:
            ValueError: If input tensor shapes are invalid.
        """
        try:
            # Validate input shapes
            if time_series.dim() != 3 or time_series.size(2) != 4:
                self.logger.log(
                    f"Ошибка: некорректная форма time_series {time_series.shape}, ожидается [batch_size, num_days, 4].",
                    "HybridModel")
                raise ValueError("Invalid time_series shape")
            if static_features.size(1) != self.fc_static.in_features:
                self.logger.log(
                    f"Ошибка: некорректная форма static_features {static_features.shape}, ожидается [batch_size, {self.fc_static.in_features}].",
                    "HybridModel")
                raise ValueError("Invalid static_features shape")
            if client_ids.size(0) != time_series.size(0):
                self.logger.log(
                    f"Ошибка: несоответствие размеров client_ids {client_ids.shape} и time_series {time_series.size(0)}.",
                    "HybridModel")
                raise ValueError("client_ids size does not match time_series batch size")

            # CNN: [batch_size, ts_features, num_days] -> [batch_size, cnn_filters, num_days]
            cnn_out = self.cnn(time_series.transpose(1, 2))
            cnn_out = self.cnn_bn(cnn_out)
            cnn_out = F.relu(cnn_out)
            cnn_out = self.dropout(cnn_out)

            # LSTM: [batch_size, num_days, cnn_filters] -> [batch_size, lstm_units]
            lstm_input = cnn_out.transpose(1, 2)
            lstm_out, _ = self.lstm(lstm_input)
            lstm_out = lstm_out[:, -1, :]  # Take the last time step
            lstm_out = self.dropout(lstm_out)

            # GNN: [num_clients, node_feature_size] -> [num_clients, 32]
            gnn_out = node_features
            for gnn_layer in self.gnn_layers:
                gnn_out = gnn_layer(gnn_out, self.edge_index)
                gnn_out = F.relu(gnn_out)
                gnn_out = self.dropout(gnn_out)

            # Select GNN output for the batch
            gnn_out = gnn_out[client_ids]

            # Static features: [batch_size, num_static_features] -> [batch_size, 32]
            static_out = self.fc_static(static_features)
            static_out = F.relu(static_out)
            static_out = self.dropout(static_out)

            # Combine: [batch_size, lstm_units + 32 + 32]
            combined = torch.cat([lstm_out, gnn_out, static_out], dim=1)

            # Final layers: [batch_size, 64] -> [batch_size, 1]
            combined = self.fc_combined(combined)
            combined = F.relu(combined)
            combined = self.dropout(combined)
            output = self.fc_output(combined)

            return output

        except Exception as e:
            self.logger.log(f"Ошибка при прямом проходе модели: {str(e)}.", "HybridModel")
            raise RuntimeError(f"Forward pass failed: {str(e)}")

    def compute_l2_regularization(self):
        """
        Compute L2 regularization term for all model parameters.

        Returns:
            torch.Tensor: L2 regularization term.
        """
        l2_norm = sum(p.pow(2).sum() for p in self.parameters() if p.requires_grad)
        return self.l2_lambda * l2_norm


# In[ ]:




