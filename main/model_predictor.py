#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pandas as pd
import numpy as np
import datetime
import shap
from hybrid_model import HybridModel


class Logger:
    def __init__(self, start_time):
        """
        Logger for tracking prediction process.

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


class ModelPredictor:
    def __init__(self, logger):
        """
        Predictor for making churn predictions with HybridModel.

        Args:
            logger (Logger): Logger instance for tracking progress.
        """
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self, model_path, node_feature_size, num_static_features, edge_index, cnn_filters, lstm_units,
                   gnn_layers):
        """
        Load the trained HybridModel.

        Args:
            model_path (str): Path to the saved model state.
            node_feature_size (int): Number of features per node.
            num_static_features (int): Number of static features.
            edge_index (torch.Tensor): Graph edge indices.
            cnn_filters (int): Number of CNN filters.
            lstm_units (int): Number of LSTM units.
            gnn_layers (int): Number of GNN layers.

        Returns:
            HybridModel: Loaded model.

        Raises:
            FileNotFoundError: If model file is not found.
            RuntimeError: If model loading fails.
        """
        self.logger.log(f"Загрузка модели из {model_path}...", "ModelPredictor")
        try:
            model = HybridModel(
                node_feature_size=node_feature_size,
                num_static_features=num_static_features,
                edge_index=edge_index.to(self.device),
                cnn_filters=cnn_filters,
                lstm_units=lstm_units,
                gnn_layers=gnn_layers
            ).to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            self.logger.log("Модель загружена.", "ModelPredictor")
            return model
        except FileNotFoundError:
            self.logger.log(f"Ошибка: файл модели {model_path} не найден.", "ModelPredictor")
            raise FileNotFoundError(f"Model file {model_path} not found")
        except Exception as e:
            self.logger.log(f"Ошибка при загрузке модели: {str(e)}.", "ModelPredictor")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def compute_shap_values(self, model, time_series, static_features, client_ids, node_features):
        """
        Compute SHAP values for model predictions.

        Args:
            model (HybridModel): Trained model.
            time_series (torch.Tensor): Time series data.
            static_features (torch.Tensor): Static features.
            client_ids (torch.Tensor): Client IDs.
            node_features (torch.Tensor): Node features.

        Returns:
            np.ndarray: SHAP values for each feature.
        """
        self.logger.log("Вычисление SHAP-значений для интерпретации...", "ModelPredictor")

        try:
            # Create a background dataset
            background_size = min(100, time_series.size(0))
            background_indices = np.random.choice(time_series.size(0), background_size, replace=False)
            background_data = [time_series[background_indices].to(self.device),
                               static_features[background_indices].to(self.device),
                               client_ids[background_indices].to(self.device),
                               node_features.to(self.device)]

            # Define model wrapper for SHAP
            def model_predict(inputs):
                ts, static, cids, nodes = inputs
                with torch.no_grad():
                    return torch.sigmoid(model(ts, static, cids, nodes)).cpu().numpy()

            # Initialize SHAP explainer
            explainer = shap.DeepExplainer(model_predict, background_data)
            shap_values = explainer.shap_values([time_series, static_features, client_ids, node_features])

            self.logger.log("SHAP-значения вычислены.", "ModelPredictor")
            return shap_values
        except Exception as e:
            self.logger.log(f"Ошибка при вычислении SHAP-значений: {str(e)}.", "ModelPredictor")
            raise RuntimeError(f"Failed to compute SHAP values: {str(e)}")

    def predict(self, data_file, transactions_file, events_file, model_path, num_days=90, ts_features=4, cnn_filters=64,
                lstm_units=64, gnn_layers=2):
        self.logger.log("Начало предсказания...", "ModelPredictor")

        try:
            # Load best hyperparameters
            import json
            try:
                with open('best_params.json', 'r') as f:
                    best_params = json.load(f)
                cnn_filters = best_params['cnn_filters']
                lstm_units = best_params['lstm_units']
                gnn_layers = best_params['gnn_layers']
                self.logger.log("Лучшие гиперпараметры загружены из best_params.json.", "ModelPredictor")
            except FileNotFoundError:
                self.logger.log("Предупреждение: best_params.json не найден. Используются параметры по умолчанию.",
                                "ModelPredictor")

            # Load and preprocess data
            from data_collector import DataCollector
            from preprocessor import Preprocessor
            from feature_engineering import FeatureEngineer

            dc = DataCollector(self.logger)
            df = dc.collect_data(data_file)
            self.logger.log("Данные загружены.", "ModelPredictor")

            pp = Preprocessor(self.logger)
            df = pp.preprocess(df)
            self.logger.log("Данные предобработаны.", "ModelPredictor")

            # Generate features
            fe = FeatureEngineer(self.logger, num_days=num_days, ts_features=ts_features)
            time_series, static_features, edge_index, node_features = fe.generate_features(df, transactions_file,
                                                                                           events_file)
            self.logger.log("Признаки сгенерированы.", "ModelPredictor")

            # Load model
            model = self.load_model(model_path, node_features.size(1), static_features.size(1), edge_index, cnn_filters,
                                    lstm_units, gnn_layers)

            # Perform predictions
            self.logger.log("Выполнение предсказаний с использованием CNN-LSTM-GNN...", "ModelPredictor")
            with torch.no_grad():
                time_series = time_series.to(self.device)
                static_features = static_features.to(self.device)
                client_ids = torch.tensor(df['client_id'].values, dtype=torch.long).to(self.device)
                node_features = node_features.to(self.device)
                predictions = torch.sigmoid(
                    model(time_series, static_features, client_ids, node_features)).squeeze().cpu().numpy()

            self.logger.log("Предсказания выполнены.", "ModelPredictor")

            # Compute SHAP values
            shap_values = self.compute_shap_values(model, time_series, static_features, client_ids, node_features)
            shap_df = pd.DataFrame({
                'client_id': df['client_id'],
                'shap_time_series': shap_values[0].mean(axis=(1, 2)),
                'shap_static_features': shap_values[1].mean(axis=1),
                'shap_node_features': shap_values[3].mean(axis=1)
            })
            shap_df.to_csv('shap_values.csv', index=False)
            self.logger.log("SHAP-значения сохранены в 'shap_values.csv'.", "ModelPredictor")

            # Save predictions
            result_df = pd.DataFrame({
                'client_id': df['client_id'],
                'churn_probability': predictions
            })
            result_df.to_csv('predictions.csv', index=False)
            self.logger.log("Результаты сохранены в 'predictions.csv'.", "ModelPredictor")

            self.logger.log("Предсказание завершено.", "ModelPredictor")
            return result_df

        except FileNotFoundError as e:
            self.logger.log(f"Ошибка: файл не найден: {str(e)}.", "ModelPredictor")
            raise
        except Exception as e:
            self.logger.log(f"Ошибка при выполнении предсказания: {str(e)}.", "ModelPredictor")
            raise RuntimeError(f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    logger = Logger(datetime.datetime.now())
    predictor = ModelPredictor(logger)
    predictor.predict("client_data.csv", "transactions.csv", "events.csv", "final_model.pth")


# In[ ]:




