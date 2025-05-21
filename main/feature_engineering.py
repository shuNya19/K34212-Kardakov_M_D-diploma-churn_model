import pandas as pd
import numpy as np
import torch
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix, degree, remove_isolated_nodes
from scipy.stats import weibull_min
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import datetime
import itertools


class Logger:
    def __init__(self, start_time):
        """
        Logger for tracking feature engineering process.

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


class FeatureEngineer:
    def __init__(self, logger, num_days=90, ts_features=4):
        """
        Feature engineer for generating time series, static, and graph features.

        Args:
            logger (Logger): Logger instance for tracking progress.
            num_days (int): Number of days in time series data.
            ts_features (int): Number of time series features per day.
        """
        self.logger = logger
        self.epsilon = 1e-6  # Small constant to prevent division by zero
        self.num_days = num_days
        self.ts_features = ts_features

    def compute_time_intervals(self, transactions_df):
        """
        Compute average time interval between transactions for each client (formula 3.2).

        Args:
            transactions_df (pd.DataFrame): DataFrame with columns 'client_id' and 'timestamp'.

        Returns:
            pd.Series: Average time intervals (in days) indexed by client_id.

        Raises:
            ValueError: If transactions_df is missing required columns.
        """
        self.logger.log("Вычисление среднего интервала между транзакциями...", "FeatureEngineer")

        required_columns = ['client_id', 'timestamp']
        missing_columns = [col for col in required_columns if col not in transactions_df.columns]
        if missing_columns:
            self.logger.log(f"Ошибка: отсутствуют столбцы {missing_columns} в transactions_df.", "FeatureEngineer")
            raise ValueError(f"Missing columns in transactions_df: {missing_columns}")

        try:
            # Convert timestamps to datetime
            transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])

            # Sort by client_id and timestamp
            transactions_df = transactions_df.sort_values(['client_id', 'timestamp'])

            # Compute differences between consecutive timestamps
            time_intervals = transactions_df.groupby('client_id')['timestamp'].diff().dt.total_seconds() / (
                        24 * 3600)  # Convert to days
            time_intervals = time_intervals.groupby(transactions_df['client_id']).mean()

            self.logger.log("Средние интервалы между транзакциями вычислены.", "FeatureEngineer")
            return time_intervals
        except Exception as e:
            self.logger.log(f"Ошибка при вычислении интервалов: {str(e)}.", "FeatureEngineer")
            raise RuntimeError(f"Failed to compute time intervals: {str(e)}")

    def optimize_loyalty_weights(self, df, time_intervals):
        """
        Optimize weights for loyalty score using cross-validation.

        Args:
            df (pd.DataFrame): Preprocessed DataFrame with client data.
            time_intervals (pd.Series): Average time intervals between transactions.

        Returns:
            tuple: Optimized weights (w1, w2, w3).
        """
        self.logger.log("Оптимизация весов для показателя лояльности...", "FeatureEngineer")

        # Compute components
        amount_columns = [col for col in df.columns if col.startswith('amount_day')]
        mean_transaction = df[amount_columns].mean(axis=1)
        registration_time = (pd.to_datetime('2025-05-20') - pd.to_datetime(df['registration_time'])).dt.days

        # Prepare data
        X = pd.DataFrame({
            'mean_transaction': mean_transaction,
            'inv_interval': 1 / (time_intervals.reindex(df['client_id']).fillna(1) + self.epsilon),
            'registration_time': registration_time
        })
        y = df['churn'].values

        # Check for NaN or invalid data
        if X.isna().any().any() or np.any(~np.isfinite(X.values)):
            self.logger.log("Предупреждение: Пропуски или некорректные данные в X. Заполнение медианой.",
                            "FeatureEngineer")
            X = X.fillna(X.median())
        if np.any(y == None) or np.any(~np.isfinite(y)):
            self.logger.log("Ошибка: Некорректные значения в y.", "FeatureEngineer")
            raise ValueError("Invalid values in churn labels")

        # Grid search for weights
        weight_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        weight_combinations = list(itertools.product(weight_values, repeat=3))
        weight_combinations = [w for w in weight_combinations if abs(sum(w) - 1.0) < 1e-3]

        if not weight_combinations:
            self.logger.log(
                "Предупреждение: Нет комбинаций весов с суммой 1.0. Используются веса по умолчанию (1/3, 1/3, 1/3).",
                "FeatureEngineer")
            return (1 / 3, 1 / 3, 1 / 3)

        best_weights = None
        best_score = -1

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        for weights in weight_combinations:
            w1, w2, w3 = weights
            scores = []
            try:
                for train_idx, val_idx in kf.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    # Compute loyalty score
                    loyalty = (w1 * X_train['mean_transaction'] +
                               w2 * X_train['inv_interval'] +
                               w3 * X_train['registration_time']).values
                    loyalty_val = (w1 * X_val['mean_transaction'] +
                                   w2 * X_val['inv_interval'] +
                                   w3 * X_val['registration_time']).values

                    # Simple logistic regression for scoring
                    clf = LogisticRegression().fit(loyalty.reshape(-1, 1), y_train)
                    pred = clf.predict_proba(loyalty_val.reshape(-1, 1))[:, 1]
                    score = roc_auc_score(y_val, pred)
                    scores.append(score)
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_weights = weights
            except Exception as e:
                self.logger.log(f"Предупреждение: Ошибка для весов {weights}: {str(e)}. Пропуск комбинации.",
                                "FeatureEngineer")
                continue

        if best_weights is None:
            self.logger.log(
                "Предупреждение: Не удалось найти оптимальные веса. Используются веса по умолчанию (1/3, 1/3, 1/3).",
                "FeatureEngineer")
            return (1 / 3, 1 / 3, 1 / 3)

        self.logger.log(
            f"Оптимизированные веса: w1={best_weights[0]}, w2={best_weights[1]}, w3={best_weights[2]} (ROC-AUC: {best_score:.4f}).",
            "FeatureEngineer")
        return best_weights

    def compute_interaction_features(self, df, time_intervals):
        self.logger.log("Генерация взаимодействующих признаков...", "FeatureEngineer")

        try:
            # Compute mean transaction amount (S_i)
            amount_columns = [col for col in df.columns if col.startswith('amount_day')]
            if not amount_columns:
                self.logger.log("Ошибка: отсутствуют столбцы amount_dayX.", "FeatureEngineer")
                raise ValueError("No amount_dayX columns found")
            mean_transaction = df[amount_columns].mean(axis=1)

            # Compute time since last activity (tau_i)
            bets_columns = [col for col in df.columns if col.startswith('bets_day')]
            if not bets_columns:
                self.logger.log("Ошибка: отсутствуют столбцы bets_dayX.", "FeatureEngineer")
                raise ValueError("No bets_dayX columns found")
            last_activity = df[bets_columns].apply(
                lambda x: self.num_days - np.argmax(x[::-1] > 0) if any(x > 0) else self.num_days, axis=1
            )

            # Transaction-to-inactivity ratio
            transaction_ratio = mean_transaction / (last_activity + self.epsilon)

            # Loyalty score with optimized weights
            w1, w2, w3 = self.optimize_loyalty_weights(df, time_intervals)
            registration_time = (pd.to_datetime('2025-05-20') - pd.to_datetime(df['registration_time'])).dt.days
            loyalty_score = (
                    w1 * mean_transaction +
                    w2 * 1 / (time_intervals.reindex(df['client_id']).fillna(1) + self.epsilon) +
                    w3 * registration_time
            )

            # Add new columns using pd.concat
            new_columns = pd.DataFrame({
                'transaction_ratio': transaction_ratio,
                'loyalty_score': loyalty_score
            }, index=df.index)
            df = pd.concat([df, new_columns], axis=1)

            self.logger.log("Взаимодействующие признаки сгенерированы.", "FeatureEngineer")
            return df
        except Exception as e:
            self.logger.log(f"Ошибка при генерации взаимодействующих признаков: {str(e)}.", "FeatureEngineer")
            raise RuntimeError(f"Failed to compute interaction features: {str(e)}")

    def apply_nonlinear_transformations(self, df):
        """
        Apply logarithmic transformation to skewed features (formula 3.5).

        Args:
            df (pd.DataFrame): DataFrame with numerical features.

        Returns:
            pd.DataFrame: DataFrame with transformed columns.

        Raises:
            ValueError: If numerical columns are missing.
        """
        self.logger.log("Применение нелинейных преобразований признаков...", "FeatureEngineer")

        try:
            feature_columns = [col for col in df.columns if
                               col.startswith(('bets_day', 'amount_day', 'wins_day', 'losses_day'))] + ['cpi']
            if not feature_columns:
                self.logger.log("Ошибка: отсутствуют числовые столбцы для преобразования.", "FeatureEngineer")
                raise ValueError("No numerical columns found for transformation")

            for col in feature_columns:
                df[col] = np.log1p(df[col].clip(lower=0))  # Ensure non-negative values

            self.logger.log("Нелинейные преобразования применены.", "FeatureEngineer")
            return df
        except Exception as e:
            self.logger.log(f"Ошибка при применении нелинейных преобразований: {str(e)}.", "FeatureEngineer")
            raise RuntimeError(f"Failed to apply nonlinear transformations: {str(e)}")

    def compute_interaction_features(self, df, time_intervals):
        self.logger.log("Генерация взаимодействующих признаков...", "FeatureEngineer")

        try:
            # Compute mean transaction amount
            amount_columns = [col for col in df.columns if col.startswith('amount_day')]
            if not amount_columns:
                self.logger.log("Ошибка: отсутствуют столбцы amount_dayX.", "FeatureEngineer")
                raise ValueError("No amount_dayX columns found")
            mean_transaction = df[amount_columns].mean(axis=1)

            # Compute time since last activity
            bets_columns = [col for col in df.columns if col.startswith('bets_day')]
            if not bets_columns:
                self.logger.log("Ошибка: отсутствуют столбцы bets_dayX.", "FeatureEngineer")
                raise ValueError("No bets_dayX columns found")
            last_activity = df[bets_columns].apply(
                lambda x: self.num_days - np.argmax(x[::-1] > 0) if any(x > 0) else self.num_days, axis=1
            )

            # Transaction-to-inactivity ratio
            transaction_ratio = mean_transaction / (last_activity + self.epsilon)

            # Loyalty score with optimized weights
            w1, w2, w3 = self.optimize_loyalty_weights(df, time_intervals)
            registration_time = (pd.to_datetime('2025-05-20') - pd.to_datetime(df['registration_time'])).dt.days
            loyalty_score = (
                    w1 * mean_transaction +
                    w2 * 1 / (time_intervals.reindex(df['client_id']).fillna(1) + self.epsilon) +
                    w3 * registration_time
            )

            new_columns = pd.DataFrame({
                'transaction_ratio': transaction_ratio,
                'loyalty_score': loyalty_score
            }, index=df.index)
            df = pd.concat([df, new_columns], axis=1)

            self.logger.log("Взаимодействующие признаки сгенерированы.", "FeatureEngineer")
            return df
        except Exception as e:
            self.logger.log(f"Ошибка при генерации взаимодействующих признаков: {str(e)}.", "FeatureEngineer")
            raise RuntimeError(f"Failed to compute interaction features: {str(e)}")

    def compute_weibull_features(self, df, time_intervals, events_df):
        self.logger.log("Вычисление признаков на основе распределения Вейбулла...", "FeatureEngineer")

        try:
            registration_time = (pd.to_datetime('2025-05-20') - pd.to_datetime(df['registration_time'])).dt.days
            weibull_lambda = registration_time / (time_intervals.reindex(df['client_id']).fillna(1) + self.epsilon)
            weibull_k = np.log(registration_time + self.epsilon)

            events_per_client = events_df.groupby('client_id').size().reindex(df['client_id']).fillna(0)
            event_activity = events_per_client / (registration_time + self.epsilon)

            new_columns = pd.DataFrame({
                'weibull_lambda': weibull_lambda,
                'weibull_k': weibull_k,
                'event_activity': event_activity
            }, index=df.index)
            df = pd.concat([df, new_columns], axis=1)

            self.logger.log("Признаки Вейбулла и событийной активности сгенерированы.", "FeatureEngineer")
            return df
        except Exception as e:
            self.logger.log(f"Ошибка при вычислении признаков Вейбулла: {str(e)}.", "FeatureEngineer")
            raise RuntimeError(f"Failed to compute Weibull features: {str(e)}")

    def compute_graph_metrics(self, df, transactions_df):
        self.logger.log("Вычисление метрик схожести для графа...", "FeatureEngineer")

        try:
            bets_columns = [col for col in df.columns if col.startswith('bets_day')]
            bets_data = df[bets_columns].fillna(0).values
            if bets_data.var(axis=1).sum() == 0:
                self.logger.log("Предупреждение: Нулевая дисперсия в bets_data. Используется нулевая матрица.",
                                "FeatureEngineer")
                S_bets = np.zeros((len(df), len(df)))
            else:
                S_bets = np.corrcoef(bets_data)
                S_bets = np.nan_to_num(S_bets, nan=0, posinf=0, neginf=0)

            promo_counts = transactions_df.groupby('client_id')['promo_used'].sum().reindex(df['client_id']).fillna(
                0).values
            if promo_counts.var() == 0:
                self.logger.log("Предупреждение: Нулевая дисперсия в promo_counts. Используется нулевая матрица.",
                                "FeatureEngineer")
                S_promo = np.zeros((len(df), len(df)))
            else:
                S_promo = np.corrcoef(promo_counts.reshape(-1, 1))
                S_promo = np.nan_to_num(S_promo, nan=0, posinf=0, neginf=0)

            time_intervals = self.compute_time_intervals(transactions_df)
            time_intervals = time_intervals.reindex(df['client_id']).fillna(1).values
            if time_intervals.var() == 0:
                self.logger.log("Предупреждение: Нулевая дисперсия в time_intervals. Используется нулевая матрица.",
                                "FeatureEngineer")
                S_time = np.zeros((len(df), len(df)))
            else:
                S_time = np.corrcoef(time_intervals.reshape(-1, 1))
                S_time = np.nan_to_num(S_time, nan=0, posinf=0, neginf=0)

            self.logger.log("Метрики схожести для графа вычислены.", "FeatureEngineer")
            return S_bets, S_promo, S_time
        except Exception as e:
            self.logger.log(f"Ошибка при вычислении метрик графа: {str(e)}.", "FeatureEngineer")
            raise RuntimeError(f"Failed to compute graph metrics: {str(e)}")

    def optimize_graph_params(self, df, S_bets, S_promo, S_time, initial_edge_index, node_features):
        self.logger.log("Оптимизация весов alpha и порога схожести для графа...", "FeatureEngineer")

        for name, S in [('S_bets', S_bets), ('S_promo', S_promo), ('S_time', S_time)]:
            if np.any(np.isnan(S)) or np.any(~np.isfinite(S)):
                self.logger.log(f"Предупреждение: {name} содержит NaN или бесконечные значения. Заполнение нулями.",
                                "FeatureEngineer")
                S[np.isnan(S) | ~np.isfinite(S)] = 0

        y = df['churn'].values
        if len(y) != node_features.shape[0]:
            self.logger.log(f"Ошибка: Несоответствие размеров y ({len(y)}) и node_features ({node_features.shape[0]}).",
                            "FeatureEngineer")
            return (1 / 3, 1 / 3, 1 / 3, 0.7)

        alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        threshold_values = [0.5, 0.7, 0.9]
        alpha_combinations = list(itertools.product(alpha_values, repeat=3))
        alpha_combinations = [alpha for alpha in alpha_combinations if abs(sum(alpha) - 1.0) < 1e-3]

        self.logger.log(f"Количество комбинаций alpha: {len(alpha_combinations)}", "FeatureEngineer")
        if not alpha_combinations:
            self.logger.log("Предупреждение: Нет комбинаций alpha. Используются параметры по умолчанию.",
                            "FeatureEngineer")
            return (1 / 3, 1 / 3, 1 / 3, 0.7)

        best_params = None
        best_score = -1
        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        for alpha in alpha_combinations:
            alpha1, alpha2, alpha3 = alpha
            for threshold in threshold_values:
                try:
                    w = alpha1 * S_bets + alpha2 * S_promo + alpha3 * S_time
                    adj_matrix = (w > threshold).astype(int)
                    edge_index = torch.tensor(
                        np.array([[i, j] for i, j in zip(*np.where(adj_matrix > 0))]).T,
                        dtype=torch.long
                    )
                    if edge_index.shape[1] == 0:
                        self.logger.log(
                            f"Предупреждение: Пустой граф для alpha={alpha}, threshold={threshold}. Пропуск.",
                            "FeatureEngineer")
                        continue

                    scores = []
                    for train_idx, val_idx in kf.split(node_features):
                        if len(train_idx) == 0 or len(val_idx) == 0:
                            self.logger.log(
                                f"Предупреждение: Пустые индексы для alpha={alpha}, threshold={threshold}. Пропуск.",
                                "FeatureEngineer")
                            continue
                        X_train = node_features[train_idx].cpu().numpy()
                        X_val = node_features[val_idx].cpu().numpy()
                        y_train = y[train_idx]
                        y_val = y[val_idx]

                        if np.any(np.isnan(X_train)) or np.any(np.isnan(X_val)):
                            self.logger.log(
                                f"Предупреждение: NaN в данных для alpha={alpha}, threshold={threshold}. Пропуск.",
                                "FeatureEngineer")
                            continue

                        clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
                        pred = clf.predict_proba(X_val)[:, 1]
                        score = roc_auc_score(y_val, pred)
                        scores.append(score)

                    if scores:
                        avg_score = np.mean(scores)
                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = (alpha1, alpha2, alpha3, threshold)
                except Exception as e:
                    self.logger.log(
                        f"Предупреждение: Ошибка для alpha={alpha}, threshold={threshold}: {str(e)}. Пропуск.",
                        "FeatureEngineer")
                    continue

        if best_params is None:
            self.logger.log(
                "Предупреждение: Не удалось найти оптимальные параметры. Используются параметры по умолчанию.",
                "FeatureEngineer")
            return (1 / 3, 1 / 3, 1 / 3, 0.7)

        self.logger.log(
            f"Оптимизированные параметры: alpha1={best_params[0]}, alpha2={best_params[1]}, alpha3={best_params[2]}, threshold={best_params[3]} (ROC-AUC: {best_score:.4f}).",
            "FeatureEngineer")
        return best_params

    def limit_degree(self, adj_matrix, max_degree=10):
        """
        Limit the degree of nodes in the adjacency matrix to max_degree.

        Args:
            adj_matrix (np.ndarray or torch.Tensor): Adjacency matrix.
            max_degree (int): Maximum degree for each node.

        Returns:
            np.ndarray: Adjusted adjacency matrix.
        """
        self.logger.log(f"Ограничение степени вершин до {max_degree}...", "FeatureEngineer")
        try:
            if isinstance(adj_matrix, torch.Tensor):
                adj_matrix = adj_matrix.cpu().numpy()
            elif not isinstance(adj_matrix, np.ndarray):
                raise ValueError(f"adj_matrix must be np.ndarray or torch.Tensor, got {type(adj_matrix)}")

            self.logger.log(
                f"adj_matrix type: {type(adj_matrix)}, shape: {adj_matrix.shape}, dtype: {adj_matrix.dtype}",
                "FeatureEngineer")

            adj_matrix = np.where(np.isnan(adj_matrix), 0, adj_matrix)
            adj_matrix = (adj_matrix > 0).astype(int)

            degrees = np.sum(adj_matrix, axis=1)
            rows, cols = np.where(adj_matrix > 0)
            keep_edges = np.ones(len(rows), dtype=bool)

            for i in range(len(degrees)):
                if degrees[i] > max_degree:
                    node_edges = (rows == i) | (cols == i)
                    edge_indices = np.where(node_edges)[0].astype(int)
                    if len(edge_indices) > max_degree:
                        np.random.shuffle(edge_indices)
                        sorted_indices = edge_indices[max_degree:]
                        keep_edges[sorted_indices] = False

            new_rows = rows[keep_edges]
            new_cols = cols[keep_edges]
            new_adj_matrix = np.zeros_like(adj_matrix)
            new_adj_matrix[new_rows, new_cols] = 1
            new_adj_matrix[new_cols, new_rows] = 1
            self.logger.log(f"New adj_matrix non-zero count: {np.sum(new_adj_matrix)}", "FeatureEngineer")
            return new_adj_matrix
        except Exception as e:
            self.logger.log(f"Ошибка при ограничении степени вершин: {str(e)}.", "FeatureEngineer")
            raise RuntimeError(f"Failed to limit node degree: {str(e)}")

    def generate_graph_data(self, df, transactions_df):
        """
        Generate graph data (edge_index and node_features) for the model.
        """
        self.logger.log("Генерация графовых данных...", "FeatureEngineer")
        try:
            S_bets, S_promo, S_time = self.compute_graph_metrics(df, transactions_df)
            self.logger.log(
                f"S_bets variance: {np.var(S_bets)}, S_promo variance: {np.var(S_promo)}, S_time variance: {np.var(S_time)}",
                "FeatureEngineer")
            initial_edge_index, node_features = self.create_initial_graph(df)
            alpha1, alpha2, alpha3, threshold = self.optimize_graph_params(
                df, S_bets, S_promo, S_time, initial_edge_index, node_features
            )
            # Ensure w is a NumPy array
            w = alpha1 * S_bets + alpha2 * S_promo + alpha3 * S_time
            if isinstance(w, torch.Tensor):
                w = w.cpu().numpy()
            self.logger.log(f"w type: {type(w)}, shape: {w.shape}, dtype: {w.dtype}, max: {w.max()}, min: {w.min()}",
                            "FeatureEngineer")

            # Lower default threshold if graph is empty
            adj_matrix = (w > threshold).astype(int)
            if np.sum(adj_matrix) == 0:
                self.logger.log(f"Предупреждение: Пустой граф с threshold={threshold}. Понижение до 0.3.",
                                "FeatureEngineer")
                adj_matrix = (w > 0.3).astype(int)

            adj_matrix = self.limit_degree(adj_matrix, max_degree=10)
            edge_index = torch.tensor(
                np.array([[i, j] for i, j in zip(*np.where(adj_matrix > 0))]).T,
                dtype=torch.long
            )
            self.logger.log(f"Final edge_index shape: {edge_index.shape}", "FeatureEngineer")
            return edge_index, node_features
        except Exception as e:
            self.logger.log(f"Ошибка при генерации графовых данных: {str(e)}.", "FeatureEngineer")
            raise RuntimeError(f"Failed to generate graph data: {str(e)}")

    def create_initial_graph(self, df):
        """
        Create an initial graph and node features based on client data.

        Args:
            df (pd.DataFrame): Client data with features like bets, age, gender, etc.

        Returns:
            tuple: (initial_edge_index, node_features)
                - initial_edge_index (torch.Tensor): Edge indices for the initial graph.
                - node_features (torch.Tensor): Node features for the graph.
        """
        self.logger.log("Создание начального графа...", "FeatureEngineer")
        try:
            # Node features: combine static features and aggregated time series
            static_features = df[['age', 'gender', 'cpi']].values
            bets_columns = [col for col in df.columns if col.startswith('bets_day')]
            amount_columns = [col for col in df.columns if col.startswith('amount_day')]

            # Aggregate time series features (mean bets and amounts)
            bets_mean = df[bets_columns].mean(axis=1).values.reshape(-1, 1)
            amount_mean = df[amount_columns].mean(axis=1).values.reshape(-1, 1)

            # Combine features
            node_features = np.hstack([static_features, bets_mean, amount_mean])

            # Handle NaN values
            node_features = np.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)
            node_features = torch.tensor(node_features, dtype=torch.float)

            self.logger.log(f"node_features shape: {node_features.shape}, dtype: {node_features.dtype}",
                            "FeatureEngineer")

            # Initial graph: create edges based on similarity of bets
            bets_data = df[bets_columns].fillna(0).values
            if bets_data.var(axis=1).sum() == 0:
                self.logger.log("Предупреждение: Нулевая дисперсия в bets_data. Создаётся пустой граф.",
                                "FeatureEngineer")
                initial_edge_index = torch.tensor([[], []], dtype=torch.long)
            else:
                # Compute correlation matrix for bets
                S_bets = np.corrcoef(bets_data)
                S_bets = np.nan_to_num(S_bets, nan=0.0, posinf=0.0, neginf=0.0)

                # Create edges where correlation > 0.5
                threshold = 0.5
                rows, cols = np.where(S_bets > threshold)
                edges = np.array([rows, cols])
                # Remove self-loops
                mask = rows != cols
                edges = edges[:, mask]
                initial_edge_index = torch.tensor(edges, dtype=torch.long)

            self.logger.log(f"initial_edge_index shape: {initial_edge_index.shape}", "FeatureEngineer")
            return initial_edge_index, node_features
        except Exception as e:
            self.logger.log(f"Ошибка при создании начального графа: {str(e)}.", "FeatureEngineer")
            raise RuntimeError(f"Failed to create initial graph: {str(e)}")

    def generate_features(self, df, transactions_file, events_file):
        self.logger.log(
            "Генерация признаков (временные ряды, графы, статические данные, дополнительные признаки)...",
            "FeatureEngineer"
        )

        try:
            # Validate input data
            required_columns = ['client_id', 'churn', 'age', 'gender', 'cpi', 'registration_time']
            ts_columns = [col for col in df.columns if
                          col.startswith(('bets_day', 'amount_day', 'wins_day', 'losses_day'))]
            if not all(col in df.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in df.columns]
                self.logger.log(f"Ошибка: отсутствуют обязательные столбцы: {missing_cols}.", "FeatureEngineer")
                raise ValueError(f"Missing required columns: {missing_cols}")
            if len(ts_columns) != self.num_days * self.ts_features:
                self.logger.log(
                    f"Ошибка: ожидалось {self.num_days * self.ts_features} столбцов временных рядов, найдено {len(ts_columns)}.",
                    "FeatureEngineer"
                )
                raise ValueError("Incorrect number of time series columns")

            # Compute time intervals
            transactions_df = pd.read_csv(transactions_file)
            time_intervals = self.compute_time_intervals(transactions_df)

            # Generate static and interaction features
            df = self.compute_interaction_features(df, time_intervals)
            self.logger.log(f"Столбцы после compute_interaction_features: {df.columns.tolist()}", "FeatureEngineer")

            # Generate Weibull and event features
            events_df = pd.read_csv(events_file)
            df = self.compute_weibull_features(df, time_intervals, events_df)
            self.logger.log(f"Столбцы после compute_weibull_features: {df.columns.tolist()}", "FeatureEngineer")

            # Generate time series tensor
            time_series = []
            for feature in ['bets', 'amount', 'wins', 'losses']:
                cols = [f"{feature}_day{i + 1}" for i in range(self.num_days)]
                feature_data = df[cols].values
                time_series.append(feature_data)
            time_series = np.stack(time_series, axis=1)
            time_series = torch.tensor(time_series, dtype=torch.float32)

            # Generate static features tensor
            static_columns = ['age', 'gender', 'cpi', 'transaction_ratio', 'loyalty_score',
                              'weibull_lambda', 'weibull_k', 'event_activity']
            missing_static_cols = [col for col in static_columns if col not in df.columns]
            if missing_static_cols:
                self.logger.log(f"Ошибка: отсутствуют статические столбцы: {missing_static_cols}.", "FeatureEngineer")
                raise ValueError(f"Missing static columns: {missing_static_cols}")
            static_features = torch.tensor(df[static_columns].values, dtype=torch.float32)
            self.logger.log(f"Типы данных static_features: {df[static_columns].dtypes}", "FeatureEngineer")

            # Generate graph data
            edge_index, node_features = self.generate_graph_data(df, transactions_df)
            self.logger.log("Признаки сгенерированы.", "FeatureEngineer")
            return time_series, static_features, edge_index, node_features

        except FileNotFoundError as e:
            self.logger.log(f"Ошибка: файл не найден: {str(e)}.", "FeatureEngineer")
            raise
        except Exception as e:
            self.logger.log(f"Ошибка при генерации признаков: {str(e)}.", "FeatureEngineer")
            raise RuntimeError(f"Failed to generate features: {str(e)}")