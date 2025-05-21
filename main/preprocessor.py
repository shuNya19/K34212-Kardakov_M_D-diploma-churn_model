#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import datetime


class Logger:
    def __init__(self, start_time):
        """
        Logger for tracking preprocessing process.

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


class Preprocessor:
    def __init__(self, logger):
        """
        Preprocessor for cleaning and normalizing client data using RobustScaler.

        Args:
            logger (Logger): Logger instance for tracking progress.
        """
        self.logger = logger
        self.scaler = RobustScaler()

    def preprocess(self, df):
        """
        Clean and normalize input client data using RobustScaler.

        Args:
            df (pd.DataFrame): Input DataFrame with client data, including client_id, churn,
                              age, gender, cpi, registration_time, and time series columns.

        Returns:
            pd.DataFrame: Preprocessed DataFrame with cleaned and normalized data.

        Raises:
            ValueError: If data format is invalid.
            RuntimeError: If normalization fails due to invalid data.
        """
        if df.empty:
            self.logger.log("Ошибка: входной DataFrame пустой.", "Preprocessor")
            raise ValueError("Input DataFrame is empty")

        self.logger.log("Предобработка данных...", "Preprocessor")

        # Verify required columns
        required_columns = ['client_id', 'churn', 'age', 'gender', 'cpi']
        ts_columns = [col for col in df.columns if col.startswith(('bets_day', 'amount_day', 'wins_day', 'losses_day'))]
        missing_columns = [col for col in required_columns + ts_columns if col not in df.columns]
        if missing_columns:
            self.logger.log(f"Ошибка: отсутствуют столбцы {missing_columns}.", "Preprocessor")
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Remove rows with missing values
        initial_rows = len(df)
        df = df.dropna()
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            self.logger.log(f"Удалено {removed_rows} строк с пропущенными значениями.", "Preprocessor")

        # Validate numerical columns for finite values
        numerical_columns = ['age', 'cpi'] + ts_columns
        for col in numerical_columns:
            if not np.isfinite(df[col]).all():
                self.logger.log(f"Ошибка: столбец {col} содержит бесконечные или NaN значения после очистки.",
                                "Preprocessor")
                raise ValueError(f"Column {col} contains non-finite values")

        # Validate gender values
        valid_genders = {'male', 'female'}
        invalid_genders = set(df['gender']) - valid_genders
        if invalid_genders:
            self.logger.log(f"Ошибка: некорректные значения в столбце 'gender': {invalid_genders}.", "Preprocessor")
            raise ValueError(f"Invalid gender values: {invalid_genders}")

        # Convert gender to binary
        df['gender'] = df['gender'].map({'male': 1, 'female': 0})
        self.logger.log("Категориальный признак 'gender' преобразован.", "Preprocessor")

        # Normalize numerical features using RobustScaler
        try:
            if numerical_columns:
                # Check for sufficient variability
                for col in numerical_columns:
                    if df[col].nunique() <= 1:
                        self.logger.log(f"Ошибка: столбец {col} имеет недостаточную вариативность для нормализации.",
                                        "Preprocessor")
                        raise ValueError(f"Column {col} has insufficient variability (constant values)")

                df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
                self.logger.log("Числовые признаки нормализованы с использованием RobustScaler.", "Preprocessor")
        except Exception as e:
            self.logger.log(f"Ошибка при нормализации данных: {str(e)}.", "Preprocessor")
            raise RuntimeError(f"Normalization failed: {str(e)}")

        self.logger.log("Предобработка завершена.", "Preprocessor")
        return df


# In[ ]:




