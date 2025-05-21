#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import datetime


class Logger:
    def __init__(self, start_time):
        """
        Logger for tracking data collection process.

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


class DataCollector:
    def __init__(self, logger):
        """
        Data collector for gathering client data from a CSV file.

        Args:
            logger (Logger): Logger instance for tracking progress.
        """
        self.logger = logger

    def collect_data(self, data_file):
        """
        Load client data from a CSV file with format validation.

        Args:
            data_file (str): Path to the CSV file with client data.

        Returns:
            pd.DataFrame: DataFrame containing client data with columns for client_id, churn,
                          age, gender, cpi, registration_time, and time series (bets_dayX, amount_dayX, wins_dayX, losses_dayX).

        Raises:
            FileNotFoundError: If the specified CSV file is not found.
            ValueError: If required columns are missing or data format is invalid.
        """
        self.logger.log(f"Загрузка данных из {data_file}...", "DataCollector")

        try:
            # Load data from CSV
            df = pd.read_csv(data_file)
        except FileNotFoundError:
            self.logger.log(f"Ошибка: файл {data_file} не найден.", "DataCollector")
            raise FileNotFoundError(f"CSV file {data_file} not found")

        # Verify required columns
        required_columns = ['client_id', 'churn', 'age', 'gender', 'cpi', 'registration_time']
        ts_columns = [f"{prefix}_day{i}" for i in range(1, 91) for prefix in ['bets', 'amount', 'wins', 'losses']]
        required_columns.extend(ts_columns)

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.logger.log(f"Ошибка: отсутствуют столбцы {missing_columns} в файле {data_file}.", "DataCollector")
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Validate data types for numerical columns
        numerical_columns = ['age', 'cpi'] + ts_columns
        initial_rows = len(df)
        for col in numerical_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Check for invalid gender values
        valid_genders = {'male', 'female'}
        invalid_genders = set(df['gender']) - valid_genders
        if invalid_genders:
            self.logger.log(f"Ошибка: некорректные значения в столбце 'gender': {invalid_genders}.", "DataCollector")
            raise ValueError(f"Invalid gender values: {invalid_genders}")

        # Remove rows with NaN in numerical columns
        df = df.dropna(subset=numerical_columns)
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            self.logger.log(f"Удалено {removed_rows} строк с нечисловыми значениями в числовых столбцах.",
                            "DataCollector")

        # Validate client_id and churn
        if df['client_id'].duplicated().any():
            self.logger.log("Ошибка: обнаружены дубликаты в столбце 'client_id'.", "DataCollector")
            raise ValueError("Duplicate client_id values found")
        if not df['churn'].isin([0, 1]).all():
            self.logger.log("Ошибка: некорректные значения в столбце 'churn'.", "DataCollector")
            raise ValueError("Invalid churn values (must be 0 or 1)")

        self.logger.log("Данные успешно загружены.", "DataCollector")
        return df

# In[ ]:




