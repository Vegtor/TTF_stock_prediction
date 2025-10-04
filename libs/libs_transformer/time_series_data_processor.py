import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


class Time_Series_Data_Processor:
    def __init__(self, features_df, target_series, window_size=10, test_size=0.2, valid_size=0.2):
        """
        Initialize the TimeSeries_Data_Processor.

        Args:
            features_df (pd.DataFrame): DataFrame containing feature columns
            target_series (pd.Series): Series containing target values (next day closing prices)
            window_size (int): Size of the sliding window
            test_size (float): Proportion of data to use for testing
        """
        self.features = features_df
        self.target = target_series
        self.window_size = window_size
        self.test_size = test_size
        self.valid_size = valid_size
        self.feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.time_scaler = RobustScaler()

    def normalize_data(self):
        """
        Normalize both features and target data using MinMaxScaler.

        Returns:
            A tuple containing:
                (np.array): Normalized features
                (np.array): Normalized target
        """
        # Reshape target for scaling if it's 1D
        target_reshaped = self.target.values.reshape(-1, 1)

        # Fit and transform the data
        normalized_features = self.feature_scaler.fit_transform(self.features)
        normalized_target = self.target_scaler.fit_transform(target_reshaped)

        return normalized_features, normalized_target.flatten()

    def create_windows(self, features, target):
        """
        Create sliding windows from the normalized data.

        Args:
            features (np.array): Normalized feature data
            target (np.array): Normalized target data

        Returns:
             A tuple containing:
                (np.array): Windowed features (num_samples, window_size, num_features)
                (np.array): Windowed target (num_samples, 1)
        """
        X, y = [], []

        for i in range(len(features) - self.window_size):
            # Create feature window
            feature_window = features[i:(i + self.window_size)]

            # Get the target (next day's price)
            target_value = target[i + self.window_size]

            X.append(feature_window)
            y.append(target_value)

        return np.array(X), np.array(y)

    def create_windows_multistep(self, features, target, forecast_horizon):
        """
        Create sliding windows for multistep time series forecasting.

        Args:
            features (np.array): Normalized feature data
            target (np.array): Normalized target
            forecast_horizon (int): Number of future steps to predict

        Returns:
            Tuple containing arrays of windowed features and corresponding targets.

            Structure:
                X: (num_samples, window_size, num_features)
                y: (num_samples, forecast_horizon)
        """
        X, y = [], []

        total_length = len(features)
        max_start = total_length - self.window_size - forecast_horizon + 1

        for i in range(max_start):
            feature_window = features[i: i + self.window_size]

            target_seq = target[i + self.window_size: i + self.window_size + forecast_horizon]

            X.append(feature_window)
            y.append(target_seq)

        return np.array(X), np.array(y)

    def create_data_loaders(self, X, y, batch_size=32):
        """
        Splits the data into training, validation, and test sets, and returns DataLoaders for each.

        Args:
            X (np.ndarray or torch.Tensor): Feature data.
            y (np.ndarray or torch.Tensor): Target labels.
            batch_size (int): Number of samples per batch.

        Returns:
            tuple containing:
                (tensor): Training data loaders
                (tensor): Validation data loaders
                (tensor): Test data loaders
        """
        split_train_idx = int(len(X) * (1 - self.valid_size - self.test_size))
        split_test_idx = int(len(X) * (1 - self.test_size))

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        train_dataset = TensorDataset(
            X_tensor[:split_train_idx],
            y_tensor[:split_train_idx]
        )

        valid_dataset = TensorDataset(
            X_tensor[split_train_idx:split_test_idx],
            y_tensor[split_train_idx:split_test_idx]
        )

        test_dataset = TensorDataset(
            X_tensor[split_test_idx:],
            y_tensor[split_test_idx:]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, valid_loader, test_loader