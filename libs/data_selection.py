import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, f_classif, f_regression
from sklearn.metrics import mean_squared_error
from scipy import stats

def rfe_feature_selection(data, target, n_features_to_select=10, ranking_threshold=None, train_size=0.7,
                          return_details=False):
    """Perform Recursive Feature Elimination (RFE) for feature selection.

    Args:
        data (pandas.DataFrame): DataFrame containing features.
        target (list): List of target variables.
        n_features_to_select (int, optional): Number of features to select.
            Defaults to 10.
        ranking_threshold (float, optional): Threshold for feature ranking.
            Defaults to None.
        train_size (float, optional): Proportion of data for training.
            Defaults to 0.7.
        return_details (bool, optional): Whether to return additional details
            about selection. Defaults to False.

    Returns:
        list: List of selected features.
    """
    X = data.copy()
    y = target.copy()
    split_index = int(len(X) * train_size)
    X_train, X_test = X.iloc[:split_index, :], X.iloc[split_index:, :]
    y_train, y_test = y[:split_index], y[split_index:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    base_model = LinearRegression()
    rfe = RFE(estimator=base_model, n_features_to_select=n_features_to_select)
    rfe.fit(X_train_scaled, y_train)

    X_train_rfe = rfe.transform(X_train_scaled)
    X_test_rfe = rfe.transform(X_test_scaled)

    model = LinearRegression()
    model.fit(X_train_rfe, y_train)

    y_pred = model.predict(X_test_rfe)
    mse = mean_squared_error(y_test, y_pred)

    feature_selection_details = {'selected_features_default': np.array(X.columns)[rfe.support_],
                                 'feature_ranking': rfe.ranking_, 'model_mse': mse, 'model_coefficients': model.coef_}

    if ranking_threshold is not None:
        selected_features = np.array(X.columns)[rfe.ranking_ <= ranking_threshold]
    else:
        selected_features = np.array(X.columns)[rfe.support_]

    if return_details:
        return selected_features, feature_selection_details
    return selected_features

def anova_feature_selection_with_details(X, y, alpha=0.05, feature_names=None):
    """
    Perform ANOVA feature selection by selecting features
    with p-values less than the significance level alpha.

    Parameters:
    - X: array-like, shape (n_samples, n_features)
         Feature matrix.
    - y: array-like, shape (n_samples,)
         Target vector.
    - alpha: float
         Significance level threshold for p-values.
    - feature_names: list or array-like, optional
         Names of the features. If None, indices will be used.

    Returns:
        array: Array of selected features.
        array: Array of selected indices
        pd.DataFrame: DataFrame of additional information and selected features.
    """
    X_np = np.array(X)

    F, p_values = f_regression(X_np, y)
    selected_indices = np.where(p_values < alpha)[0]
    X_selected = X_np[:, selected_indices]

    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X_np.shape[1])]

    selection_df = pd.DataFrame({
        'Feature': feature_names,
        'F-score': F,
        'p-value': p_values,
        'Selected': p_values < alpha
    })

    return X_selected, selected_indices, selection_df