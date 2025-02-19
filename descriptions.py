# descriptions.py

FORECAST_DESCRIPTIONS = {
    "prophet": (
        "This module uses Facebook's Prophet library for forecasting, modeling trends, "
        "weekly/yearly seasonality, and holiday effects. The CSV is read and columns like "
        "day-of-week (dow), month-of-year (moy), and holiday indicators are used as regressors. "
        "We train the model, forecast 30 days ahead, and compute MAPE over the last six months."
    ),
    "lstm": (
        "A base LSTM forecast model that reads the time series data, scales it, and trains a simple "
        "univariate or multivariate LSTM neural network. It then makes predictions for a specified horizon. "
        "We display the forecast plot along with the MAPE metric."
    ),
    "lstm_multiservice": (
        "This LSTM module is designed for data with multiple 'services' or categories. "
        "It preprocesses the dataset (including day-of-week and service codes), builds an LSTM network, "
        "and forecasts future orders for each service. The MAPE metric quantifies forecast accuracy."
    ),
    "lstm_multiservice_aggregate": (
        "This approach aggregates data by day and service, then trains an LSTM model. "
        "It forecasts future orders for each service category and plots the results. "
        "Useful when you have multiple services in the same dataset."
    ),
    "holtwinters": (
        "A Holt-Winters (or SARIMAX proxy) approach to forecasting, capturing level, trend, and seasonality. "
        "It trains on 80% of the data, forecasts on the remaining portion, and computes MAPE. "
        "The final plot shows historical data, the test set predictions, and a future 28-day forecast."
    ),
    "lstm_univariate": (
        "A univariate LSTM model that uses only the main time series (e.g., orders_count) as input. "
        "It transforms the data, builds an LSTM, and forecasts the next few steps, displaying a forecast plot "
        "and MAPE for validation."
    ),
    "lstm_multivariate": (
        "A multivariate LSTM that includes additional features (day-of-week, month, holiday indicators, etc.). "
        "This approach can improve accuracy by providing more context to the model. "
        "The forecast horizon and MAPE are displayed in the resulting plot."
    ),
    "lstm_multivariate_interval": (
        "A multivariate LSTM approach that attempts to generate not just point forecasts but also intervals. "
        "We fill the area between upper and lower bounds to illustrate forecast uncertainty."
    ),
    "lstm_multivariate_v2": (
        "An alternate multivariate LSTM model variant with a different architecture or training approach. "
        "It again leverages multiple input features, training on 80% of the data and forecasting the future. "
        "We compute MAPE to measure accuracy."
    ),
    "hybrid_prophet_xgboost": (
        "A hybrid approach combining Prophet (for seasonality and holiday effects) and XGBoost (for "
        "gradient boosting on exogenous features). The final forecast can merge or ensemble the two predictions. "
        "MAPE indicates how close the forecasts are to the actual data."
    ),
}
