from flask import Blueprint, render_template, request, redirect, flash, url_for
import pandas as pd

from forecast import (
    prophet_forecast, lstm_forecast, lstm_multiservice_forecast, 
    lstm_multiservice_aggregate_forecast, holtwinters_forecast,
    lstm_univariate_forecast, lstm_multivariate_forecast,
    lstm_multivariate_interval_forecast, lstm_multivariate_forecast_v2,
    hybrid_prophet_xgboost_forecast
)
# Import the dictionary of descriptions
from descriptions import FORECAST_DESCRIPTIONS

main_blueprint = Blueprint('main', __name__)

@main_blueprint.route('/', methods=['GET'])
def index():
    forecast_options = [
        {"name": "Prophet Forecast", "value": "prophet"},
        {"name": "Base LSTM Forecast", "value": "lstm"},
        {"name": "LSTM Multiservice Forecast", "value": "lstm_multiservice"},
        {"name": "LSTM Multiservice Aggregate Forecast", "value": "lstm_multiservice_aggregate"},
        {"name": "Holt-Winters Forecast", "value": "holtwinters"},
        {"name": "LSTM Univariate Forecast", "value": "lstm_univariate"},
        {"name": "LSTM Multivariate Forecast", "value": "lstm_multivariate"},
        {"name": "LSTM Multivariate Interval Forecast", "value": "lstm_multivariate_interval"},
        {"name": "LSTM Multivariate Forecast v2", "value": "lstm_multivariate_v2"},
        {"name": "Hybrid Prophet-XGBoost Forecast", "value": "hybrid_prophet_xgboost"}
    ]
    return render_template('index.html', forecast_options=forecast_options)

@main_blueprint.route('/forecast', methods=['POST'])
def forecast_route():
    forecast_type = request.form.get("forecast_type")
    if 'datafile' not in request.files:
        flash("No file part")
        return redirect(url_for('main.index'))
    file = request.files['datafile']
    if file.filename == "":
        flash("No file selected")
        return redirect(url_for('main.index'))

    try:
        if forecast_type == "prophet":
            data = prophet_forecast.read_csv(file)
            # example if you call both train and run
            future_holidays = [
                pd.Timestamp('2025-01-20'),
                pd.Timestamp('2025-02-14'),
                pd.Timestamp('2025-03-17')
            ]
            model, forecast = prophet_forecast.train_forecast_model(data, future_holidays)
            img_base64, mape = prophet_forecast.run_forecast(data, forecast)
            label = "Prophet Forecast"

        elif forecast_type == "lstm":
            data = lstm_forecast.read_csv(file)
            img_base64, mape = lstm_forecast.run_forecast(data)
            label = "Base LSTM Forecast"

        elif forecast_type == "lstm_multiservice":
            data = lstm_multiservice_forecast.read_csv(file)
            img_base64, mape = lstm_multiservice_forecast.run_forecast(data)
            label = "LSTM Multiservice Forecast"

        elif forecast_type == "lstm_multiservice_aggregate":
            data = lstm_multiservice_aggregate_forecast.read_csv(file)
            img_base64, mape = lstm_multiservice_aggregate_forecast.run_forecast(data)
            label = "LSTM Multiservice Aggregate Forecast"

        elif forecast_type == "holtwinters":
            data = holtwinters_forecast.read_csv(file)
            img_base64, mape = holtwinters_forecast.run_forecast(data)
            label = "Holt-Winters Forecast"

        elif forecast_type == "lstm_univariate":
            data = lstm_univariate_forecast.read_csv(file)
            img_base64, mape = lstm_univariate_forecast.run_forecast(data)
            label = "LSTM Univariate Forecast"

        elif forecast_type == "lstm_multivariate":
            data = lstm_multivariate_forecast.read_csv(file)
            img_base64, mape = lstm_multivariate_forecast.run_forecast(data)
            label = "LSTM Multivariate Forecast"

        elif forecast_type == "lstm_multivariate_interval":
            data = lstm_multivariate_interval_forecast.read_csv(file)
            img_base64, mape = lstm_multivariate_interval_forecast.run_forecast(data)
            label = "LSTM Multivariate Interval Forecast"

        elif forecast_type == "lstm_multivariate_v2":
            data = lstm_multivariate_forecast_v2.read_csv(file)
            img_base64, mape = lstm_multivariate_forecast_v2.run_forecast(data)
            label = "LSTM Multivariate Forecast v2"

        elif forecast_type == "hybrid_prophet_xgboost":
            data = hybrid_prophet_xgboost_forecast.read_csv(file)
            img_base64, mape = hybrid_prophet_xgboost_forecast.run_forecast(data)
            label = "Hybrid Prophet-XGBoost Forecast"

        else:
            flash("Invalid forecast approach selected")
            return redirect(url_for('main.index'))
        
        # Retrieve the description text from the dictionary (default to an empty string if not found)
        module_description = FORECAST_DESCRIPTIONS.get(forecast_type, "")

        return render_template(
            'result.html',
            img_data=img_base64,
            mape=mape,
            forecast_type=label,
            module_description=module_description
        )
    except Exception as e:
        flash(f"An error occurred: {e}")
        return redirect(url_for('main.index'))
