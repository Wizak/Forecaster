from flask import Blueprint, render_template, request, redirect, flash, url_for
import pandas as pd
from forecast import (
    prophet_forecast,
    lstm_forecast,
    lstm_multiservice_forecast,
    lstm_multiservice_aggregate_forecast,
    holtwinters_forecast,
    lstm_univariate_forecast,
    lstm_multivariate_forecast,
    lstm_multivariate_interval_forecast,
    lstm_multivariate_forecast_v2,
    hybrid_prophet_xgboost_forecast
)
from descriptions import FORECAST_DESCRIPTIONS

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

main_blueprint = Blueprint('main', __name__)

@main_blueprint.route('/', methods=['GET'])
def index():
    return render_template('index.html', forecast_options=forecast_options)

@main_blueprint.route('/forecast', methods=['POST'])
def forecast_route():
    forecast_type = request.form.get("forecast_type")
    file = request.files.get("datafile")
    if not file or file.filename == "":
        flash("No file selected")
        return redirect(url_for('main.index'))
    mape_val = None
    xaxis = { "title": "Date", "type": "date" }
    try:
        if forecast_type == "prophet":
            training_cutoff_date = request.form.get("training_cutoff_date")
            forecast_days_ahead = request.form.get("forecast_days_ahead")
            data, mape_val = prophet_forecast.run_forecast_table(
                file,
                training_cutoff_date=training_cutoff_date if training_cutoff_date else None,
                forecast_days_ahead=int(forecast_days_ahead) if forecast_days_ahead else 30,
                future_holidays=[pd.Timestamp('2025-01-20'),
                                 pd.Timestamp('2025-02-14'),
                                 pd.Timestamp('2025-03-17')]
            )
        elif forecast_type == "lstm":
            train_start = request.form.get("train_start_date")
            train_end = request.form.get("train_end_date")
            test_start = request.form.get("test_start_date")
            test_end = request.form.get("test_end_date")
            forecast_start = request.form.get("forecast_start_date")
            forecast_end = request.form.get("forecast_end_date")
            data, mape_val = lstm_forecast.run_forecast_table(
                file,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                forecast_start=forecast_start,
                forecast_end=forecast_end,
            )
        elif forecast_type == "lstm_multiservice":
            train_start = request.form.get("train_start_date")
            train_end = request.form.get("train_end_date")
            test_start = request.form.get("test_start_date")
            test_end = request.form.get("test_end_date")
            forecast_start = request.form.get("forecast_start_date")
            forecast_end = request.form.get("forecast_end_date")
            data, mape_val = lstm_multiservice_forecast.run_forecast_table(
                file,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                forecast_start=forecast_start,
                forecast_end=forecast_end,
            )
        elif forecast_type == "lstm_multiservice_aggregate":
            forecast_start = request.form.get("forecast_start_date")
            forecast_end = request.form.get("forecast_end_date")
            data = lstm_multiservice_aggregate_forecast.run_forecast_table(
                file,
                forecast_start=forecast_start,
                forecast_end=forecast_end,
            )
        elif forecast_type == "holtwinters":
            forecast_start = request.form.get("forecast_start_date")
            forecast_end = request.form.get("forecast_end_date")
            data, mape_val = holtwinters_forecast.run_forecast_table(
                file,
                forecast_start=forecast_start,
                forecast_end=forecast_end,
            )
        elif forecast_type == "lstm_univariate":
            forecast_start = request.form.get("forecast_start_date")
            forecast_end = request.form.get("forecast_end_date")
            data = lstm_univariate_forecast.run_forecast_table(
                file,
                forecast_start=forecast_start,
                forecast_end=forecast_end,
            )
        elif forecast_type == "lstm_multivariate":
            forecast_start = request.form.get("forecast_start_date")
            forecast_end = request.form.get("forecast_end_date")
            xaxis = { "title": "Steps", "type": "linear" }
            data = lstm_multivariate_forecast.run_forecast_table(
                file,
                forecast_start=forecast_start,
                forecast_end=forecast_end,
            )
        elif forecast_type == "lstm_multivariate_interval":
            forecast_start = request.form.get("forecast_start_date")
            forecast_end = request.form.get("forecast_end_date")
            xaxis = { "title": "Steps", "type": "linear" }
            data = lstm_multivariate_interval_forecast.run_forecast_table(
                file,
                forecast_start=forecast_start,
                forecast_end=forecast_end,
            )
        elif forecast_type == "lstm_multivariate_v2":
            xaxis = { "title": "Steps", "type": "linear" }
            data = lstm_multivariate_forecast_v2.run_forecast_table(
                file,
            )
        elif forecast_type == "hybrid_prophet_xgboost":
            forecast_days_ahead = request.form.get("forecast_days_ahead")
            data = hybrid_prophet_xgboost_forecast.run_forecast_table(
                file,
                forecast_period=int(forecast_days_ahead) if forecast_days_ahead else 30,
            )
        else:
            flash("Invalid forecast approach selected.")
            return redirect(url_for('main.index'))
        
        description = FORECAST_DESCRIPTIONS.get(forecast_type, "")
        return render_template(
            'result.html',
            forecast_type=list(filter(lambda x: x["value"] == forecast_type, forecast_options))[0]["name"],
            data=data,
            xaxis=xaxis,
            mape=mape_val,
            module_description=description
        )
    except Exception as e:
        flash(f"An error occurred: {e}")
        return redirect(url_for('main.index'))
