# Forecast App

A Flask web application that lets you upload a CSV file with time series data and generate forecasts using different approaches. Currently, it implements a Prophet-based forecast, and additional methods can be added as separate modules.

## Project Structure

```
forecast_app/
├── app.py
├── config.py
├── views.py
├── forecast
│   ├── __init__.py
│   └── prophet_forecast.py
├── static
│   └── css
│       └── style.css
├── templates
│   ├── base.html
│   ├── index.html
│   └── result.html
├── requirements.txt
└── README.md
```

## Setup

1. **Clone or Download the Project**

   Clone this repository or create the folder structure as shown above.

2. **Navigate to the Project Directory**

   ```bash
   cd forecast_app
   ```

3. **(Optional) Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venvScriptsactivate
   ```

4. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Running the App

Start the application with:

```bash
docker-compose up --build
```

Then open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) to use the app.

## CSV File Format

The CSV file must include the following columns:
- **date**: Date of the record.
- **orders_count**: Count of orders.
- **is_holiday**: Boolean indicating if the date is a holiday.

### Example:

```csv
date,orders_count,is_holiday
2024-07-01,150,False
2024-07-02,200,True
...
```

## Adding New Forecast Approaches

1. Create a new module in the `forecast` folder (e.g., `new_forecast.py`) with the forecast functions.
2. Update `views.py` and the UI (templates) to include the new approach.
3. Follow the pattern used in the Prophet forecast module.

## License

This project is licensed under the MIT License.
