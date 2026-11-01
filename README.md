# Ahmedabad Weather Analysis and Climate Trend Visualization

A college data science project that analyzes daily Ahmedabad weather data,
estimates long-term temperature trends, trains daily temperature prediction
models, and presents the results in a Streamlit dashboard.

## Data source

The daily weather observations were collected from the
[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
using ERA5 reanalysis data. Weather data is © Open-Meteo.com; ERA5 is provided
by the Copernicus Climate Change Service (C3S).

## Project files

- `LoadData.py` downloads historical daily weather data.
- `train_model.py` cleans the data, creates lag features, trains the trend and
  daily prediction models, and saves evaluation outputs.
- `app.py` provides the interactive Streamlit dashboard.
- `weather_ahmedabad.csv` is the collected source dataset.
- `outputs/` contains the cleaned dataset, metrics, and plots.
- `models/` contains the trained model and its ordered feature list.

## Setup

Use Python 3.10 or newer. From the project folder, install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run the dashboard

The generated model and dashboard data are included. Start the dashboard with:

```powershell
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

Create an app from this repository's `main` branch with `app.py` as the entrypoint.
In Advanced settings, select Python 3.11, then set the app subdomain to
`weatheranalysis29`. If the app already exists, reboot it and clear its cache so
Community Cloud reinstalls the dependencies from the root `requirements.txt`.

## Retrain the models

To regenerate the model and analysis outputs from the source CSV:

```powershell
python train_model.py
```

The daily models train on data before 2024 and evaluate on dates from 2024
onward. `models/best_model.joblib` is saved with scikit-learn 1.5.1; use the
version pinned in `requirements.txt` when loading it.
