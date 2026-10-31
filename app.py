"""Streamlit dashboard for Ahmedabad weather analysis and prediction."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "outputs" / "clean_data.csv"
METRICS_FILE = PROJECT_DIR / "outputs" / "metrics.csv"
MODEL_FILE = PROJECT_DIR / "models" / "best_model.joblib"
FEATURES_FILE = PROJECT_DIR / "models" / "features.json"

DATE_COLUMN = "time"
TARGET = "temperature_2m_mean"
RAINFALL = "precipitation_sum"
HUMIDITY = "relative_humidity_2m_mean"
WIND_SPEED = "wind_speed_10m_max"
TEST_START = pd.Timestamp("2024-01-01")
MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

st.set_page_config(
    page_title="Ahmedabad Weather Dashboard",
    page_icon="🌤️",
    layout="wide",
)


@st.cache_data
def load_dashboard_data(data_path: str, metrics_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned observations and the saved model comparison metrics."""
    data = pd.read_csv(data_path, parse_dates=[DATE_COLUMN])
    metrics = pd.read_csv(metrics_path)

    required_data_columns = {
        DATE_COLUMN,
        TARGET,
        RAINFALL,
        HUMIDITY,
        WIND_SPEED,
        "month",
        "day_of_year",
        "temperature_lag_1",
        "temperature_rolling_mean_7",
    }
    missing_columns = required_data_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "clean_data.csv is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
    required_metric_columns = {"Model", "MAE", "RMSE", "R2"}
    missing_metrics = required_metric_columns.difference(metrics.columns)
    if missing_metrics:
        raise ValueError(
            "metrics.csv is missing required columns: "
            + ", ".join(sorted(missing_metrics))
        )

    data = data.sort_values(DATE_COLUMN).reset_index(drop=True)
    return data, metrics


@st.cache_data
def load_feature_names(features_path: str) -> list[str]:
    """Load the exact ordered feature names used to train the saved model."""
    with open(features_path, encoding="utf-8") as feature_file:
        features = json.load(feature_file)
    if not isinstance(features, list) or not features or not all(
        isinstance(feature, str) for feature in features
    ):
        raise ValueError("features.json must contain a non-empty list of feature names.")
    return features


@st.cache_resource
def load_model(model_path: str):
    """Load and cache the trained prediction model."""
    return joblib.load(model_path)


def show_sidebar(data: pd.DataFrame) -> pd.DataFrame:
    """Show project context and return observations within the chosen dates."""
    st.sidebar.title("About this project")
    st.sidebar.write(
        "Explore Ahmedabad's historical weather, climate patterns, and daily "
        "temperature predictions."
    )
    st.sidebar.markdown(
        "Daily data source: [Open-Meteo Historical Weather API]"
        "(https://open-meteo.com/en/docs/historical-weather-api), "
        "using ERA5 reanalysis."
    )
    st.sidebar.caption(
        "Weather data © Open-Meteo.com. ERA5 data is provided by the "
        "Copernicus Climate Change Service (C3S)."
    )
    st.sidebar.subheader("Date range")
    first_date = data[DATE_COLUMN].min().date()
    last_date = data[DATE_COLUMN].max().date()
    selected_dates = st.sidebar.date_input(
        "Filter historical charts",
        value=(first_date, last_date),
        min_value=first_date,
        max_value=last_date,
    )

    if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
        st.sidebar.info("Choose both a start and end date.")
        return data.iloc[0:0]

    start_date, end_date = (pd.Timestamp(day) for day in selected_dates)
    return data[data[DATE_COLUMN].between(start_date, end_date)].copy()


def show_overview(data: pd.DataFrame) -> None:
    """Display headline weather summaries for the selected date range."""
    st.subheader("Ahmedabad at a glance")
    if data.empty:
        st.info("No observations are available for the selected date range.")
        return

    average_temperature = data[TARGET].mean()
    hottest = data.loc[data[TARGET].idxmax()]
    coldest = data.loc[data[TARGET].idxmin()]
    annual_rainfall = data.groupby(data[DATE_COLUMN].dt.year)[RAINFALL].sum()
    average_annual_rainfall = annual_rainfall.mean()

    cards = st.columns(4)
    cards[0].metric("Average temperature", f"{average_temperature:.1f} °C")
    cards[1].metric(
        "Hottest day",
        f"{hottest[TARGET]:.1f} °C",
        delta=hottest[DATE_COLUMN].strftime("%d %b %Y"),
        delta_color="off",
    )
    cards[2].metric(
        "Coldest day",
        f"{coldest[TARGET]:.1f} °C",
        delta=coldest[DATE_COLUMN].strftime("%d %b %Y"),
        delta_color="off",
    )
    cards[3].metric("Average rainfall per year", f"{average_annual_rainfall:.0f} mm")
    st.caption(
        f"Showing {data[DATE_COLUMN].min():%d %b %Y} to "
        f"{data[DATE_COLUMN].max():%d %b %Y}. Rainfall is averaged across "
        "calendar years represented in the selected range; edge years may be partial."
    )

    temperature_chart = px.line(
        data,
        x=DATE_COLUMN,
        y=TARGET,
        title="Daily Mean Temperature",
        labels={DATE_COLUMN: "Date", TARGET: "Temperature (°C)"},
    )
    st.plotly_chart(temperature_chart, use_container_width=True)


def show_climate_trend(data: pd.DataFrame) -> None:
    """Plot annual mean temperatures and their linear trend."""
    st.subheader("Yearly Mean Temperature Trend")
    yearly = (
        data.dropna(subset=[TARGET])
        .assign(year=lambda frame: frame[DATE_COLUMN].dt.year)
        .groupby("year", as_index=False)[TARGET]
        .mean()
    )
    if len(yearly) < 2:
        st.info("Select dates containing at least two years to calculate a trend.")
        return

    slope, intercept = np.polyfit(yearly["year"], yearly[TARGET], 1)
    slope_per_decade = slope * 10
    yearly["trend"] = intercept + slope * yearly["year"]

    chart = go.Figure()
    chart.add_trace(
        go.Scatter(
            x=yearly["year"],
            y=yearly[TARGET],
            mode="lines+markers",
            name="Yearly mean",
        )
    )
    chart.add_trace(
        go.Scatter(
            x=yearly["year"],
            y=yearly["trend"],
            mode="lines",
            name="Linear trend",
            line={"dash": "dash"},
        )
    )
    chart.update_layout(
        xaxis_title="Year",
        yaxis_title="Mean temperature (°C)",
        legend_title="",
    )
    st.metric("Linear trend", f"{slope_per_decade:+.3f} °C per decade")
    st.caption(
        "Trend is fitted to the yearly means in the selected date range. "
        "Years at the edges may include only part of that year."
    )
    st.plotly_chart(chart, use_container_width=True)


def show_seasonality(data: pd.DataFrame) -> None:
    """Display a year-by-month temperature heatmap and rainfall seasonality."""
    st.subheader("Seasonality")
    if data.empty:
        st.info("No observations are available for the selected date range.")
        return

    monthly = (
        data.assign(
            year=data[DATE_COLUMN].dt.year,
            month=data[DATE_COLUMN].dt.month,
        )
        .groupby(["year", "month"], as_index=False)
        .agg(
            mean_temperature=(TARGET, "mean"),
            mean_rainfall=(RAINFALL, "mean"),
        )
    )
    heatmap_values = monthly.pivot(
        index="year",
        columns="month",
        values="mean_temperature",
    ).reindex(columns=range(1, 13))
    heatmap = go.Figure(
        data=go.Heatmap(
            z=heatmap_values.to_numpy(),
            x=MONTH_NAMES,
            y=heatmap_values.index,
            colorscale="RdYlBu_r",
            colorbar={"title": "°C"},
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Mean: %{z:.1f} °C<extra></extra>",
        )
    )
    heatmap.update_layout(
        title="Monthly Mean Temperature (Year × Month)",
        xaxis_title="Month",
        yaxis_title="Year",
    )
    st.plotly_chart(heatmap, use_container_width=True)

    rainfall_by_month = (
        data.assign(month=data[DATE_COLUMN].dt.month)
        .groupby("month", as_index=False)[RAINFALL]
        .mean()
        .set_index("month")
        .reindex(range(1, 13))
        .rename_axis("month")
        .reset_index()
    )
    rainfall_chart = px.bar(
        rainfall_by_month,
        x="month",
        y=RAINFALL,
        labels={"month": "Month", RAINFALL: "Average daily rainfall (mm)"},
        title="Average Daily Rainfall by Month",
    )
    rainfall_chart.update_xaxes(
        tickmode="array",
        tickvals=list(range(1, 13)),
        ticktext=MONTH_NAMES,
    )
    st.plotly_chart(rainfall_chart, use_container_width=True)
    st.caption("Monthly rainfall bars show average daily rainfall within each month.")


def show_model_performance(
    data: pd.DataFrame,
    metrics: pd.DataFrame,
    model,
    features: list[str],
) -> None:
    """Show saved test metrics and predictions for dates from 2024 onward."""
    st.subheader("Model comparison")
    st.dataframe(
        metrics.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Random Forest: actual vs. predicted")
    # The caller passes date-filtered data, so this retains only selected test dates.
    test_data = data[data[DATE_COLUMN] >= TEST_START].copy()
    test_data = test_data.dropna(subset=[*features, TARGET])
    if test_data.empty:
        st.info(
            "No usable test-period observations (2024 onward) fall within "
            "the selected date range."
        )
        return

    actual_vs_predicted = test_data[[DATE_COLUMN, TARGET]].copy()
    actual_vs_predicted["Predicted temperature"] = model.predict(test_data[features])
    chart = go.Figure()
    chart.add_trace(
        go.Scatter(
            x=actual_vs_predicted[DATE_COLUMN],
            y=actual_vs_predicted[TARGET],
            mode="lines",
            name="Actual",
        )
    )
    chart.add_trace(
        go.Scatter(
            x=actual_vs_predicted[DATE_COLUMN],
            y=actual_vs_predicted["Predicted temperature"],
            mode="lines",
            name="Random Forest prediction",
        )
    )
    chart.update_layout(
        xaxis_title="Date",
        yaxis_title="Mean temperature (°C)",
        legend_title="",
    )
    st.plotly_chart(chart, use_container_width=True)


def show_prediction(model, features: list[str]) -> None:
    """Collect model inputs and predict a daily mean temperature."""
    st.subheader("Predict daily mean temperature")
    st.write("Enter the conditions and recent temperatures for the day to predict.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            month = st.selectbox(
                "Month",
                options=range(1, 13),
                format_func=lambda value: MONTH_NAMES[value - 1],
            )
            day_of_year = st.number_input(
                "Day of year",
                min_value=1,
                max_value=366,
                value=180,
                help="Use 1–365, or 366 in a leap year.",
            )
            humidity = st.number_input(
                "Relative humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=60.0,
                step=1.0,
            )
            rainfall = st.number_input(
                "Rainfall (mm)",
                min_value=0.0,
                value=0.0,
                step=1.0,
            )
        with col2:
            wind_speed = st.number_input(
                "Maximum wind speed (km/h)",
                min_value=0.0,
                value=10.0,
                step=1.0,
            )
            yesterday_temperature = st.number_input(
                "Yesterday's mean temperature (°C)",
                value=30.0,
                step=0.5,
            )
            rolling_temperature = st.number_input(
                "Previous 7-day average temperature (°C)",
                value=30.0,
                step=0.5,
                help="Average of the seven days before the day being predicted.",
            )
        submitted = st.form_submit_button("Predict temperature")

    if submitted:
        input_values = {
            "month": month,
            "day_of_year": day_of_year,
            HUMIDITY: humidity,
            RAINFALL: rainfall,
            WIND_SPEED: wind_speed,
            "temperature_lag_1": yesterday_temperature,
            "temperature_rolling_mean_7": rolling_temperature,
        }
        missing_features = [feature for feature in features if feature not in input_values]
        if missing_features:
            st.error(
                "The saved feature list contains unsupported inputs: "
                + ", ".join(missing_features)
            )
            return

        feature_row = pd.DataFrame(
            [[input_values[feature] for feature in features]],
            columns=features,
        )
        prediction = float(model.predict(feature_row)[0])
        st.success(f"Predicted daily mean temperature: {prediction:.1f} °C")


def main() -> None:
    """Load artifacts and render the dashboard tabs."""
    required_files = [DATA_FILE, METRICS_FILE, MODEL_FILE, FEATURES_FILE]
    missing_files = [
        str(path.relative_to(PROJECT_DIR))
        for path in required_files
        if not path.is_file()
    ]
    if missing_files:
        st.error(
            "Dashboard files are missing. Run the training script first or "
            "restore these project files: " + ", ".join(missing_files)
        )
        st.stop()

    try:
        all_data, metrics = load_dashboard_data(str(DATA_FILE), str(METRICS_FILE))
        features = load_feature_names(str(FEATURES_FILE))
        model = load_model(str(MODEL_FILE))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        st.error(f"Could not load dashboard data or configuration: {error}")
        st.stop()
    except (OSError, ValueError, EOFError, ImportError) as error:
        st.error(f"Could not load the saved model: {error}")
        st.stop()

    if all_data.empty:
        st.error("clean_data.csv contains no weather observations.")
        st.stop()

    st.title("Ahmedabad Weather Data Analysis & Climate Trends")
    st.caption("Historical climate patterns and daily mean temperature predictions")
    selected_data = show_sidebar(all_data)

    overview_tab, trend_tab, seasonality_tab, performance_tab, predict_tab = st.tabs(
        [
            "Overview",
            "Climate Trend",
            "Seasonality",
            "Model Performance",
            "Predict",
        ]
    )
    with overview_tab:
        show_overview(selected_data)
    with trend_tab:
        show_climate_trend(selected_data)
    with seasonality_tab:
        show_seasonality(selected_data)
    with performance_tab:
        show_model_performance(selected_data, metrics, model, features)
    with predict_tab:
        show_prediction(model, features)


if __name__ == "__main__":
    main()
