"""Train climate trend and daily temperature models for Ahmedabad."""

from pathlib import Path
import sys

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_DIR / "weather_ahmedabad.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"

DATE_COLUMN = "time"
TARGET = "temperature_2m_mean"
WEATHER_COLUMNS = [
    TARGET,
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "relative_humidity_2m_mean",
]
FEATURE_COLUMNS = [
    "month",
    "day_of_year",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "temperature_lag_1",
    "temperature_rolling_mean_7",
]
MAX_INTERPOLATED_GAP_DAYS = 3
TEST_START_DATE = pd.Timestamp("2024-01-01")


def load_and_clean_data(csv_path: Path) -> pd.DataFrame:
    """Load daily observations, remove duplicate dates, and fill short gaps."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find the input CSV: {csv_path}\n"
            "Place weather_ahmedabad.csv in the project folder."
        )

    data = pd.read_csv(csv_path)
    required_columns = [DATE_COLUMN, *WEATHER_COLUMNS]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "The CSV is missing required columns: " + ", ".join(missing_columns)
        )

    print("\n=== 1. Load and clean data ===")
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
    invalid_dates = int(data[DATE_COLUMN].isna().sum())
    if invalid_dates:
        print(f"Rows with invalid dates removed: {invalid_dates}")
        data = data.dropna(subset=[DATE_COLUMN])

    for column in WEATHER_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    duplicate_rows = int(data.duplicated().sum())
    duplicate_dates = int(data[DATE_COLUMN].duplicated().sum())
    print(f"Exact duplicate rows found: {duplicate_rows}")
    print(f"Duplicate dates found: {duplicate_dates}")
    data = data.drop_duplicates(subset=DATE_COLUMN, keep="first")
    data = data.sort_values(DATE_COLUMN).set_index(DATE_COLUMN)

    # Restore omitted calendar days so a one-row lag always means the prior day.
    daily_index = pd.date_range(data.index.min(), data.index.max(), freq="D", name=DATE_COLUMN)
    missing_days = len(daily_index) - len(data)
    data = data.reindex(daily_index)
    print(f"Missing calendar days added to the series: {missing_days}")

    print("Missing values before interpolation:")
    print(data[WEATHER_COLUMNS].isna().sum().to_string())

    # Interpolate only fully enclosed gaps of at most three consecutive days.
    for column in WEATHER_COLUMNS:
        values = data[column]
        missing = values.isna()
        group_ids = (~missing).cumsum()
        gap_lengths = missing.groupby(group_ids).transform("sum")
        small_gap = missing & (gap_lengths <= MAX_INTERPOLATED_GAP_DAYS)
        interpolated = values.interpolate(method="time", limit_area="inside")
        data.loc[small_gap, column] = interpolated.loc[small_gap]

    print("Missing values after interpolation:")
    print(data[WEATHER_COLUMNS].isna().sum().to_string())

    data = data.reset_index()
    return data


def create_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and shifted temperature-history features."""
    featured = data.copy()
    featured["month"] = featured[DATE_COLUMN].dt.month
    featured["day_of_year"] = featured[DATE_COLUMN].dt.day_of_year

    # Shift before rolling so the current day's temperature is never included.
    previous_day = featured[TARGET].shift(1)
    featured["temperature_lag_1"] = previous_day
    featured["temperature_rolling_mean_7"] = previous_day.rolling(
        window=7, min_periods=7
    ).mean()
    return featured


def fit_and_plot_yearly_trend(data: pd.DataFrame) -> None:
    """Fit a linear trend using only complete calendar years."""
    print("\n=== 2. Climate trend model ===")
    complete_years = []
    for year, yearly_data in data.groupby(data[DATE_COLUMN].dt.year):
        days_in_year = 366 if pd.Timestamp(year=year, month=12, day=31).day_of_year == 366 else 365
        is_complete = (
            len(yearly_data) == days_in_year
            and yearly_data[DATE_COLUMN].min() == pd.Timestamp(year, 1, 1)
            and yearly_data[DATE_COLUMN].max() == pd.Timestamp(year, 12, 31)
            and yearly_data[TARGET].notna().all()
        )
        if is_complete:
            complete_years.append(
                {
                    "year": year,
                    "mean_temperature": yearly_data[TARGET].mean(),
                }
            )
        else:
            print(f"Skipping incomplete year: {year}")

    yearly_means = pd.DataFrame(complete_years)
    if len(yearly_means) < 2:
        raise ValueError("At least two complete years are needed to fit the climate trend.")

    model = LinearRegression()
    model.fit(yearly_means[["year"]], yearly_means["mean_temperature"])
    predicted = model.predict(yearly_means[["year"]])
    r_squared = r2_score(yearly_means["mean_temperature"], predicted)
    slope_per_decade = float(model.coef_[0]) * 10
    print(f"Yearly mean temperature trend: {slope_per_decade:.4f} °C per decade")
    print(f"Trend model R²: {r_squared:.4f}")

    line_years = np.arange(yearly_means["year"].min(), yearly_means["year"].max() + 1)
    line_temperatures = model.predict(pd.DataFrame({"year": line_years}))
    plt.figure(figsize=(10, 6))
    plt.scatter(
        yearly_means["year"],
        yearly_means["mean_temperature"],
        color="tab:blue",
        label="Yearly mean temperature",
    )
    plt.plot(line_years, line_temperatures, color="tab:red", label="Linear trend")
    plt.xlabel("Year")
    plt.ylabel("Mean temperature (°C)")
    plt.title("Ahmedabad Yearly Mean Temperature Trend")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trend_yearly.png", dpi=150)
    plt.close()


def train_daily_models(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Train and evaluate baseline and random-forest daily models."""
    print("\n=== 3. Daily temperature prediction models ===")
    usable_data = data.dropna(subset=[*FEATURE_COLUMNS, TARGET]).copy()
    train_data = usable_data[usable_data[DATE_COLUMN] < TEST_START_DATE]
    test_data = usable_data[usable_data[DATE_COLUMN] >= TEST_START_DATE]
    if train_data.empty or len(test_data) < 2:
        raise ValueError(
            "The time split needs training rows before 2024 and at least two "
            "usable test rows from 2024 onward."
        )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
        ),
    }
    results = []
    predictions = {}
    for name, model in models.items():
        model.fit(train_data[FEATURE_COLUMNS], train_data[TARGET])
        predicted = model.predict(test_data[FEATURE_COLUMNS])
        predictions[name] = predicted
        results.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(test_data[TARGET], predicted),
                "RMSE": np.sqrt(mean_squared_error(test_data[TARGET], predicted)),
                "R2": r2_score(test_data[TARGET], predicted),
            }
        )

    metrics = pd.DataFrame(results)
    print(f"Training rows: {len(train_data)} (before {TEST_START_DATE.date()})")
    print(f"Test rows: {len(test_data)} (from {TEST_START_DATE.date()} onward)")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    plt.figure(figsize=(12, 6))
    plt.plot(test_data[DATE_COLUMN], test_data[TARGET], label="Actual", linewidth=1.5)
    plt.plot(
        test_data[DATE_COLUMN],
        predictions["RandomForestRegressor"],
        label="Random Forest prediction",
        linewidth=1.2,
    )
    plt.xlabel("Date")
    plt.ylabel("Mean temperature (°C)")
    plt.title("Actual vs. Predicted Daily Mean Temperature (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    forest = models["RandomForestRegressor"]
    importances = pd.Series(forest.feature_importances_, index=FEATURE_COLUMNS)
    importances = importances.sort_values()
    plt.figure(figsize=(9, 6))
    importances.plot(kind="barh", color="tab:green")
    plt.xlabel("Feature importance")
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close()

    return metrics, test_data, models


def save_artifacts(
    data: pd.DataFrame,
    metrics: pd.DataFrame,
    models: dict,
) -> None:
    """Save data, model, feature names, and evaluation results for the dashboard."""
    print("\n=== 4. Save artifacts ===")
    data.to_csv(OUTPUT_DIR / "clean_data.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    best_model_name = metrics.loc[metrics["RMSE"].idxmin(), "Model"]
    joblib.dump(models[best_model_name], MODEL_DIR / "best_model.joblib")
    pd.Series(FEATURE_COLUMNS).to_json(MODEL_DIR / "features.json", orient="values", indent=2)
    print(f"Best model by test RMSE: {best_model_name}")
    print(f"Saved cleaned data and plots to: {OUTPUT_DIR}")
    print(f"Saved model and feature list to: {MODEL_DIR}")


def main() -> None:
    """Run the full training workflow from the project root."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    cleaned_data = load_and_clean_data(INPUT_FILE)
    engineered_data = create_engineered_features(cleaned_data)
    fit_and_plot_yearly_trend(cleaned_data)
    metrics, _, models = train_daily_models(engineered_data)
    save_artifacts(engineered_data, metrics, models)


if __name__ == "__main__":
    main()
