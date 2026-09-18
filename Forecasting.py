import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb

raw_data = pd.read_csv('PJM_load_hourly.csv', parse_dates=['Datetime'])
raw_data = raw_data.set_index('Datetime').sort_index()

hourly_data = raw_data[~raw_data.index.duplicated(keep='first')].asfreq('h')
hourly_data['consumption'] = hourly_data['PJM_Load_MW'].interpolate()
clean_series = hourly_data['consumption']

train_series = clean_series[:'2001-06-01']
test_series = clean_series['2001-06-02':'2001-06-30']
forecast_horizon = len(test_series)

daily_model = ARIMA(train_series, order=(3, 1, 3)).fit()
daily_forecast = daily_model.forecast(steps=forecast_horizon)

daily_residual = train_series - daily_model.fittedvalues
weekly_residual = daily_residual.resample('D').mean()
weekly_model = ARIMA(weekly_residual.dropna(), order=(2, 0, 2)).fit()

projected_dates = pd.date_range(test_series.index[0], periods=forecast_horizon, freq='h')
weekly_forecast = weekly_model.predict(
    start=0, end=forecast_horizon // 24 + 10
).reindex(projected_dates, method='ffill')[:forecast_horizon]

try:
    monthly_residual = (daily_residual - weekly_model.fittedvalues.reindex(daily_residual.index).ffill()).resample('ME').mean()
except Exception:
    monthly_residual = (daily_residual - weekly_model.fittedvalues.reindex(daily_residual.index).ffill()).resample('M').mean()

yearly_model = ARIMA(monthly_residual.dropna(), order=(1, 0, 1)).fit()
yearly_forecast = yearly_model.predict(
    start=0, end=forecast_horizon // 720 + 5
).reindex(projected_dates, method='ffill')[:forecast_horizon]

har_forecast = daily_forecast + weekly_forecast.values + yearly_forecast.values

recent_window = train_series[-672:]
weekly_profile = recent_window[-168:].values
repeat_count = -(-forecast_horizon // 168)
repeated_profile = np.tile(weekly_profile, repeat_count)[:forecast_horizon]

trend_sample = train_series[-720:]
trend_slope = np.polyfit(range(len(trend_sample)), trend_sample, 1)[0]
linear_trend = trend_slope * np.arange(forecast_horizon)

baseline_values = repeated_profile + linear_trend + train_series.mean() - np.mean(weekly_profile)
sarima_forecast = pd.Series(baseline_values, index=test_series.index)

def generate_time_features(series_data):
    frame = pd.DataFrame({'consumption': series_data})
    frame['hour'] = frame.index.hour
    frame['day_of_week'] = frame.index.dayofweek
    frame['lag_24h'] = frame['consumption'].shift(24)
    frame['lag_168h'] = frame['consumption'].shift(168)
    frame['rolling_mean_24h'] = frame['consumption'].rolling(24).mean()
    return frame.dropna()

train_features = generate_time_features(train_series)
X_train = train_features.drop('consumption', axis=1)
y_train = train_features['consumption']

regressor = xgb.XGBRegressor(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)
regressor.fit(X_train, y_train)

full_features = generate_time_features(clean_series)
X_test = full_features.loc[test_series.index].drop('consumption', axis=1)
xgb_forecast = regressor.predict(X_test)

metrics_summary = pd.DataFrame({
    'Model': ['HAR-ARIMA', 'XGBoost', 'SARIMA'],
    'RMSE': [
        np.sqrt(mean_squared_error(test_series, har_forecast)),
        np.sqrt(mean_squared_error(test_series, xgb_forecast)),
        np.sqrt(mean_squared_error(test_series, sarima_forecast))
    ],
    'MAE': [
        mean_absolute_error(test_series, har_forecast),
        mean_absolute_error(test_series, xgb_forecast),
        mean_absolute_error(test_series, sarima_forecast)
    ],
    'MAPE (%)': [
        100 * mean_absolute_percentage_error(test_series, har_forecast),
        100 * mean_absolute_percentage_error(test_series, xgb_forecast),
        100 * mean_absolute_percentage_error(test_series, sarima_forecast)
    ]
}).round(3)

print(metrics_summary.to_string(index=False))

plt.figure(figsize=(16, 7))
plt.plot(test_series.index, test_series, label='Actual Demand', linewidth=2.5, color='black')
plt.plot(test_series.index, har_forecast, label='HAR-ARIMA', linewidth=2.2, color='#1b9e77')
plt.plot(test_series.index, xgb_forecast, label='XGBoost', linewidth=1.8, color='#e6ab02', linestyle='--')
plt.plot(test_series.index, sarima_forecast, label='SARIMA Baseline', linewidth=1.5, color='#e41a1c', alpha=0.7)
plt.title('PJM Hourly Electricity Load Forecasting Evaluation', fontsize=15, pad=12)
plt.xlabel('Timestamp', fontsize=12)
plt.ylabel('Electricity Consumption (MW)', fontsize=12)
plt.legend(loc='upper right', frameon=True)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('actual_vs_forecast_lineplot.png', dpi=150)
plt.show()

figure, axes = plt.subplots(1, 3, figsize=(18, 5))
figure.suptitle('Forecasting Model Accuracy Comparison', fontsize=16)

chart_colors = ['#1b9e77', '#e6ab02', '#e41a1c']
metric_names = ['RMSE', 'MAE', 'MAPE (%)']

for ax, metric in zip(axes, metric_names):
    bars = ax.bar(metrics_summary['Model'], metrics_summary[metric], color=chart_colors)
    ax.set_title(metric, fontsize=14, pad=8)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar in bars:
        bar_height = bar.get_height()
        formatted_value = f'{bar_height:.2f}' if metric == 'MAPE (%)' else f'{bar_height:.1f}'
        ax.text(bar.get_x() + bar.get_width() / 2, bar_height, formatted_value,
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('model_comparison_barchart.png', dpi=150)
plt.show()