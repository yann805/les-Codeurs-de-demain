import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib

print("Loading dataset...")

# ── Load dataset ─────────────────────────────────────────────
df = pd.read_excel("data/Dataset_complet_Meteo.xlsx")
print("Dataset loaded:", df.shape)

# ── Convert numeric columns ──────────────────────────────────
num_cols = [
    'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
    'apparent_temperature_mean', 'precipitation_sum', 'rain_sum',
    'wind_speed_10m_max', 'wind_gusts_10m_max',
    'shortwave_radiation_sum', 'et0_fao_evapotranspiration',
    'sunshine_duration', 'latitude', 'longitude'
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ── Time features ────────────────────────────────────────────
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values(['city', 'time'])  # VERY IMPORTANT

df['month'] = df['time'].dt.month
df['day_of_year'] = df['time'].dt.dayofyear

# Cyclical encoding
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# ── Feature Engineering ──────────────────────────────────────
df['temp_amplitude'] = df['temperature_2m_max'] - df['temperature_2m_min']

df['sunshine_ratio'] = df['sunshine_duration'] / (24 * 3600)

df['is_no_wind'] = (df['wind_speed_10m_max'] < 1).astype(int)
df['is_no_rain'] = (df['precipitation_sum'] == 0).astype(int)

# Dry season (Cameroon approximation)
df['is_dry_season'] = df['month'].isin([12, 1, 2]).astype(int)

# ── Lag Features (CRITICAL) ──────────────────────────────────
df['temp_lag1'] = df.groupby('city')['temperature_2m_mean'].shift(1)
df['temp_lag7'] = df.groupby('city')['temperature_2m_mean'].shift(7)
df['wind_lag1'] = df.groupby('city')['wind_speed_10m_max'].shift(1)

df['temp_roll7'] = (
    df.groupby('city')['temperature_2m_mean']
    .rolling(7).mean()
    .reset_index(0, drop=True)
)

# ── Encode location ──────────────────────────────────────────
df['city_enc'] = df['city'].astype('category').cat.codes
df['region_enc'] = df['region'].astype('category').cat.codes

# ── PM2.5 Proxy (FULL VERSION) ───────────────────────────────
df['pm25_proxy'] = (
    0.35 * df['temperature_2m_mean'].fillna(df['temperature_2m_mean'].mean())
    + 0.25 * df['shortwave_radiation_sum'].fillna(0)
    + 0.20 * df['et0_fao_evapotranspiration'].fillna(0)
    + 8.0  * df['is_no_wind']
    + 5.0  * df['is_no_rain']
    + 4.0  * df['is_dry_season']
).clip(lower=0)

print("Feature engineering complete")

# ── Feature selection ────────────────────────────────────────
FEATURES = [
    'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min',
    'precipitation_sum', 'wind_speed_10m_max', 'wind_gusts_10m_max',
    'shortwave_radiation_sum', 'et0_fao_evapotranspiration',
    'sunshine_ratio', 'temp_amplitude',
    'is_no_wind', 'is_no_rain', 'is_dry_season',
    'month_sin', 'month_cos', 'day_of_year',
    'temp_lag1', 'temp_lag7', 'wind_lag1', 'temp_roll7',
    'latitude', 'longitude',
    'region_enc', 'city_enc'
]

TARGET = 'pm25_proxy'

df_model = df[FEATURES + [TARGET]].copy()

# Fill missing values
for col in FEATURES:
    df_model[col] = df_model[col].fillna(df_model[col].median())

df_model = df_model.dropna(subset=[TARGET])

X = df_model[FEATURES]
y = df_model[TARGET]

# ── Time-based split ─────────────────────────────────────────
split = int(len(df_model) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print("Train size:", len(X_train))
print("Test size:", len(X_test))

# ── Model ───────────────────────────────────────────────────
model = XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

print("Training model...")
model.fit(X_train, y_train)

# ── Prediction ──────────────────────────────────────────────
y_pred = model.predict(X_test)

# ── Evaluation ──────────────────────────────────────────────
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nRESULTS:")
print("R²  :", round(r2, 4))
print("MAE :", round(mae, 4))

# ── Save model ──────────────────────────────────────────────
joblib.dump(model, "models/xgb_model.pkl")

print("Model saved successfully!")