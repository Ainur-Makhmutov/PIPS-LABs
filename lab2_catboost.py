import pandas as pd
import numpy as np
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool

# -------------------------------
# 1. ЗАГРУЗКА ДАННЫХ
# -------------------------------

FILE_PATH = "Data/Dataset-24.csv"

print("Загрузка CSV...")
df = pd.read_csv(FILE_PATH)
print(f"Загружено строк: {len(df)}")
print(f"Столбцы: {df.columns.tolist()}")

# -------------------------------
# 2. ПАРСИНГ JSON-ПОЛЕЙ
# -------------------------------

def parse_usage_json(val):
    if pd.isna(val):
        return 0.0, 0.0
    if isinstance(val, str):
        try:
            data = json.loads(val.replace("'", '"'))
            return data.get('cpus', 0.0), data.get('memory', 0.0)
        except:
            return 0.0, 0.0
    return 0.0, 0.0

print("\nПарсинг JSON полей...")

# Парсинг average_usage
df['avg_cpu'] = 0.0
df['avg_memory'] = 0.0
for idx, val in enumerate(df['average_usage']):
    cpu, mem = parse_usage_json(val)
    df.loc[idx, 'avg_cpu'] = cpu
    df.loc[idx, 'avg_memory'] = mem

# Парсинг maximum_usage
df['max_cpu'] = 0.0
df['max_memory'] = 0.0
for idx, val in enumerate(df['maximum_usage']):
    cpu, mem = parse_usage_json(val)
    df.loc[idx, 'max_cpu'] = cpu
    df.loc[idx, 'max_memory'] = mem

# -------------------------------
# 3. ПРЕОБРАЗОВАНИЕ ВРЕМЕНИ
# -------------------------------

print("Преобразование времени...")
df['start_datetime'] = pd.to_datetime(df['start_time'], unit='us')
df['end_datetime'] = pd.to_datetime(df['end_time'], unit='us')

df['hour'] = df['start_datetime'].dt.hour
df['day_of_week'] = df['start_datetime'].dt.dayofweek
df['minute'] = df['start_datetime'].dt.minute
df['interval_duration'] = (df['end_datetime'] - df['start_datetime']).dt.total_seconds()

# -------------------------------
# 4. ДОПОЛНИТЕЛЬНЫЕ ПРИЗНАКИ
# -------------------------------

df['cpu_peak_ratio'] = df['max_cpu'] / (df['avg_cpu'] + 1e-6)
df['memory_peak_ratio'] = df['max_memory'] / (df['avg_memory'] + 1e-6)
df['memory_utilization'] = df['avg_memory'] / (df['assigned_memory'] + 1e-6)

# -------------------------------
# 5. СОРТИРОВКА И ЛАГОВЫЕ ПРИЗНАКИ
# -------------------------------

print("Создание лаговых признаков...")
df = df.sort_values(['collection_id', 'instance_index', 'start_time'])

for lag in [1, 2, 6, 12]:
    df[f'avg_cpu_lag_{lag}'] = df.groupby(['collection_id', 'instance_index'])['avg_cpu'].shift(lag)

df['avg_cpu_ma_6'] = df.groupby(['collection_id', 'instance_index'])['avg_cpu'].transform(
    lambda x: x.rolling(window=6, min_periods=1).mean()
)
df['avg_cpu_ma_12'] = df.groupby(['collection_id', 'instance_index'])['avg_cpu'].transform(
    lambda x: x.rolling(window=12, min_periods=1).mean()
)

# -------------------------------
# 6. ПОДГОТОВКА ВЫБОРКИ
# -------------------------------

df_clean = df.dropna().reset_index(drop=True)
print(f"После удаления NaN: {len(df_clean)} строк")

feature_cols = [
    'hour', 'day_of_week', 'minute',
    'avg_cpu_lag_1', 'avg_cpu_lag_2', 'avg_cpu_lag_6', 'avg_cpu_lag_12',
    'avg_cpu_ma_6', 'avg_cpu_ma_12',
    'cpu_peak_ratio', 'memory_peak_ratio',
    'assigned_memory', 'memory_utilization',
    'collection_id', 'machine_id'
]

existing_features = [col for col in feature_cols if col in df_clean.columns]
print(f"\nИспользуемые признаки: {existing_features}")

X = df_clean[existing_features].copy()
y = df_clean['avg_cpu'].copy()

cat_features = ['collection_id', 'machine_id']
for col in cat_features:
    if col in X.columns:
        X[col] = X[col].astype(str)

print(f"\nРазмер выборки: {len(X)} записей")
print(f"Количество признаков: {len(existing_features)}")

# -------------------------------
# 7. РАЗДЕЛЕНИЕ НА ОБУЧЕНИЕ И ТЕСТ
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

print(f"\nОбучающая выборка: {len(X_train)} записей")
print(f"Тестовая выборка: {len(X_test)} записей")

memory_mb = X_train.memory_usage(deep=True).sum() / (1024 * 1024)
print(f"Объем памяти данных (train): {memory_mb:.2f} MB")

# -------------------------------
# 8. ОБУЧЕНИЕ МОДЕЛИ
# -------------------------------

print("\n=== Обучение CatBoost ===")

train_pool = Pool(
    data=X_train,
    label=y_train,
    cat_features=[c for c in cat_features if c in X_train.columns]
)
test_pool = Pool(
    data=X_test,
    label=y_test,
    cat_features=[c for c in cat_features if c in X_train.columns]
)

model = CatBoostRegressor(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    verbose=50,
    random_seed=42
)

start_train = time.time()
model.fit(
    train_pool,
    eval_set=test_pool,
    early_stopping_rounds=50,
    verbose=False
)
train_time = time.time() - start_train

start_pred = time.time()
y_pred = model.predict(X_test)
pred_time_ms = (time.time() - start_pred) / len(X_test) * 1000

# -------------------------------
# 9. ОЦЕНКА КАЧЕСТВА
# -------------------------------

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n=== Результаты ===")
print(f"Время обучения: {train_time:.2f} сек")
print(f"Время прогноза на 1 объект: {pred_time_ms:.4f} мс")

# -------------------------------
# 10. ВАЖНОСТЬ ПРИЗНАКОВ
# -------------------------------

print("\n=== Важность признаков ===")
feature_importance = model.get_feature_importance()
for name, imp in sorted(zip(existing_features, feature_importance), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.2f}")

# -------------------------------
# 11. ПРИМЕР ПРОГНОЗА
# -------------------------------

print("\n=== Пример прогноза ===")
sample_idx = 0
sample = X_test.iloc[[sample_idx]]
pred_cpu = model.predict(sample)[0]
actual_cpu = y_test.iloc[sample_idx]

print(f"Прогноз CPU usage: {pred_cpu:.6f}")
print(f"Фактический CPU usage: {actual_cpu:.6f}")
print(f"Абсолютная ошибка: {abs(pred_cpu - actual_cpu):.6f}")

# -------------------------------
# 12. ЛИНГВИСТИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ
# -------------------------------

def get_load_label(cpu_usage, max_cpu=1.0):
    norm = min(cpu_usage / max_cpu, 1.0)
    if norm <= 0.25:
        return "Низкая"
    elif norm <= 0.5:
        return "Умеренная"
    elif norm <= 0.75:
        return "Повышенная"
    else:
        return "Высокая (близкая к пиковой)"

print(f"\n=== Лингвистическая интерпретация ===")
print(f"Прогноз: {pred_cpu:.4f} ядер → {get_load_label(pred_cpu)} нагрузка")
print(f"Факт: {actual_cpu:.4f} ядер → {get_load_label(actual_cpu)} нагрузка")

print("\nГотово!")