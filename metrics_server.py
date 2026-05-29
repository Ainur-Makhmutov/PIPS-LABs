import os
import time
import json
import threading
import psutil
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import ast  # добавьте в начало файла

# ===== Импорт модели CatBoost =====
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===== Конфигурация =====
HISTORY_SIZE = 200           # храним последние 200 срезов метрик
UPDATE_INTERVAL = 30         # интервал обновления метрик (сек)
CSV_FILE = "Data/Dataset-24.csv"     # путь к вашему файлу данных
MODEL_FILE = "model.cbm"     # файл сохранённой модели (если есть)
TRAIN_RATIO = 0.8            # доля обучающей выборки (если модель не найдена)

# ===== Глобальное хранилище метрик =====
class MetricsStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.current = {
            "timestamp": None,
            "infra": {},
            "service": {},
            "business": {}
        }
        self.history = deque(maxlen=HISTORY_SIZE)

    def update(self, metrics_dict):
        with self.lock:
            metrics_dict["timestamp"] = datetime.now().isoformat()
            self.current = metrics_dict
            self.history.append(metrics_dict)

    def get_current(self):
        with self.lock:
            return self.current.copy()

    def get_history(self):
        with self.lock:
            return list(self.history)

store = MetricsStore()

# ===== Загрузка данных и обучение/загрузка модели =====


def load_and_prepare_data(csv_path):
    """Загружает CSV, извлекает среднее использование CPU из JSON-подобной строки"""
    df = pd.read_csv(csv_path)

    # Функция безопасного парсинга
    def parse_usage(x):
        if pd.isna(x):
            return {'cpus': 0.0, 'memory': 0.0}
        try:
            # ast.literal_eval работает с одинарными кавычками
            return ast.literal_eval(x)
        except:
            # запасной вариант: замена кавычек
            x_fixed = x.replace("'", '"')
            return json.loads(x_fixed)

    df['avg_cpu'] = df['average_usage'].apply(lambda x: parse_usage(x)['cpus'])
    df['timestamp'] = pd.to_datetime(df['start_time'], unit='us')
    df = df.sort_values(['collection_id', 'timestamp'])
    df['cpu_lag1'] = df.groupby('collection_id')['avg_cpu'].shift(1)
    df = df.dropna()
    return df

def train_or_load_model(df):
    if os.path.exists(MODEL_FILE):
        print("Загрузка существующей модели из файла", MODEL_FILE)
        model = CatBoostRegressor()
        model.load_model(MODEL_FILE)
        return model
    else:
        print("Модель не найдена. Обучаем новую...")
        # Используем лаг CPU как признак, целевую переменную – текущий CPU
        X = df[['cpu_lag1']].values
        y = df['avg_cpu'].values
        split = int(TRAIN_RATIO * len(X))
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
        model = CatBoostRegressor(iterations=100, depth=6, learning_rate=0.05, verbose=0)
        model.fit(X_train, y_train)
        # Сохраняем модель
        model.save_model(MODEL_FILE)
        print(f"Модель обучена. MAE на тесте: {mean_absolute_error(y_test, model.predict(X_test)):.4f}")
        # Сохраняем тестовые данные для вычисления бизнес-метрик
        global test_features, test_targets
        test_features = X_test
        test_targets = y_test
        return model

# Загружаем данные и модель
print("Загрузка данных...")
data_df = load_and_prepare_data(CSV_FILE)
print(f"Загружено {len(data_df)} записей")

model = train_or_load_model(data_df)

# Для эмуляции потока прогнозов будем использовать тестовые данные (если есть)
# Если модель была загружена, создадим тестовый набор из последних 10% данных
test_features = None
test_targets = None
if not os.path.exists(MODEL_FILE):
    # уже созданы внутри train_or_load_model
    pass
else:
    # Создаём тестовый набор для оценки качества
    X_all = data_df[['cpu_lag1']].values
    y_all = data_df['avg_cpu'].values
    split = int(TRAIN_RATIO * len(X_all))
    test_features = X_all[split:]
    test_targets = y_all[split:]

# Индекс для циклического прохода по тестовым данным
test_index = 0
test_predictions = deque(maxlen=1000)
test_actuals = deque(maxlen=1000)

def get_next_prediction_actual():
    """Возвращает следующую пару (прогноз, факт) из тестовых данных"""
    global test_index
    if test_features is None or len(test_features) == 0:
        return None, None
    x = test_features[test_index].reshape(1, -1)
    pred = model.predict(x)[0]
    actual = test_targets[test_index]
    test_index = (test_index + 1) % len(test_features)
    return pred, actual

# ===== Функции сбора метрик =====
def collect_infra_metrics():
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "ram_used_mb": psutil.virtual_memory().used / (1024 * 1024),
        "ram_percent": psutil.virtual_memory().percent
    }

def collect_service_metrics():
    # Размер модели (если сохранена)
    model_size_kb = 0
    if os.path.exists(MODEL_FILE):
        model_size_kb = os.path.getsize(MODEL_FILE) / 1024
    # Время инференса – будем измерять на одном объекте (имитация)
    if test_features is not None and len(test_features) > 0:
        x = test_features[0].reshape(1, -1)
        start = time.perf_counter()
        _ = model.predict(x)
        inference_time_ms = (time.perf_counter() - start) * 1000
    else:
        inference_time_ms = 0.0
    return {
        "model_size_kb": round(model_size_kb, 2),
        "inference_time_ms": round(inference_time_ms, 4),
        "throughput_rps": 0.0,  # будет обновляться отдельно
        "last_retrain_time": None
    }

def collect_business_metrics():
    """Вычисляет MAE, RMSE, MAPE, R2 на скользящем окне последних 100 прогнозов"""
    if len(test_predictions) < 10:
        return {
            "mae": None,
            "rmse": None,
            "mape": None,
            "r2": None,
            "total_predictions": len(test_predictions)
        }
    preds = np.array(test_predictions)
    actuals = np.array(test_actuals)
    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    # MAPE: избегаем деления на ноль
    mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-8))) * 100
    r2 = r2_score(actuals, preds)
    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "mape": round(mape, 2),
        "r2": round(r2, 4),
        "total_predictions": len(test_predictions)
    }

# Фоновый поток для обновления метрик и эмуляции прогнозов
def background_metrics_updater():
    while True:
        # 1. Получаем очередной прогноз и фактическое значение (эмуляция)
        pred, actual = get_next_prediction_actual()
        if pred is not None:
            test_predictions.append(pred)
            test_actuals.append(actual)
        # 2. Собираем метрики
        infra = collect_infra_metrics()
        service = collect_service_metrics()
        business = collect_business_metrics()
        # 3. Обновляем хранилище
        store.update({
            "infra": infra,
            "service": service,
            "business": business
        })
        time.sleep(UPDATE_INTERVAL)

# Запускаем фоновый поток
thread = threading.Thread(target=background_metrics_updater, daemon=True)
thread.start()

# ===== Создание FastAPI приложения =====
app = FastAPI(title="Metrics API for Prediction System")

# Разрешаем CORS для Grafana
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def get_metrics():
    """Текущие метрики"""
    return store.get_current()

@app.get("/history")
def get_history(limit: int = 100):
    """История метрик (последние limit записей)"""
    hist = store.get_history()
    return hist[-limit:]

@app.get("/all")
def get_all():
    """Все данные: текущие метрики + история"""
    return {
        "current": store.get_current(),
        "history": store.get_history()
    }

# Дополнительный эндпоинт для проверки модели
@app.get("/predict_sample")
def predict_sample():
    """Возвращает последний сделанный прогноз и фактическое значение"""
    if len(test_predictions) == 0:
        return {"error": "no predictions yet"}
    return {
        "last_prediction": float(test_predictions[-1]),
        "last_actual": float(test_actuals[-1]) if test_actuals else None,
        "window_mae": store.get_current().get("business", {}).get("mae")
    }

if __name__ == "__main__":
    print("Запуск сервера метрик на http://localhost:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002)