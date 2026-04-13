# Store Sales Forecasting

Энд-ту-энд проект по прогнозированию дневных продаж для соревнования Kaggle **Store Sales**.
Кейс близок к продакшену: feature engineering, валидация по времени без утечек, рекурсивный инференс,
FastAPI, Docker и разбор расхождения между тем, как мы считали метрику offline, и тем, как модель реально шагает по горизонту.


| Блок                 | Значение                                                                      |
| -------------------- | ----------------------------------------------------------------------------- |
| Задача               | Регрессия по `store × family` с горизонтом 15 дней                            |
| Модель               | Глобальный LightGBM на `log1p(sales)`                                         |
| Лучший offline       | `Val RMSLE = 0.3730` на фичах с лагами по факту (оптимистично, лаги из реальных продаж) |
| Рекурсивный offline  | `~0.397` — реалистичный режим, лаги строятся из предсказаний                  |
| Публичный leaderboard | `~0.58` — разрыв объяснён в разделе «Валидация и разбор ошибки»              |
| API                  | `FastAPI` + `/health`, `/predict`, `/predict/batch`, `/predict/explain`       |
| Главный инсайт       | Если train и serve считаются по-разному, offline-метрика легко врёт           |


## Архитектура

```mermaid
flowchart LR
    A[train.csv / stores.csv / oil.csv / holidays_events.csv] --> B[HistoricalData + registries]
    B --> C[compute_features()]
    C --> D[LightGBM predictor]
    D --> E[FastAPI]
    E --> F[Streamlit / CLI clients]
```



## Данные

Таргет:

```text
sales — суммарные продажи товарной семьи в конкретном магазине за конкретную дату.
```

Источники данных:

- `train.csv` - история продаж и промо.
- `test.csv` - те же признаки без `sales`.
- `stores.csv` - город, штат, тип магазина, кластер.
- `oil.csv` - ежедневная цена нефти.
- `holidays_events.csv` - праздники и события.

Ключевые наблюдения по данным:

- Около 31.3% строк имеют `sales = 0`; для этой задачи это нормально.
- Пропуски на `25 Dec` заполняются нулями до расчёта лагов, чтобы ряд оставался непрерывным для лагов.
- Категориальные признаки магазина и семьи кодируются отдельно, энкодеры fit только на train.

## Фичи

После feature engineering используется 30 признаков. Основные группы:


| Группа          | Примеры                                                                                                                       | Зачем                            |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| Лаги            | `lag_7`, `lag_14`, `lag_28`, `lag_364`                                                                                        | Сильная автокорреляция спроса    |
| Rolling         | `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_30`, `rolling_median_7`                                                    | Сглаживание шума                 |
| Календарь       | `day_of_week`, `month`, `week_of_year`, `quarter`, `is_weekend`, `year`                                                       | Сезонность и календарные эффекты |
| Внешние сигналы | `oil_price`, `oil_ma_7`, `is_holiday`, `days_to_next_holiday`, `days_since_last_holiday`, `is_payday`, `is_earthquake_period` | Шоки спроса и макро-факторы      |
| Магазин         | `city`, `state`, `type`, `cluster`                                                                                            | Различия между локациями         |
| Промо           | `onpromotion`, `promo_rolling_mean_7`                                                                                         | Краткосрочные всплески спроса    |


Что реально сработало:

- `rolling_mean_7` стал одним из самых сильных признаков по gain.
- `lag_7` на train имел корреляцию с `sales` около 0.94.
- Гипотеза про сложную логику `transferred` для праздников не подтвердилась, поэтому оставлен простой holiday flag.
- Лаги нефти длиннее 7 дней почти не давали сигнала, поэтому остались текущая цена и `oil_ma_7`.
- `is_payday` полезен как дешёвый календарный сигнал.

## Моделирование

### Baseline

Базовая схема - один глобальный LightGBM по всем `(store_nbr, family)` с целевым преобразованием `log1p(sales)`.

Почему именно так:

- RMSLE по своей природе согласуется с лог-пространством.
- `log1p` смягчает тяжёлый правый хвост распределения продаж; нули при этом остаются валидными в лог-таргете.
- Глобальная модель проще в поддержке, чем отдельная модель на каждый ряд.
- На этой задаче доминируют lag/rolling признаки, а не сложная модельная архитектура.

### Сравнение экспериментов


| Версия                           | Val RMSLE  | Комментарий          |
| -------------------------------- | ---------- | -------------------- |
| LightGBM, параметры по умолчанию | 0.3749     | Около 397 итераций   |
| **LightGBM + Optuna**            | **0.3730** | Финальная модель     |
| LightGBM Tweedie + Optuna        | 0.3869     | Проиграл log1p + MSE |


Что я сознательно не делал:

- Не запускал XGBoost: при текущем наборе фичей ожидаемый выигрыш был бы меньше стоимости экспериментов.
- Не строил Prophet / statsmodels-пайплайн: один глобальный табличный бустинг оказался проще и практичнее.
- Не собирал ансамбль из сильно похожих моделей: низкое разнообразие моделей не оправдывало усложнение.
- Не делал two-stage `zero / non-zero` classifier: выигрыш был бы слишком маленьким относительно сложности.

## Валидация и разбор ошибки

### Где был провал

В `02_feature_engineering.ipynb` train и test склеивались в один ряд, после чего лаги и rolling-фичи считались на этом объединённом наборе.
На тестовом горизонте сильные признаки `lag_7`, `rolling_mean_7` и похожие начинали смотреть в окно, где фактических продаж ещё нет,
и превращались в NaN. В результате модель получала режим, на котором она не обучалась.

### Как исправлено

В `05_submission.ipynb` прогноз стал рекурсивным по дням:

1. вычисляю фичи из текущей истории;
2. предсказываю следующий день;
3. записываю предсказание обратно в историю;
4. использую его для следующих лагов.

Тот же причинно-корректный расчёт фичей используется в `src/preprocessing/features.py` через `HistoricalData` и `compute_features`.
FastAPI поднимает тот же код при инференсе, а не отдельную «упрощённую» копию.

### Что это дало

- `~0.397` в честной рекурсивной валидации.
- На публичном leaderboard скор заметно ниже, чем у «оптимистичной» offline-валидации на фичах с лагами по факту.
- Главный вывод: для табличной модели с сильными лагами нельзя верить метрике, если фичи на валидации ближе к «оракулу», чем к реальному шагу вперёд.

## Сервинг и объяснимость

Сервис стартует из `src/main.py` и на запуске загружает:

- `artifacts/lgbm_tuned.pkl`
- `artifacts/label_encoders.pkl`
- `data/stores.csv`
- `data/oil.csv`
- `data/holidays_events.csv`
- `data/train.csv`

Схема API:

- `GET /health` - статус и метаданные.
- `POST /predict` - одиночный прогноз.
- `POST /predict/batch` - батч до 1000 запросов.
- `POST /predict/explain` - прогноз + feature contributions.

`Streamlit`-дашборд в `app.py` использует тот же API и умеет показывать SHAP вклад признаков через `pred_contrib` LightGBM,
без отдельной библиотеки `shap`.

## Как запустить

### 1. Создать окружение

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Скачать данные Kaggle

```powershell
kaggle competitions download -c store-sales-time-series-forecasting
Expand-Archive .\store-sales-time-series-forecasting.zip -DestinationPath .\data
```

Нужны как минимум:

- `train.csv`
- `test.csv`
- `stores.csv`
- `oil.csv`
- `holidays_events.csv`
- `sample_submission.csv`

### 3. Поднять API

```powershell
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI:

- `http://localhost:8000/docs`

### 4. Поднять Streamlit

```powershell
streamlit run app.py --server.port 8501
```

### 5. Запустить через Docker

```powershell
docker compose up --build
docker compose down
```

Для контейнера нужны `data/` и `artifacts/` на хосте.

## Структура репозитория

```text
Store_Sales_TS/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_advanced_models.ipynb
│   └── 05_submission.ipynb
├── src/
│   ├── preprocessing/
│   ├── model/
│   ├── routers/
│   └── schemas/
├── app.py
├── artifacts/
├── data/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── requirements-api.txt
```

## Что бы я улучшил дальше

- Прогнал бы backtest на нескольких временных окнах, а не только на одном holdout.
- Увеличил бы окно лагов до более «безопасного» горизонта, чтобы слабее зависеть от ошибки на предыдущих шагах.
- Добавил бы `pytest`-тесты на совпадение фичей между notebook и API.
- Попробовал бы CatBoost или очень простой ensemble как следующий итерационный шаг.

## Полезные ссылки

- [Kaggle competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview)
- [Competition data](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)
- [Leaderboard](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/leaderboard)
- [LightGBM docs](https://lightgbm.readthedocs.io/)
- [FastAPI docs](https://fastapi.tiangolo.com/)

