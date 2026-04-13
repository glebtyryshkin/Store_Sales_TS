import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path


def _insert_christmas_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Match `02_feature_engineering`: missing 25 Dec rows → sales=0 for lag continuity."""
    christmas_dates = pd.to_datetime(
        ["2013-12-25", "2014-12-25", "2015-12-25", "2016-12-25"]
    )
    existing_dates = df["date"].unique()
    missing_xmas = [d for d in christmas_dates if d not in existing_dates]
    if not missing_xmas:
        return df
    store_family = df[["store_nbr", "family"]].drop_duplicates()
    rows = []
    for d in missing_xmas:
        chunk = store_family.copy()
        chunk["date"] = d
        chunk["sales"] = 0.0
        chunk["onpromotion"] = 0
        rows.append(chunk)
    xmas_df = pd.concat(rows, ignore_index=True)
    out = pd.concat([df, xmas_df], ignore_index=True)
    return out.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)


class HistoricalData:
    """Sales & promo history per (store_nbr, family) for lag/rolling computation."""

    def __init__(
        self,
        train_path: Path,
        max_date: pd.Timestamp | str | None = None,
    ):
        """max_date: если задан — только строки с date <= max_date (для рекурсивной валидации val)."""
        print(f"  Loading train.csv for historical data...")
        df = pd.read_csv(
            train_path,
            parse_dates=["date"],
            usecols=["date", "store_nbr", "family", "sales", "onpromotion"],
            dtype={"store_nbr": "int16", "onpromotion": "int16"},
        )
        df = _insert_christmas_gaps(df)
        df = df.sort_values(["store_nbr", "family", "date"])
        if max_date is not None:
            md = pd.Timestamp(max_date).normalize()
            df = df.loc[df["date"] <= md].copy()
            print(f"  Truncated to date <= {md.date()} ({len(df):,} rows)")

        self._sales: dict[tuple[int, str], pd.Series] = {}
        self._promo: dict[tuple[int, str], pd.Series] = {}

        for (snbr, fam), grp in df.groupby(["store_nbr", "family"]):
            indexed = grp.set_index("date").sort_index()
            self._sales[(int(snbr), fam)] = indexed["sales"]
            self._promo[(int(snbr), fam)] = indexed["onpromotion"]

        print(f"  Loaded {len(self._sales)} time series")

    def record_observation(
        self,
        store_nbr: int,
        family: str,
        date: pd.Timestamp,
        sales: float,
        onpromotion: int,
    ) -> None:
        """Append one day (e.g. recursive test forecast) so lags/rolling stay causal."""
        key = (store_nbr, family)
        d = pd.Timestamp(date).normalize()
        if key not in self._sales:
            self._sales[key] = pd.Series(dtype=float)
        if key not in self._promo:
            self._promo[key] = pd.Series(dtype=float)
        self._sales[key].loc[d] = float(sales)
        self._sales[key] = self._sales[key].sort_index()
        self._promo[key].loc[d] = int(onpromotion)
        self._promo[key] = self._promo[key].sort_index()

    def get_lag(self, store_nbr: int, family: str, date: pd.Timestamp, lag_days: int) -> float:
        series = self._sales.get((store_nbr, family))
        if series is None:
            return np.nan
        target = date - timedelta(days=lag_days)
        if target in series.index:
            return float(series[target])
        return np.nan

    def get_rolling_stats(
        self, store_nbr: int, family: str, date: pd.Timestamp, window: int
    ) -> tuple[float, float]:
        """Return (mean, median) of sales in [date-window, date-1]."""
        series = self._sales.get((store_nbr, family))
        if series is None:
            return np.nan, np.nan
        end = date - timedelta(days=1)
        start = date - timedelta(days=window)
        vals = series.loc[(series.index >= start) & (series.index <= end)].values
        if len(vals) == 0:
            return np.nan, np.nan
        return float(np.mean(vals)), float(np.median(vals))

    def get_promo_rolling(
        self, store_nbr: int, family: str, date: pd.Timestamp, onpromotion: int, window: int = 7
    ) -> float:
        """Rolling mean of onpromotion over [date-6, date] (current day included)."""
        series = self._promo.get((store_nbr, family))
        start = date - timedelta(days=window - 1)
        end = date - timedelta(days=1)

        if series is not None:
            vals = series.loc[(series.index >= start) & (series.index <= end)].values.tolist()
        else:
            vals = []
        vals.append(onpromotion)
        return float(np.mean(vals))


def compute_features(
    date_str: str,
    store_nbr: int,
    family: str,
    onpromotion: int,
    stores,
    oil,
    holidays,
    encoders,
    history: HistoricalData,
) -> dict:
    date = pd.Timestamp(date_str)

    store = stores.get(store_nbr)
    if store is None:
        raise ValueError(f"Unknown store_nbr={store_nbr}. Valid: {stores.valid_store_ids}")

    city, st, type_, cluster = store["city"], store["state"], store["type"], store["cluster"]

    oil_price, oil_ma_7 = oil.get(date)

    day_of_week = date.dayofweek
    day_of_month = date.day
    month = date.month
    week_of_year = int(date.isocalendar()[1])
    quarter = date.quarter
    is_weekend = int(day_of_week >= 5)
    year = date.year

    lag_7 = history.get_lag(store_nbr, family, date, 7)
    lag_14 = history.get_lag(store_nbr, family, date, 14)
    lag_28 = history.get_lag(store_nbr, family, date, 28)
    lag_364 = history.get_lag(store_nbr, family, date, 364)

    rm7_mean, rm7_med = history.get_rolling_stats(store_nbr, family, date, 7)
    rm14_mean, _ = history.get_rolling_stats(store_nbr, family, date, 14)
    rm30_mean, _ = history.get_rolling_stats(store_nbr, family, date, 30)

    is_hol = holidays.is_holiday(date, city, st)
    days_to_next, days_since = holidays.holiday_distances(date)

    is_earthquake = int(pd.Timestamp("2016-04-16") <= date <= pd.Timestamp("2016-05-15"))
    promo_rm7 = history.get_promo_rolling(store_nbr, family, date, onpromotion)
    is_payday = int(date.is_month_end or day_of_month == 15)

    family_enc = encoders.encode("family", family)
    if family_enc == -1:
        raise ValueError(f"Unknown family='{family}'. Valid: {encoders.valid_values('family')}")
    city_enc = encoders.encode("city", city)
    state_enc = encoders.encode("state", st)
    type_enc = encoders.encode("type", type_)

    return {
        "store_nbr": store_nbr,
        "family": family_enc,
        "onpromotion": onpromotion,
        "city": city_enc,
        "state": state_enc,
        "type": type_enc,
        "cluster": cluster,
        "oil_price": oil_price,
        "day_of_week": day_of_week,
        "day_of_month": day_of_month,
        "month": month,
        "week_of_year": week_of_year,
        "quarter": quarter,
        "is_weekend": is_weekend,
        "year": year,
        "lag_7": lag_7,
        "lag_14": lag_14,
        "lag_28": lag_28,
        "lag_364": lag_364,
        "rolling_mean_7": rm7_mean,
        "rolling_mean_14": rm14_mean,
        "rolling_mean_30": rm30_mean,
        "rolling_median_7": rm7_med,
        "oil_ma_7": oil_ma_7,
        "is_holiday": is_hol,
        "days_to_next_holiday": days_to_next,
        "days_since_last_holiday": days_since,
        "is_earthquake_period": is_earthquake,
        "promo_rolling_mean_7": promo_rm7,
        "is_payday": is_payday,
    }
