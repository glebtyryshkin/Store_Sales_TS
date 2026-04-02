import pandas as pd
import numpy as np
import joblib
from pathlib import Path


class StoreRegistry:
    """Lookup: store_nbr -> {city, state, type, cluster}."""

    def __init__(self, stores_path: Path):
        df = pd.read_csv(stores_path)
        self._data: dict[int, dict] = {}
        for _, row in df.iterrows():
            self._data[int(row["store_nbr"])] = {
                "city": str(row["city"]),
                "state": str(row["state"]),
                "type": str(row["type"]),
                "cluster": int(row["cluster"]),
            }

    def get(self, store_nbr: int) -> dict | None:
        return self._data.get(store_nbr)

    @property
    def valid_store_ids(self) -> list[int]:
        return sorted(self._data.keys())


class OilRegistry:
    """Lookup: date -> (oil_price, oil_ma_7)."""

    def __init__(self, oil_path: Path):
        df = pd.read_csv(oil_path, parse_dates=["date"])
        df = df.rename(columns={"dcoilwtico": "oil_price"})
        df = df.sort_values("date").reset_index(drop=True)
        df["oil_price"] = df["oil_price"].ffill().bfill()
        df["oil_ma_7"] = df["oil_price"].rolling(7, min_periods=1).mean()
        self._data = df.set_index("date")[["oil_price", "oil_ma_7"]]

    def get(self, date: pd.Timestamp) -> tuple[float, float]:
        if date in self._data.index:
            row = self._data.loc[date]
            return float(row["oil_price"]), float(row["oil_ma_7"])
        mask = self._data.index <= date
        if mask.any():
            row = self._data.loc[mask].iloc[-1]
            return float(row["oil_price"]), float(row["oil_ma_7"])
        row = self._data.iloc[0]
        return float(row["oil_price"]), float(row["oil_ma_7"])


class HolidayRegistry:
    """Holiday flags and distance features."""

    def __init__(self, holidays_path: Path):
        df = pd.read_csv(holidays_path, parse_dates=["date"])

        self._national_dates: set[pd.Timestamp] = set(
            df.loc[df["locale"] == "National", "date"].unique()
        )
        self._regional: set[tuple[pd.Timestamp, str]] = set(
            zip(
                df.loc[df["locale"] == "Regional", "date"],
                df.loc[df["locale"] == "Regional", "locale_name"],
            )
        )
        self._local: set[tuple[pd.Timestamp, str]] = set(
            zip(
                df.loc[df["locale"] == "Local", "date"],
                df.loc[df["locale"] == "Local", "locale_name"],
            )
        )
        self._holiday_dates = np.sort(
            df["date"].unique().to_numpy().astype("datetime64[D]")
        )

    def is_holiday(self, date: pd.Timestamp, city: str, state: str) -> int:
        if date in self._national_dates:
            return 1
        if (date, state) in self._regional:
            return 1
        if (date, city) in self._local:
            return 1
        return 0

    def holiday_distances(self, date: pd.Timestamp) -> tuple[int, int]:
        d = np.datetime64(date, "D")
        idx = np.searchsorted(self._holiday_dates, d, side="left")

        if idx < len(self._holiday_dates):
            days_to_next = int(
                (self._holiday_dates[idx] - d) / np.timedelta64(1, "D")
            )
        else:
            days_to_next = 999

        if idx < len(self._holiday_dates) and self._holiday_dates[idx] == d:
            days_since = 0
        elif idx > 0:
            days_since = int(
                (d - self._holiday_dates[idx - 1]) / np.timedelta64(1, "D")
            )
        else:
            days_since = 999

        return min(days_to_next, 999), min(days_since, 999)


class EncoderRegistry:
    """LabelEncoder lookup: col + string_value -> int code."""

    def __init__(self, encoders_path: Path):
        self._encoders: dict = joblib.load(encoders_path)
        self._mappings: dict[str, dict[str, int]] = {}
        for col, le in self._encoders.items():
            self._mappings[col] = {
                cls: idx for idx, cls in enumerate(le.classes_)
            }

    def encode(self, col: str, value: str) -> int:
        return self._mappings.get(col, {}).get(value, -1)

    def valid_values(self, col: str) -> list[str]:
        if col in self._encoders:
            return list(self._encoders[col].classes_)
        return []
