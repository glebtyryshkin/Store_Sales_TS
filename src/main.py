from contextlib import asynccontextmanager
from pathlib import Path
import time

from fastapi import FastAPI

from .state import state
from .preprocessing.transforms import (
    StoreRegistry,
    OilRegistry,
    HolidayRegistry,
    EncoderRegistry,
)
from .preprocessing.features import HistoricalData
from .model.predictor import ModelPredictor
from .routers import predict

BASE_DIR = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    print("=" * 50)
    print("Loading model and reference data...")

    state.model = ModelPredictor(BASE_DIR / "artifacts" / "lgbm_tuned.pkl")
    state.encoders = EncoderRegistry(BASE_DIR / "artifacts" / "label_encoders.pkl")
    state.stores = StoreRegistry(BASE_DIR / "data" / "stores.csv")
    state.oil = OilRegistry(BASE_DIR / "data" / "oil.csv")
    state.holidays = HolidayRegistry(BASE_DIR / "data" / "holidays_events.csv")
    state.history = HistoricalData(BASE_DIR / "data" / "train.csv")

    elapsed = time.time() - t0
    print(f"Ready in {elapsed:.1f}s")
    print("=" * 50)
    yield


app = FastAPI(
    title="Store Sales Prediction API",
    description=(
        "Прогнозирование дневных продаж по магазинам и товарным категориям "
        "(Corporación Favorita). LightGBM, Optuna-tuned, Val RMSLE=0.3730."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(predict.router)


@app.get("/health", tags=["system"])
def health():
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "families": state.encoders.valid_values("family") if state.encoders else [],
        "stores": state.stores.valid_store_ids if state.stores else [],
    }
