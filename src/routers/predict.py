from fastapi import APIRouter, HTTPException
import pandas as pd

from ..state import state
from ..schemas.request import SaleRequest, SaleResponse, BatchSaleResponse, ExplainResponse
from ..preprocessing.features import compute_features

router = APIRouter(tags=["predictions"])


@router.post("/predict", response_model=SaleResponse)
def predict_single(req: SaleRequest):
    try:
        features = compute_features(
            req.date, req.store_nbr, req.family, req.onpromotion,
            state.stores, state.oil, state.holidays, state.encoders, state.history,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    predicted = state.model.predict(features)

    return SaleResponse(
        store_nbr=req.store_nbr,
        family=req.family,
        date=req.date,
        predicted_sales=round(predicted, 2),
    )


@router.post("/predict/batch", response_model=BatchSaleResponse)
def predict_batch(requests: list[SaleRequest]):
    if not requests:
        raise HTTPException(status_code=422, detail="Empty request list")
    if len(requests) > 1000:
        raise HTTPException(status_code=422, detail="Max 1000 items per batch")

    feature_rows = []
    for req in requests:
        try:
            features = compute_features(
                req.date, req.store_nbr, req.family, req.onpromotion,
                state.stores, state.oil, state.holidays, state.encoders, state.history,
            )
            feature_rows.append(features)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    df = pd.DataFrame(feature_rows)
    preds = state.model.predict_batch(df)

    predictions = [
        SaleResponse(
            store_nbr=req.store_nbr,
            family=req.family,
            date=req.date,
            predicted_sales=round(pred, 2),
        )
        for req, pred in zip(requests, preds)
    ]

    return BatchSaleResponse(predictions=predictions, count=len(predictions))


@router.post("/predict/explain", response_model=ExplainResponse)
def predict_explain(req: SaleRequest):
    try:
        features = compute_features(
            req.date, req.store_nbr, req.family, req.onpromotion,
            state.stores, state.oil, state.holidays, state.encoders, state.history,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    pred, base, contribs = state.model.predict_explain(features)

    return ExplainResponse(
        store_nbr=req.store_nbr,
        family=req.family,
        date=req.date,
        predicted_sales=round(pred, 2),
        base_value=round(base, 4),
        contributions=contribs,
    )
