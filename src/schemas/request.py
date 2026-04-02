from pydantic import BaseModel, Field


class SaleRequest(BaseModel):
    date: str = Field(..., examples=["2017-08-16"], description="Дата в формате YYYY-MM-DD")
    store_nbr: int = Field(..., ge=1, le=54, examples=[1], description="ID магазина (1-54)")
    family: str = Field(..., examples=["GROCERY I"], description="Товарная категория")
    onpromotion: int = Field(0, ge=0, examples=[10], description="Кол-во товаров в промо")


class SaleResponse(BaseModel):
    store_nbr: int
    family: str
    date: str
    predicted_sales: float


class BatchSaleResponse(BaseModel):
    predictions: list[SaleResponse]
    count: int


class FeatureContribution(BaseModel):
    feature: str
    value: float | None = None
    shap_value: float


class ExplainResponse(BaseModel):
    store_nbr: int
    family: str
    date: str
    predicted_sales: float
    base_value: float
    contributions: list[FeatureContribution]
