import joblib
import numpy as np
import pandas as pd
from pathlib import Path


class ModelPredictor:
    def __init__(self, model_path: Path):
        self._model = joblib.load(model_path)
        self._is_booster = hasattr(self._model, "feature_name") and not hasattr(
            self._model, "fit"
        )
        if self._is_booster:
            self._best_iter = getattr(self._model, "best_iteration", None)
            print(f"  Model type: LightGBM Booster, best_iteration={self._best_iter}")
        else:
            self._best_iter = getattr(self._model, "best_iteration_", None)
            print(f"  Model type: sklearn-like, best_iteration_={self._best_iter}")

    def predict(self, features: dict) -> float:
        df = pd.DataFrame([features])
        return self.predict_batch(df)[0]

    def predict_batch(self, df: pd.DataFrame) -> list[float]:
        if self._is_booster:
            pred_log = self._model.predict(df, num_iteration=self._best_iter)
        else:
            pred_log = self._model.predict(df)
        pred = np.expm1(pred_log).clip(0)
        return pred.tolist()

    def predict_explain(self, features: dict) -> tuple[float, float, list[dict]]:
        df = pd.DataFrame([features])

        if self._is_booster:
            pred_log = self._model.predict(df, num_iteration=self._best_iter)[0]
            contribs = self._model.predict(df, pred_contrib=True, num_iteration=self._best_iter)
            feat_names = self._model.feature_name()
        else:
            pred_log = self._model.predict(df)[0]
            booster = self._model.booster_
            contribs = booster.predict(df, pred_contrib=True)
            feat_names = booster.feature_name()

        shap_vals = contribs[0, :-1]
        base = float(contribs[0, -1])
        pred = float(max(np.expm1(pred_log), 0))

        result = []
        for name, sv in zip(feat_names, shap_vals):
            fv = features.get(name)
            # numpy floats and plain float NaN → None (JSON null)
            try:
                if fv is not None and np.isnan(float(fv)):
                    fv = None
                elif fv is not None:
                    fv = float(fv)
            except (TypeError, ValueError):
                fv = None
            result.append({"feature": name, "value": fv, "shap_value": round(float(sv), 6)})

        result.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return pred, base, result
