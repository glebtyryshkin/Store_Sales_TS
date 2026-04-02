"""
Store Sales Forecast — Streamlit Dashboard.
Фронтенд для FastAPI-бэкенда (localhost:8000).
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import time

API_URL = "http://localhost:8000"

# ── Пресеты ──────────────────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {
    "— выбрать вручную —": {},
    "🛒 Продуктовый бум (GROCERY I, маг. 44, промо)": {
        "date": date(2017, 8, 18),
        "store_nbr": 44,
        "family": "GROCERY I",
        "onpromotion": 120,
    },
    "🥛 Молочка без промо (DAIRY, маг. 3)": {
        "date": date(2017, 8, 20),
        "store_nbr": 3,
        "family": "DAIRY",
        "onpromotion": 0,
    },
    "🍺 Алкоголь в выходной (LIQUOR,WINE,BEER, маг. 10)": {
        "date": date(2017, 8, 19),
        "store_nbr": 10,
        "family": "LIQUOR,WINE,BEER",
        "onpromotion": 5,
    },
    "🚗 Автотовары (AUTOMOTIVE, маг. 1)": {
        "date": date(2017, 8, 16),
        "store_nbr": 1,
        "family": "AUTOMOTIVE",
        "onpromotion": 0,
    },
    "🧹 Уборка + промо (CLEANING, маг. 50)": {
        "date": date(2017, 8, 25),
        "store_nbr": 50,
        "family": "CLEANING",
        "onpromotion": 30,
    },
    "💰 День зарплаты 15-е (BEVERAGES, маг. 25)": {
        "date": date(2017, 8, 15),
        "store_nbr": 25,
        "family": "BEVERAGES",
        "onpromotion": 15,
    },
}

BATCH_PRESETS: dict[str, dict] = {
    "— настроить вручную —": {},
    "📈 Топ-5 категорий, маг. 44, неделя": {
        "dates": (date(2017, 8, 16), date(2017, 8, 22)),
        "stores": [44],
        "families": ["GROCERY I", "BEVERAGES", "PRODUCE", "DAIRY", "MEATS"],
        "onpromotion": 10,
    },
    "🏪 3 магазина × GROCERY I, 15 дней": {
        "dates": (date(2017, 8, 16), date(2017, 8, 31)),
        "stores": [3, 25, 44],
        "families": ["GROCERY I"],
        "onpromotion": 0,
    },
    "🍞 Скоропорт за неделю (маг. 10)": {
        "dates": (date(2017, 8, 16), date(2017, 8, 22)),
        "stores": [10],
        "families": ["DAIRY", "BREAD/BAKERY", "DELI", "EGGS", "MEATS", "POULTRY", "SEAFOOD"],
        "onpromotion": 0,
    },
}


# ── API ──────────────────────────────────────────────────────────────────────


@st.cache_data(ttl=300, show_spinner=False)
def fetch_metadata() -> tuple[list[str], list[int], bool]:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        d = r.json()
        return d["families"], d["stores"], True
    except Exception:
        return [], [], False


def api_predict_explain(date_str: str, store_nbr: int, family: str, onpromotion: int):
    t0 = time.time()
    r = requests.post(
        f"{API_URL}/predict/explain",
        json={"date": date_str, "store_nbr": store_nbr, "family": family, "onpromotion": onpromotion},
        timeout=15,
    )
    elapsed = time.time() - t0
    r.raise_for_status()
    return r.json(), elapsed


def api_predict_batch(items: list[dict]):
    t0 = time.time()
    r = requests.post(f"{API_URL}/predict/batch", json=items, timeout=120)
    elapsed = time.time() - t0
    r.raise_for_status()
    return r.json(), elapsed


# ── State ────────────────────────────────────────────────────────────────────


def init_state():
    if "history" not in st.session_state:
        st.session_state.history = []


# ── Callbacks ────────────────────────────────────────────────────────────────


def _on_preset_change():
    """Пресет → обновляем session_state виджетов формы."""
    p = PRESETS.get(st.session_state.preset_select, {})
    if p:
        st.session_state.f_date = p["date"]
        st.session_state.f_store = p["store_nbr"]
        st.session_state.f_family = p["family"]
        st.session_state.f_promo = p["onpromotion"]


def _on_batch_preset_change():
    bp = BATCH_PRESETS.get(st.session_state.bp_select, {})
    if bp:
        st.session_state.b_dates = bp["dates"]
        st.session_state.b_stores = bp["stores"]
        st.session_state.b_families = bp["families"]
        st.session_state.b_promo = bp["onpromotion"]


# ── UI: Sidebar ──────────────────────────────────────────────────────────────


def render_sidebar(families: list[str], stores: list[int]):
    with st.sidebar:
        st.header("🛒 Параметры прогноза")

        st.selectbox(
            "Быстрый пример",
            options=list(PRESETS.keys()),
            index=0,
            key="preset_select",
            on_change=_on_preset_change,
        )

        st.divider()

        with st.form(key="predict_form"):
            pred_date = st.date_input(
                "Дата",
                value=date(2017, 8, 16),
                min_value=date(2013, 1, 1),
                max_value=date(2025, 12, 31),
                key="f_date",
            )

            store = st.selectbox(
                "Магазин",
                options=stores,
                index=0,
                key="f_store",
            )

            fam_default = families.index("GROCERY I") if "GROCERY I" in families else 0
            family = st.selectbox(
                "Категория",
                options=families,
                index=fam_default,
                key="f_family",
            )

            promo = st.number_input(
                "Товаров в промо",
                min_value=0,
                max_value=1000,
                value=0,
                step=1,
                key="f_promo",
            )

            submitted = st.form_submit_button("🔮 Получить прогноз", use_container_width=True)

        return pred_date, store, family, promo, submitted


# ── UI: SHAP chart ───────────────────────────────────────────────────────────


def _fmt_val(v):
    if v is None:
        return "—"
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return "—"
        return str(int(f)) if f == int(f) else f"{f:.2f}"
    except (TypeError, ValueError, OverflowError):
        return str(v)


FEATURE_RU = {
    "store_nbr": "Магазин",
    "family": "Категория",
    "onpromotion": "Промо",
    "city": "Город",
    "state": "Штат",
    "type": "Тип магазина",
    "cluster": "Кластер",
    "oil_price": "Цена нефти",
    "oil_ma_7": "Нефть MA-7",
    "day_of_week": "День недели",
    "day_of_month": "День месяца",
    "month": "Месяц",
    "week_of_year": "Неделя года",
    "quarter": "Квартал",
    "is_weekend": "Выходной",
    "year": "Год",
    "lag_7": "Лаг 7д",
    "lag_14": "Лаг 14д",
    "lag_28": "Лаг 28д",
    "lag_364": "Лаг 364д",
    "rolling_mean_7": "Ср. 7д",
    "rolling_mean_14": "Ср. 14д",
    "rolling_mean_30": "Ср. 30д",
    "rolling_median_7": "Медиана 7д",
    "is_holiday": "Праздник",
    "days_to_next_holiday": "До праздника",
    "days_since_last_holiday": "После праздника",
    "is_earthquake_period": "Землетрясение",
    "promo_rolling_mean_7": "Промо ср. 7д",
    "is_payday": "День зарплаты",
}


def render_shap_chart(contributions: list[dict], base_value: float):
    top = contributions[:15]
    df = pd.DataFrame(top)
    df = df.iloc[::-1]  # reverse for bottom-to-top display

    labels = [
        f"{FEATURE_RU.get(r['feature'], r['feature'])} = {_fmt_val(r['value'])}"
        for _, r in df.iterrows()
    ]
    colors = ["#e74c3c" if v < 0 else "#27ae60" for v in df["shap_value"]]

    fig = go.Figure(go.Bar(
        x=df["shap_value"],
        y=labels,
        orientation="h",
        marker_color=colors,
        hovertemplate="%{y}<br>SHAP: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Вклад признаков (SHAP, log-пространство) · base = {base_value:.4f}",
        xaxis_title="SHAP value (→ больше продаж | ← меньше продаж)",
        height=max(350, len(top) * 28 + 100),
        margin=dict(l=10, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── UI: Tab 1 — Одиночный прогноз ───────────────────────────────────────────


def render_single_tab(submitted: bool, pred_date, store, family, promo):
    if submitted:
        with st.spinner("Прогнозируем..."):
            try:
                result, elapsed = api_predict_explain(
                    pred_date.isoformat(), store, family, promo,
                )
            except requests.exceptions.HTTPError as e:
                st.error(f"API ошибка: {e.response.text}")
                return
            except Exception as e:
                st.error(f"Ошибка соединения: {e}")
                return

        sales = result["predicted_sales"]

        st.session_state.history.append({
            "store_nbr": result["store_nbr"],
            "family": result["family"],
            "date": result["date"],
            "predicted_sales": sales,
        })

        col_metric, col_store, col_time = st.columns([3, 1, 1])
        with col_metric:
            st.metric("Прогноз продаж", f"{sales:,.2f}", help="Единиц товара за день")
        with col_store:
            st.metric("Магазин", f"#{result['store_nbr']}")
        with col_time:
            st.metric("Время", f"{elapsed * 1000:.0f} мс")

        if sales > 500:
            st.success(f"**{result['family']}** на {result['date']} → **{sales:,.2f}** ед.")
        elif sales > 0:
            st.info(f"**{result['family']}** на {result['date']} → **{sales:,.2f}** ед.")
        else:
            st.warning(f"**{result['family']}** на {result['date']} → **0.00** (нулевые продажи)")

        with st.expander("🧠 SHAP — почему модель дала такой прогноз", expanded=True):
            st.caption(
                "Зелёные полосы **увеличивают** прогноз, красные — **уменьшают**. "
                "Значения в log-пространстве (модель предсказывает log1p(sales))."
            )
            render_shap_chart(result["contributions"], result["base_value"])

        with st.expander("📊 Детали запроса"):
            dc1, dc2 = st.columns(2)
            with dc1:
                st.markdown(f"""
| Параметр | Значение |
|----------|----------|
| Дата | `{result['date']}` |
| Магазин | `{result['store_nbr']}` |
| Категория | `{result['family']}` |
| Промо | `{promo}` |
""")
            with dc2:
                st.caption("Модель: LightGBM Tuned (Optuna)")
                st.caption("Val RMSLE: 0.3730")
                st.caption(f"Ответ за {elapsed * 1000:.0f} мс")

    else:
        st.info("👈 Выберите **быстрый пример** или заполните параметры в сайдбаре и нажмите «Получить прогноз»")

    _render_history()


def _render_history():
    hist = st.session_state.history
    if not hist:
        return

    with st.expander(f"📜 История прогнозов ({len(hist)})", expanded=False):
        col_tbl, col_btn = st.columns([5, 1])
        with col_btn:
            if st.button("🗑️ Очистить", key="clear_hist"):
                st.session_state.history = []
                st.rerun()

        df = pd.DataFrame(hist)
        st.dataframe(df, use_container_width=True, hide_index=True)

        if len(df) >= 2:
            fig = px.bar(
                df, x=df.index, y="predicted_sales", color="family",
                hover_data=["date", "store_nbr"],
                title="Прогнозы текущей сессии",
                labels={"predicted_sales": "Продажи", "index": "#", "family": "Категория"},
            )
            fig.update_layout(height=300, margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)


# ── UI: Tab 2 — Батч-прогноз ────────────────────────────────────────────────


def render_batch_tab(families: list[str], stores: list[int]):
    st.subheader("Батч-прогноз")
    st.caption("Прогноз для комбинаций дат × магазинов × категорий")

    st.selectbox(
        "Быстрый пример",
        options=list(BATCH_PRESETS.keys()),
        index=0,
        key="bp_select",
        on_change=_on_batch_preset_change,
    )

    c1, c2 = st.columns(2)
    with c1:
        date_range = st.date_input(
            "Диапазон дат",
            value=(date(2017, 8, 16), date(2017, 8, 22)),
            key="b_dates",
        )
        selected_stores = st.multiselect(
            "Магазины", options=stores, default=[1], key="b_stores",
        )

    with c2:
        selected_families = st.multiselect(
            "Категории", options=families,
            default=["GROCERY I", "BEVERAGES", "DAIRY"],
            key="b_families",
        )
        batch_promo = st.number_input(
            "Промо (для всех)", min_value=0, value=0, step=1, key="b_promo",
        )

    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.warning("Выберите начало и конец диапазона дат.")
        return
    if not selected_stores:
        st.warning("Выберите хотя бы один магазин.")
        return
    if not selected_families:
        st.warning("Выберите хотя бы одну категорию.")
        return

    start_d, end_d = date_range
    num_days = (end_d - start_d).days + 1
    total = num_days * len(selected_stores) * len(selected_families)

    if total > 1000:
        st.error(f"Слишком много комбинаций: **{total}** (макс. 1000). Сузьте выборку.")
        return

    st.write(f"Комбинаций: **{total}** ({num_days} дн. × {len(selected_stores)} маг. × {len(selected_families)} кат.)")

    if not st.button("🚀 Запустить батч", key="batch_go", use_container_width=True):
        return

    items = [
        {"date": (start_d + timedelta(days=d)).isoformat(), "store_nbr": s, "family": f, "onpromotion": batch_promo}
        for d in range(num_days)
        for s in selected_stores
        for f in selected_families
    ]

    with st.spinner(f"Прогнозируем {total} значений..."):
        try:
            result, elapsed = api_predict_batch(items)
        except requests.exceptions.HTTPError as e:
            st.error(f"API ошибка: {e.response.text}")
            return
        except Exception as e:
            st.error(f"Ошибка: {e}")
            return

    st.success(f"Готово за **{elapsed:.1f} с** — {result['count']} прогнозов")

    df = pd.DataFrame(result["predictions"])
    df["date"] = pd.to_datetime(df["date"])

    _render_batch_charts(df, selected_stores)

    with st.expander("📋 Таблица прогнозов"):
        st.dataframe(
            df.sort_values(["date", "store_nbr", "family"]),
            use_container_width=True, hide_index=True,
        )


def _render_batch_charts(df: pd.DataFrame, selected_stores: list[int]):
    if len(selected_stores) == 1:
        fig = px.line(
            df.sort_values("date"), x="date", y="predicted_sales", color="family",
            title=f"Прогноз продаж — магазин #{selected_stores[0]}",
            labels={"predicted_sales": "Продажи", "date": "Дата", "family": "Категория"},
            markers=True,
        )
    else:
        fig = px.line(
            df.sort_values("date"), x="date", y="predicted_sales", color="family",
            facet_col="store_nbr", facet_col_wrap=2,
            title="Прогноз продаж по магазинам",
            labels={"predicted_sales": "Продажи", "date": "Дата", "family": "Категория"},
            markers=True,
        )
    fig.update_layout(height=450, margin=dict(t=50, b=30))
    st.plotly_chart(fig, use_container_width=True)

    totals = (
        df.groupby("family", as_index=False)["predicted_sales"]
        .sum()
        .sort_values("predicted_sales", ascending=True)
    )
    fig2 = px.bar(
        totals, x="predicted_sales", y="family", orientation="h",
        title="Суммарный прогноз по категориям",
        labels={"predicted_sales": "Сумма продаж", "family": ""},
        color="predicted_sales", color_continuous_scale="Tealgrn",
    )
    fig2.update_layout(
        height=max(300, len(totals) * 30 + 80),
        margin=dict(t=50, b=20, l=10), showlegend=False,
    )
    fig2.update_coloraxes(showscale=False)
    st.plotly_chart(fig2, use_container_width=True)


# ── UI: Tab 3 — О модели ────────────────────────────────────────────────────


def render_about_tab():
    st.subheader("О модели")
    st.markdown("""
**Store Sales Prediction** — прогнозирование дневных продаж сети супермаркетов
Corporación Favorita (Эквадор).

| Параметр | Значение |
|----------|----------|
| Алгоритм | LightGBM (Optuna, 50 trials, 574 итерации) |
| Метрика (val) | RMSLE = **0.3730** |
| Признаков | 30 |
| Магазинов | 54 |
| Категорий | 33 |
| Горизонт | 15 дней |
""")

    with st.expander("📐 Полный список 30 признаков"):
        st.markdown("""
| Группа | Признаки |
|--------|----------|
| Идентификаторы | `store_nbr`, `family`, `onpromotion` |
| Магазин | `city`, `state`, `type`, `cluster` |
| Нефть | `oil_price`, `oil_ma_7` |
| Календарь | `day_of_week`, `day_of_month`, `month`, `week_of_year`, `quarter`, `is_weekend`, `year` |
| Лаги продаж | `lag_7`, `lag_14`, `lag_28`, `lag_364` |
| Скользящие | `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_30`, `rolling_median_7` |
| Праздники | `is_holiday`, `days_to_next_holiday`, `days_since_last_holiday` |
| Прочее | `is_earthquake_period`, `promo_rolling_mean_7`, `is_payday` |
""")

    with st.expander("🔌 API-эндпоинты"):
        st.code("GET  /health            — статус + метаданные", language="text")
        st.code("POST /predict           — одиночный прогноз", language="text")
        st.code("POST /predict/explain   — прогноз + SHAP values", language="text")
        st.code("POST /predict/batch     — батч до 1000 запросов", language="text")
        st.markdown("Swagger UI: [localhost:8000/docs](http://localhost:8000/docs)")

    st.markdown("""
---
**Пользователь вводит 4 параметра** (дата, магазин, категория, промо),
а сервер **вычисляет все 30 фичей на лету** из справочных таблиц и исторических данных,
включая **SHAP-объяснения** через `pred_contrib` LightGBM (без отдельной библиотеки shap).
""")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    st.set_page_config(page_title="Store Sales Forecast", page_icon="🛒", layout="wide")
    init_state()

    st.title("🛒 Store Sales Forecast")

    families, stores, api_ok = fetch_metadata()

    if not api_ok:
        st.error("⚠️ API-сервер недоступен (`localhost:8000`).")
        st.markdown("Запустите бэкенд:")
        st.code(
            "cd d:\\sst\n.\\venv\\Scripts\\activate\nuvicorn src.main:app --host 0.0.0.0 --port 8000",
            language="powershell",
        )
        st.button("🔄 Повторить", on_click=fetch_metadata.clear, key="retry")
        return

    pred_date, store, family, promo, submitted = render_sidebar(families, stores)

    tab1, tab2, tab3 = st.tabs(["🔮 Прогноз", "📊 Батч-анализ", "ℹ️ О модели"])

    with tab1:
        render_single_tab(submitted, pred_date, store, family, promo)
    with tab2:
        render_batch_tab(families, stores)
    with tab3:
        render_about_tab()


if __name__ == "__main__":
    main()
