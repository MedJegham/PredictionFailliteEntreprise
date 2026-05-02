"""
Streamlit dashboard - CRISP-DM (Deep Learning)
Project: Prediction de la faillite d'entreprise

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_TITLE = "CRISP-DM – Prédiction de la Faillite d'Entreprise (Deep Learning)"
DATA_PATH = Path("data.csv")
OUTPUT_DIR = Path("output")
TARGET_COL = "Bankrupt?"

MODEL_PATH = OUTPUT_DIR / "best_model.keras"
SCALER_PATH = OUTPUT_DIR / "scaler.joblib"
RESULTS_CSV_DL = OUTPUT_DIR / "model_results.csv"
RESULTS_CSV_ML = OUTPUT_DIR / "model_results_ml.csv"
RUN_META = OUTPUT_DIR / "run_metadata.json"


def inject_custom_css() -> None:
    """Lightweight custom CSS for a modern dashboard look."""
    st.markdown(
        """
        <style>
        .stApp {
            background: #F7F9FC;
            color: #0F172A;
        }
        .block-container {padding-top: 1.0rem; padding-bottom: 2rem;}
        h1, h2, h3 {color: #0B1220 !important;}
        p, li, div, label, span {color: #111827;}
        .app-card {
            background: #FFFFFF;
            border: 1px solid #DDE4EE;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
        }
        .kpi-card {
            border: 1px solid #DDE4EE;
            border-radius: 12px;
            padding: 0.8rem 0.9rem;
            background: #ffffff;
            min-height: 96px;
            box-shadow: 0 1px 6px rgba(15, 23, 42, 0.04);
        }
        .small-muted {color: #334155; font-size: 0.92rem; font-weight: 600;}
        .risk-badge {
            display:inline-block;
            padding:0.35rem 0.65rem;
            border-radius:999px;
            font-weight:700;
            font-size:0.86rem;
            border:1px solid transparent;
        }
        .risk-high {background:#FEE2E2; color:#991B1B; border-color:#FCA5A5;}
        .risk-med {background:#FEF3C7; color:#92400E; border-color:#FCD34D;}
        .risk-low {background:#DCFCE7; color:#166534; border-color:#86EFAC;}
        [data-testid="stSidebar"] {
            background: #FFFFFF !important;
            border-right: 1px solid #E2E8F0;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid #E2E8F0;
            border-radius: 10px;
        }
        .stButton > button,
        [data-testid="stFormSubmitButton"] button,
        [data-testid="stFileUploaderDropzone"] button {
            background: #0B3B8C !important;
            color: #FFFFFF !important;
            border: 1px solid #0B3B8C !important;
            font-weight: 700 !important;
            border-radius: 10px !important;
            padding: 0.45rem 0.85rem !important;
        }
        .stButton > button:hover,
        [data-testid="stFormSubmitButton"] button:hover,
        [data-testid="stFileUploaderDropzone"] button:hover {
            background: #082C67 !important;
            border-color: #082C67 !important;
            color: #FFFFFF !important;
        }
        .stButton > button:focus,
        [data-testid="stFormSubmitButton"] button:focus,
        [data-testid="stFileUploaderDropzone"] button:focus {
            box-shadow: 0 0 0 0.18rem rgba(11, 59, 140, 0.28) !important;
            color: #FFFFFF !important;
        }
        .section-card {
            background: #FFFFFF;
            border: 1px solid #DDE4EE;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.8rem;
        }
        .meta-card {
            background: #0F172A;
            border: 1px solid #1E293B;
            border-radius: 12px;
            padding: 0.85rem;
            color: #E2E8F0 !important;
        }
        .meta-card * { color: #E2E8F0 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _safe_read_json(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@st.cache_resource
def load_keras_model(model_path: Path):
    """Load Keras model lazily to keep the app responsive."""
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorFlow n'est pas installé dans l'environnement Python utilisé par Streamlit. "
            "Active .venv311 puis lance: streamlit run streamlit_app.py"
        ) from exc

    return tf.keras.models.load_model(model_path)


@st.cache_resource
def load_scaler(scaler_path: Path):
    return joblib.load(scaler_path)


@st.cache_data
def load_reference_dataframe(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c != TARGET_COL]
    if not cols:
        raise ValueError("No feature columns found.")
    return cols


def get_feature_medians(df: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
    """Medians used for imputation (same spirit as training script)."""
    return df[feature_cols].median(numeric_only=True)


def preprocess_input(
    df_in: pd.DataFrame,
    feature_cols: List[str],
    medians: pd.Series,
    scaler,
) -> np.ndarray:
    """Align columns, impute with medians, scale using training scaler."""
    X = df_in.copy()
    missing_cols = [c for c in feature_cols if c not in X.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes: {missing_cols[:8]}" + (" ..." if len(missing_cols) > 8 else ""))

    X = X[feature_cols]
    X = X.fillna(medians)
    return scaler.transform(X)


def predict_proba(model, X_scaled: np.ndarray) -> np.ndarray:
    proba = model.predict(X_scaled, verbose=0).ravel()
    return proba


def make_single_input_form(feature_cols: List[str], medians: pd.Series) -> pd.DataFrame:
    """Manual entry form (one company)."""
    st.caption("Saisie manuelle (1 entreprise) — valeurs par défaut = médiane du dataset.")
    with st.form("single_form", clear_on_submit=False):
        cols_per_row = 3
        values = {}
        for i in range(0, len(feature_cols), cols_per_row):
            row = st.columns(cols_per_row)
            for j, col_name in enumerate(feature_cols[i : i + cols_per_row]):
                default = float(medians.get(col_name, 0.0))
                values[col_name] = row[j].number_input(
                    label=col_name,
                    value=default,
                    format="%.6f",
                )
        submitted = st.form_submit_button("Prédire")
    if not submitted:
        return pd.DataFrame()
    return pd.DataFrame([values])


def _read_results(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _display_results_df(df: pd.DataFrame, model_col: str = "Model") -> None:
    f1_col = "F1-score" if "F1-score" in df.columns else ("f1_score" if "f1_score" in df.columns else None)
    roc_col = "ROC-AUC" if "ROC-AUC" in df.columns else ("roc_auc" if "roc_auc" in df.columns else None)
    sort_cols = [c for c in [f1_col, roc_col] if c]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)
    if model_col in df.columns and f1_col:
        top = df.iloc[0]
        roc_val = top[roc_col] if roc_col else float("nan")
        st.success(
            f"Meilleur modèle actuel: **{top[model_col]}** | "
            f"F1={top[f1_col]:.3f} | ROC-AUC={roc_val:.3f}"
        )


def show_results_table() -> None:
    st.subheader("Comparaison des modèles")
    st.markdown("### Comparaison des modèles (Deep Learning)")
    df_dl = _read_results(RESULTS_CSV_DL)
    if df_dl is None:
        st.info("`output/model_results.csv` introuvable. Lance `python deep_learning_bankruptcy.py` pour le générer.")
    else:
        _display_results_df(df_dl, model_col="Model")

    st.markdown("### Comparaison des modèles (Machine Learning)")
    df_ml = _read_results(RESULTS_CSV_ML)
    if df_ml is None:
        st.info(
            "`output/model_results_ml.csv` introuvable. "
            "Astuce: exécute le pipeline ML et sauvegarde ses résultats sous ce nom pour l'affichage ici."
        )
    else:
        model_col = "model" if "model" in df_ml.columns else "Model"
        _display_results_df(df_ml, model_col=model_col)


def show_artifacts_gallery() -> None:
    st.subheader("Figures générées")
    expected_ml = [
        ("Distribution de la cible", OUTPUT_DIR / "class_distribution.png"),
        ("Heatmap de corrélation", OUTPUT_DIR / "correlation_heatmap.png"),
        ("Matrice de confusion (ML)", OUTPUT_DIR / "confusion_matrix.png"),
        ("Courbe ROC (ML)", OUTPUT_DIR / "roc_curve.png"),
        ("Courbe précision-rappel (ML)", OUTPUT_DIR / "precision_recall_curve.png"),
        ("Matrice de confusion (seuil optimal ML)", OUTPUT_DIR / "confusion_matrix_threshold_opt.png"),
    ]
    expected_dl = [
        ("Courbes d'entraînement - Modèle 1", OUTPUT_DIR / "training_curves_model1.png"),
        ("Courbes d'entraînement - Modèle 2", OUTPUT_DIR / "training_curves_model2.png"),
        ("Courbes d'entraînement - Modèle 3", OUTPUT_DIR / "training_curves_model3.png"),
        ("Matrice de confusion (DL)", OUTPUT_DIR / "confusion_matrix.png"),
        ("Courbe ROC (DL)", OUTPUT_DIR / "roc_curve.png"),
        ("Importance des variables (SHAP, DL)", OUTPUT_DIR / "feature_importance.png"),
    ]
    st.markdown("### Figures ML")
    cols = st.columns(2, gap="large")
    for i, (title, path) in enumerate(expected_ml):
        with cols[i % 2]:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown(f"**{title}**")
            st.caption(path.as_posix())
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.warning("Fichier manquant. Exécute `python modeling_bankruptcy.py` pour générer cette figure.")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Figures DL")
    cols = st.columns(2, gap="large")
    for i, (title, path) in enumerate(expected_dl):
        with cols[i % 2]:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown(f"**{title}**")
            st.caption(path.as_posix())
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.warning("Fichier manquant. Exécute `python deep_learning_bankruptcy.py` pour générer cette figure.")
            st.markdown("</div>", unsafe_allow_html=True)


def show_metadata() -> None:
    st.subheader("Métadonnées de run")
    meta = _safe_read_json(RUN_META)
    if meta is None:
        st.info("`output/run_metadata.json` introuvable (optionnel).")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Projet", str(meta.get("projet", "N/A"))[:28] + "...")
    c2.metric("Random state", str(meta.get("random_state", "N/A")))
    c3.metric("Meilleur modèle", str(meta.get("meilleur_modele", "N/A")))
    st.markdown("<div class='meta-card'>", unsafe_allow_html=True)
    st.code(json.dumps(meta, indent=2, ensure_ascii=False), language="json")
    st.markdown("</div>", unsafe_allow_html=True)


def show_risk_distribution(df_pred: pd.DataFrame) -> None:
    """Small dashboard chart: risk band distribution."""
    if "risk_band" not in df_pred.columns:
        return
    dist = (
        df_pred["risk_band"]
        .value_counts()
        .reindex(["Risque faible", "Risque modéré", "Risque élevé"], fill_value=0)
        .reset_index()
    )
    dist.columns = ["Niveau", "Nombre"]
    st.markdown("#### Distribution des niveaux de risque")
    st.bar_chart(dist.set_index("Niveau"))


def risk_label(prob: float, threshold: float) -> Tuple[str, str]:
    """Map probability to business-friendly risk band."""
    if prob >= max(0.75, threshold):
        return "Risque élevé", "risk-high"
    if prob >= max(0.45, threshold * 0.8):
        return "Risque modéré", "risk-med"
    return "Risque faible", "risk-low"


def show_header_kpis(df_ref: pd.DataFrame, threshold: float) -> None:
    """Top-level KPI row for quick executive view."""
    n_rows = len(df_ref)
    base_rate = float(df_ref[TARGET_COL].mean() * 100.0) if TARGET_COL in df_ref.columns else np.nan
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f"<div class='kpi-card'><div class='small-muted'>Entreprises dans le dataset</div>"
        f"<h3 style='margin:0.2rem 0'>{n_rows:,}</h3></div>",
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"<div class='kpi-card'><div class='small-muted'>Taux historique de faillite</div>"
        f"<h3 style='margin:0.2rem 0'>{base_rate:.2f}%</h3></div>",
        unsafe_allow_html=True,
    )
    c3.markdown(
        f"<div class='kpi-card'><div class='small-muted'>Seuil actif de décision</div>"
        f"<h3 style='margin:0.2rem 0'>{threshold:.2f}</h3></div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_custom_css()
    st.title(APP_TITLE)
    st.markdown(
        "<div class='app-card'>"
        "<b>Objectif:</b> estimer la probabilité de faillite d'une société à partir de 95 indicateurs financiers. <br>"
        "Le dashboard est conçu pour une lecture <i>business</i> (risque faible / modéré / élevé) et une intégration opérationnelle (upload CSV)."
        "</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Paramètres")
        threshold = st.slider("Seuil de décision", min_value=0.05, max_value=0.95, value=0.50, step=0.01)
        st.caption("Le seuil transforme la probabilité en classe: proba >= seuil → faillite (1).")

        st.divider()
        st.markdown("**Fichiers attendus**")
        st.write(f"- Modèle: `{MODEL_PATH.as_posix()}`")
        st.write(f"- Scaler: `{SCALER_PATH.as_posix()}`")
        st.write(f"- Dataset: `{DATA_PATH.as_posix()}`")

    # --- Load reference data (for schema + medians) ---
    if not DATA_PATH.exists():
        st.error("`data.csv` introuvable. Place le fichier à la racine du projet.")
        st.stop()
    df_ref = load_reference_dataframe(DATA_PATH)
    feature_cols = get_feature_columns(df_ref)
    medians = get_feature_medians(df_ref, feature_cols)
    show_header_kpis(df_ref, threshold)

    # --- Load model + scaler ---
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        st.error("Modèle/scaler introuvables. Lance d'abord `python deep_learning_bankruptcy.py`.")
        st.stop()
    try:
        model = load_keras_model(MODEL_PATH)
    except Exception as exc:
        st.error(str(exc))
        st.info(
            "Commande recommandée (PowerShell):\n"
            "1) .\\.venv311\\Scripts\\Activate.ps1\n"
            "2) python -m pip install tensorflow streamlit\n"
            "3) python -m streamlit run .\\streamlit_app.py"
        )
        st.stop()
    scaler = load_scaler(SCALER_PATH)

    tab_predict, tab_results, tab_artifacts, tab_meta = st.tabs(
        ["Prédiction", "Comparaison", "Figures", "Métadonnées"]
    )

    with tab_predict:
        st.subheader("Prédire la probabilité de faillite")
        st.write(
            "Deux modes: **upload CSV** (plusieurs entreprises) ou **saisie manuelle** (une entreprise). "
            "Les colonnes doivent correspondre aux 95 indicateurs financiers."
        )
        st.markdown("### 1) Upload CSV")
        up = st.file_uploader("Choisir un fichier CSV", type=["csv"])
        if up is not None:
            df_in = pd.read_csv(up)
            df_in.columns = df_in.columns.str.strip()
            st.caption(f"Fichier chargé: {df_in.shape[0]} lignes, {df_in.shape[1]} colonnes.")
            try:
                X_scaled = preprocess_input(df_in, feature_cols, medians, scaler)
                proba = predict_proba(model, X_scaled)
                pred = (proba >= threshold).astype(int)

                out_df = df_in.copy()
                out_df["proba_bankruptcy"] = proba
                out_df["pred_bankruptcy"] = pred
                out_df["risk_band"] = [
                    risk_label(float(p), threshold)[0] for p in proba
                ]
                st.success("Prédictions générées.")
                st.dataframe(out_df.head(30), use_container_width=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Lignes scorées", f"{len(out_df):,}")
                c2.metric("Faillites prédites", f"{int((out_df['pred_bankruptcy'] == 1).sum()):,}")
                c3.metric("Probabilité moyenne", f"{out_df['proba_bankruptcy'].mean():.3f}")
                show_risk_distribution(out_df)

                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Télécharger les prédictions (CSV)",
                    data=csv_bytes,
                    file_name="predictions_bankruptcy.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Erreur de prétraitement/prédiction: {exc}")

        st.divider()
        st.markdown("### 2) Saisie manuelle")
        df_one = make_single_input_form(feature_cols, medians)
        if not df_one.empty:
            try:
                X_scaled = preprocess_input(df_one, feature_cols, medians, scaler)
                proba = float(predict_proba(model, X_scaled)[0])
                pred = int(proba >= threshold)
                band, css = risk_label(proba, threshold)

                st.markdown("#### Résultat")
                m1, m2 = st.columns(2)
                m1.metric("Probabilité de faillite", f"{proba:.3f}")
                m2.metric("Classe prédite", "Faillite (1)" if pred == 1 else "Saine (0)")
                st.markdown(
                    f"<span class='risk-badge {css}'>Niveau de risque: {band}</span>",
                    unsafe_allow_html=True,
                )

                st.caption("NB: Interpréter la probabilité avec le seuil choisi dans la barre latérale.")
            except Exception as exc:
                st.error(f"Erreur: {exc}")

    with tab_results:
        show_results_table()

    with tab_artifacts:
        show_artifacts_gallery()

    with tab_meta:
        show_metadata()


if __name__ == "__main__":
    main()

