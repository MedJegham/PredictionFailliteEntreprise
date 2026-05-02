"""
CRISP-DM — Phase « Modélisation » & « Évaluation ».
Projet : prédiction de la faillite d'entreprise (Kaggle Company Bankruptcy).

Pipeline : StandardScaler → SMOTE → classifieur. GridSearchCV (F1) sur Random Forest.
Sorties : output/ (CSV, figures, best_model.pkl, metadata, seuil PR).

Usage :
    python modeling_bankruptcy.py
    python modeling_bankruptcy.py --data chemin/data.csv --output chemin/out --no-xgb
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42
TARGET_COL = "Bankrupt?"
DATA_PATH = PROJECT_ROOT / "data.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
TEST_SIZE = 0.2
CV_FOLDS = 5

COLOR_PRIMARY = "#2E5AAC"
COLOR_SECONDARY = "#C44E52"
COLOR_NEUTRAL = "#4C4C4C"


def apply_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["figure.dpi"] = 120


apply_plot_style()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CRISP-DM — Modélisation : prédiction de la faillite d'entreprise."
    )
    p.add_argument("--data", type=Path, default=DATA_PATH)
    p.add_argument("--output", type=Path, default=OUTPUT_DIR)
    p.add_argument("--no-xgb", action="store_true")
    return p.parse_args()


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_and_prepare_features(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if TARGET_COL not in df.columns:
        raise ValueError(f"Colonne cible « {TARGET_COL} » absente du fichier.")
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])
    return X, y


def plot_class_distribution(y: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    counts = y.value_counts().sort_index()
    labels = ["Entreprise saine (0)", "Faillite (1)"]
    colors = [COLOR_PRIMARY, COLOR_SECONDARY]
    bars = ax.bar(
        [labels[i] for i in range(len(counts))],
        counts.values,
        color=colors[: len(counts)],
        edgecolor="white",
        linewidth=1.2,
    )
    ax.set_ylabel("Nombre d'observations", fontweight="medium")
    ax.set_xlabel("Classe (variable cible)", fontweight="medium")
    ax.set_title(
        "Distribution de la cible — Prédiction de faillite d'entreprise",
        fontweight="bold",
        pad=12,
    )
    pct = 100.0 * counts.values / counts.values.sum()
    for b, v, p in zip(bars, counts.values, pct):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + max(counts.values) * 0.02,
            f"{v:,}\n({p:.2f} %)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, max(counts.values) * 1.18)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_correlation_heatmap(X: pd.DataFrame, y: pd.Series, out_path: Path, top_k: int = 30) -> None:
    numeric = X.select_dtypes(include=[np.number])
    numeric = numeric.loc[:, numeric.nunique() > 1]
    if numeric.shape[1] == 0:
        return
    corr_target = numeric.corrwith(y).abs().sort_values(ascending=False)
    cols = corr_target.head(top_k).index.tolist()
    sub = numeric[cols]
    cm = sub.corr()
    fig, ax = plt.subplots(figsize=(14.5, 12))
    sns.heatmap(
        cm,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"shrink": 0.55, "label": "Corrélation de Pearson"},
    )
    ax.set_title(
        f"Corrélations entre les {top_k} prédicteurs les plus associés à la faillite\n"
        f"(|corrélation| avec « {TARGET_COL} »)",
        fontweight="bold",
        pad=14,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_base_pipeline(classifier, smote_k_neighbors: int = 3) -> ImbPipeline:
    return ImbPipeline(
        [
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=smote_k_neighbors)),
            ("clf", classifier),
        ]
    )


def get_param_grids() -> Dict[str, Dict[str, List[Any]]]:
    return {
        "random_forest": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [8, 16, None],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__class_weight": [None, "balanced"],
        },
    }


def build_model_configs(smote_k_neighbors: int = 3) -> List[Dict[str, Any]]:
    return [
        {
            "name": "Logistic Regression",
            "key": "logistic_regression",
            "pipe": make_base_pipeline(
                LogisticRegression(
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
                smote_k_neighbors=smote_k_neighbors,
            ),
            "grid": None,
        },
        {
            "name": "Decision Tree",
            "key": "decision_tree",
            "pipe": make_base_pipeline(
                DecisionTreeClassifier(
                    random_state=RANDOM_STATE,
                    max_depth=12,
                    min_samples_leaf=4,
                    class_weight="balanced",
                ),
                smote_k_neighbors=smote_k_neighbors,
            ),
            "grid": None,
        },
    ]


def train_with_optional_grid(
    pipe: ImbPipeline,
    param_grid: Optional[Dict[str, List[Any]]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_key: str,
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    print(
        f"  -> Entrainement : {model_key}"
        + (" (GridSearchCV, metrique = F1)..." if param_grid else "..."),
        flush=True,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    if param_grid:
        gs = GridSearchCV(
            pipe,
            param_grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        return gs.best_estimator_, gs.best_params_
    pipe.fit(X_train, y_train)
    return pipe, None


def cross_val_scores_pipeline(
    fitted_estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        clone(fitted_estimator),
        X_train,
        y_train,
        cv=cv,
        scoring={"f1": "f1", "roc_auc": "roc_auc", "accuracy": "accuracy"},
        n_jobs=-1,
        return_train_score=False,
    )
    return {
        "cv_f1_mean": float(np.mean(scores["test_f1"])),
        "cv_f1_std": float(np.std(scores["test_f1"])),
        "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(scores["test_roc_auc"])),
        "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
        "cv_accuracy_std": float(np.std(scores["test_accuracy"])),
    }


def evaluate_binary(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }


def extract_feature_importance(model: Any, feature_names: List[str]) -> np.ndarray:
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        return np.asarray(clf.feature_importances_)
    if hasattr(clf, "coef_"):
        return np.asarray(np.abs(clf.coef_).ravel())
    return np.zeros(len(feature_names))


def plot_confusion_matrix(
    y_true, y_pred, out_path: Path, title: str, subtitle: str = ""
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    cm_pct = 100.0 * cm / row_sum
    labels = np.array(
        [[f"{n}\n({p:.1f} %)" for n, p in zip(row, pct_row)] for row, pct_row in zip(cm, cm_pct)]
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        ax=ax,
        cbar_kws={"label": "Effectif"},
        linewidths=1,
        linecolor="white",
        xticklabels=["Prédit : saine (0)", "Prédit : faillite (1)"],
        yticklabels=["Réel : saine (0)", "Réel : faillite (1)"],
    )
    ax.set_title(title, fontweight="bold", pad=10)
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=10, style="italic")
    ax.set_ylabel("Vérité terrain (étiquette réelle)", fontweight="medium")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_roc_curve(y_true, y_score, out_path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.fill_between(fpr, tpr, alpha=0.15, color=COLOR_PRIMARY)
    ax.plot(fpr, tpr, color=COLOR_PRIMARY, lw=2.2, label=f"Modèle (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], ls="--", color=COLOR_NEUTRAL, lw=1.2, label="Classifieur aléatoire")
    ax.set_xlabel("Taux de faux positifs (1 - spécificité)", fontweight="medium")
    ax.set_ylabel("Taux de vrais positifs (rappel)", fontweight="medium")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.35, linestyle=":")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    out_path: Path,
    top_n: int = 25,
    model_name: str = "meilleur modèle",
) -> None:
    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(11, 8))
    y_pos = np.arange(len(order))
    ax.barh(y_pos, importances[order], color=COLOR_PRIMARY, edgecolor="white", height=0.75)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance relative", fontweight="medium")
    ax.set_title(
        f"Variables les plus associées à la prédiction — {model_name}\n(Top {top_n})",
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, axis="x", alpha=0.35, linestyle=":")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def optimal_threshold_f1(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, Dict[str, float]]:
    thresholds = np.concatenate(([0.0], np.linspace(0.01, 0.99, 199), [1.0]))
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f = f1_score(y_true, y_pred, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = float(t)
    y_pred_best = (y_proba >= best_t).astype(int)
    metrics = {
        "threshold": best_t,
        "f1": float(f1_score(y_true, y_pred_best, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred_best, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_best, zero_division=0)),
    }
    return best_t, metrics


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    out_path: Path,
    model_name: str,
    threshold_opt: Optional[float] = None,
    prec_at_t: Optional[float] = None,
    rec_at_t: Optional[float] = None,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    baseline = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.plot(recall, precision, color=COLOR_PRIMARY, lw=2.2, label=f"Modèle (AP = {ap:.4f})")
    ax.axhline(
        baseline,
        color=COLOR_SECONDARY,
        ls="--",
        lw=1.5,
        label=f"Référence (proportion faillites = {baseline:.4f})",
    )
    ax.fill_between(recall, precision, alpha=0.12, color=COLOR_PRIMARY)
    if threshold_opt is not None and prec_at_t is not None and rec_at_t is not None:
        ax.scatter(
            [rec_at_t],
            [prec_at_t],
            s=140,
            color=COLOR_SECONDARY,
            edgecolors="white",
            zorder=5,
            label=f"Seuil optimal (train OOF) = {threshold_opt:.3f}",
        )
        ax.annotate(
            f"P={prec_at_t:.3f}, R={rec_at_t:.3f}",
            xy=(rec_at_t, prec_at_t),
            xytext=(rec_at_t - 0.12, prec_at_t + 0.08),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color=COLOR_NEUTRAL, lw=0.8),
        )
    ax.set_xlabel("Rappel (sensibilité)", fontweight="medium")
    ax.set_ylabel("Précision", fontweight="medium")
    ax.set_title(
        f"Courbe précision–rappel — Classe faillite (1)\n{model_name}",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.35, linestyle=":")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_business_interpretation_md(
    out_path: Path,
    best_name: str,
    best_metrics: Dict[str, float],
    results_sorted: pd.DataFrame,
    threshold_info: Optional[Dict[str, Any]] = None,
) -> None:
    thr_block = ""
    if threshold_info:
        thr_block = f"""

## Seuil (probabilité)

Seuil **{threshold_info['threshold']:.4f}** (max F1 sur prédictions OOF du train). OOF F1 = **{threshold_info['f1_oof']:.4f}**. Sur le test avec ce seuil : F1 = **{threshold_info['f1_test']:.4f}**, P = **{threshold_info['precision_test']:.4f}**, R = **{threshold_info['recall_test']:.4f}**.
"""
    text = f"""# Interprétation métier — prédiction de faillite

## Déséquilibre des classes

Les faillites sont rares ; sans SMOTE le modèle tend à prédire la majorité. Le pipeline applique SMOTE **uniquement sur le train** (chaque pli de CV).

## F1-score et validation croisée

Le F1 combine précision et rappel. Les colonnes `cv_*` dans `model_results.csv` donnent la moyenne et l'écart-type sur {CV_FOLDS} plis (train).

## Faux négatifs

Une faillite non détectée peut entraîner pertes de créance et sous-estimation du risque.

## Modèle retenu

**{best_name}** — Test (seuil 0,5) : F1 = **{best_metrics['f1_score']:.4f}**, ROC-AUC = **{best_metrics['roc_auc']:.4f}**.
GridSearchCV (F1) sur la forêt aléatoire. Tri final : F1 test puis ROC-AUC test.
{thr_block}
## Tableau comparatif

```
{results_sorted.to_string(index=False)}
```
"""
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    out = ensure_output_dir(args.output.resolve())

    print("CRISP-DM - Modelisation : prediction de la faillite d'entreprise")
    print(f"  Donnees : {data_path}")
    print(f"  Sortie  : {out}\n")

    # 1) Chargement
    X, y = load_and_prepare_features(data_path)
    feature_names = X.columns.tolist()

    # 2) EDA figures
    plot_class_distribution(y, out / "class_distribution.png")
    plot_correlation_heatmap(X, y, out / "correlation_heatmap.png")

    # 3) Split stratifie
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    n_minority = int(y_train.sum())
    smote_kn = min(3, max(1, n_minority - 1))

    grids = get_param_grids()
    configs = build_model_configs(smote_k_neighbors=smote_kn)
    configs.append(
        {
            "name": "Random Forest",
            "key": "random_forest",
            "pipe": make_base_pipeline(
                RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                smote_k_neighbors=smote_kn,
            ),
            "grid": grids["random_forest"],
        }
    )
    configs.append(
        {
            "name": "Gradient Boosting",
            "key": "gradient_boosting",
            "pipe": make_base_pipeline(
                GradientBoostingClassifier(
                    random_state=RANDOM_STATE,
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.1,
                    min_samples_leaf=3,
                ),
                smote_k_neighbors=smote_kn,
            ),
            "grid": None,
        }
    )

    if args.no_xgb:
        print("  (XGBoost desactive via --no-xgb.)")
    else:
        try:
            from xgboost import XGBClassifier

            configs.append(
                {
                    "name": "XGBoost",
                    "key": "xgboost",
                    "pipe": make_base_pipeline(
                        XGBClassifier(
                            random_state=RANDOM_STATE,
                            eval_metric="logloss",
                            n_estimators=200,
                            max_depth=4,
                            learning_rate=0.08,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            n_jobs=-1,
                        ),
                        smote_k_neighbors=smote_kn,
                    ),
                    "grid": None,
                }
            )
        except ImportError:
            print("  (XGBoost non installe - ignore.)")

    rows: List[Dict[str, Any]] = []
    fitted: Dict[str, Any] = {}
    print("\nValidation croisee sur le train (clone du pipeline, 5 plis)...")

    for cfg in configs:
        model, best_params = train_with_optional_grid(
            cfg["pipe"], cfg["grid"], X_train, y_train, cfg["key"]
        )
        cv_stats = cross_val_scores_pipeline(model, X_train, y_train)
        test_metrics = evaluate_binary(model, X_test, y_test)
        rows.append(
            {
                "model": cfg["name"],
                "key": cfg["key"],
                **test_metrics,
                **cv_stats,
                "best_params": json.dumps(best_params, ensure_ascii=False) if best_params else "",
            }
        )
        fitted[cfg["key"]] = model

    results = pd.DataFrame(rows)
    results.drop(columns=["key"], errors="ignore").to_csv(out / "model_results.csv", index=False)

    results_sorted = results.sort_values(
        ["f1_score", "roc_auc"], ascending=[False, False]
    ).reset_index(drop=True)
    display_cols = [
        c
        for c in [
            "model",
            "cv_f1_mean",
            "cv_f1_std",
            "cv_roc_auc_mean",
            "cv_roc_auc_std",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
        ]
        if c in results_sorted.columns
    ]
    print("\n=== Comparaison des modeles (F1 test, puis ROC-AUC test) ===\n")
    print(results_sorted[display_cols].to_string(index=False))

    best_key = results_sorted.iloc[0]["key"]
    best_name = results_sorted.iloc[0]["model"]
    best_model = fitted[best_key]
    best_metrics = {
        "f1_score": float(results_sorted.iloc[0]["f1_score"]),
        "roc_auc": float(results_sorted.iloc[0]["roc_auc"]),
        "recall": float(results_sorted.iloc[0]["recall"]),
        "precision": float(results_sorted.iloc[0]["precision"]),
    }

    joblib.dump(best_model, out / "best_model.pkl")
    y_proba_best = best_model.predict_proba(X_test)[:, 1]
    y_pred_best = best_model.predict(X_test)

    plot_confusion_matrix(
        y_test,
        y_pred_best,
        out / "confusion_matrix.png",
        f"Matrice de confusion (seuil = 0,5) — {best_name}",
        subtitle="Pourcentages par ligne (vraie classe)",
    )
    plot_roc_curve(
        y_test,
        y_proba_best,
        out / "roc_curve.png",
        f"Courbe ROC — Jeu de test\n{best_name}",
    )
    imp = extract_feature_importance(best_model, feature_names)
    plot_feature_importance(
        imp, feature_names, out / "feature_importance.png", model_name=best_name
    )

    cv_oof = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = cross_val_predict(
        clone(best_model),
        X_train,
        y_train,
        cv=cv_oof,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    thr_opt, metrics_oof = optimal_threshold_f1(y_train.values, oof_proba)
    y_pred_thr = (y_proba_best >= thr_opt).astype(int)
    threshold_info = {
        "threshold": thr_opt,
        "f1_oof": metrics_oof["f1"],
        "precision_oof": metrics_oof["precision"],
        "recall_oof": metrics_oof["recall"],
        "f1_test": float(f1_score(y_test, y_pred_thr, zero_division=0)),
        "precision_test": float(precision_score(y_test, y_pred_thr, zero_division=0)),
        "recall_test": float(recall_score(y_test, y_pred_thr, zero_division=0)),
    }
    prec_t = threshold_info["precision_test"]
    rec_t = threshold_info["recall_test"]

    plot_precision_recall_curve(
        y_test.values,
        y_proba_best,
        out / "precision_recall_curve.png",
        best_name,
        threshold_opt=thr_opt,
        prec_at_t=prec_t,
        rec_at_t=rec_t,
    )
    plot_confusion_matrix(
        y_test,
        y_pred_thr,
        out / "confusion_matrix_threshold_opt.png",
        f"Matrice de confusion (seuil optimal = {thr_opt:.3f}) — {best_name}",
        subtitle="Seuil calibré sur OOF train (max F1)",
    )

    (out / "threshold_metrics.json").write_text(
        json.dumps(threshold_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    meta = {
        "projet": "CRISP-DM — Prédiction de la faillite d'entreprise",
        "fichier_donnees": str(data_path),
        "dossier_sortie": str(out),
        "meilleur_modele": best_name,
        "random_state": RANDOM_STATE,
        "cv_plis": CV_FOLDS,
        "metriques_test_seuil_05": best_metrics,
        "seuil_optimal": threshold_info,
    }
    (out / "run_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md_cols = [c for c in results_sorted.columns if c not in ("key", "best_params")]
    write_business_interpretation_md(
        out / "business_interpretation.md",
        best_name,
        best_metrics,
        results_sorted[md_cols],
        threshold_info=threshold_info,
    )

    print(f"\nFichiers generes dans : {out.resolve()}")
    print(f"Meilleur modele : {best_name} (cle : {best_key})")
    print(f"Seuil optimal (OOF train) : {thr_opt:.4f}")


if __name__ == "__main__":
    main()
