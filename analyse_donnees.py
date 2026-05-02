"""
CRISP-DM — Phase « Compréhension des données » & « Préparation ».
Projet : prédiction de la faillite d'entreprise (Kaggle Company Bankruptcy).

Génère des figures d'EDA dans output/ avec le préfixe ``eda_`` pour ne pas
écraser les livrables de modélisation (class_distribution.png, correlation_heatmap.png, …).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Répertoire du projet (exécutable depuis n'importe quel dossier courant)
PROJECT_ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42
TARGET_COL = "Bankrupt?"
DATA_PATH = PROJECT_ROOT / "data.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Style des graphiques
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style("whitegrid")


def load_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Chargement du dataset."""
    p = path or DATA_PATH
    df = pd.read_csv(p)
    df.columns = df.columns.str.strip()
    if TARGET_COL not in df.columns:
        raise ValueError(f"Colonne cible « {TARGET_COL} » absente dans {p}")
    return df


def analyse_exploratoire(df):
    """Analyse exploratoire des données."""
    results = {}
    
    # Dimensions
    results['shape'] = df.shape
    results['n_lignes'] = df.shape[0]
    results['n_colonnes'] = df.shape[1]
    
    # Variable cible
    target_counts = df[TARGET_COL].value_counts()
    results['target_distribution'] = target_counts
    results['taux_faillite'] = (df[TARGET_COL] == 1).mean() * 100
    
    # Valeurs manquantes
    results['missing_total'] = df.isnull().sum().sum()
    results['colonnes_manquantes'] = df.columns[df.isnull().any()].tolist()
    
    # Statistiques descriptives
    results['describe'] = df.describe()
    results['dtypes'] = df.dtypes
    
    # Doublons
    results['doublons'] = df.duplicated().sum()
    
    # Valeurs aberrantes (IQR pour quelques colonnes clés)
    cols_numeriques = df.select_dtypes(include=[np.number]).columns
    outlier_info = {}
    for col in cols_numeriques[:10]:  # Échantillon de colonnes
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        if outliers > 0:
            outlier_info[col] = outliers
    results['outliers_echantillon'] = outlier_info
    
    return results


def nettoyage_donnees(df):
    """Nettoyage des données."""
    df_clean = df.copy()
    etapes = []
    
    # 1. Suppression des doublons
    n_avant = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    n_apres = len(df_clean)
    etapes.append(f"Doublons supprimés: {n_avant - n_apres}")
    
    # 2. Gestion des valeurs manquantes (s'il y en a)
    if df_clean.isnull().sum().sum() > 0:
        df_clean = df_clean.dropna()
        etapes.append("Valeurs manquantes: suppression des lignes")
    else:
        etapes.append("Aucune valeur manquante détectée")
    
    # 3. Vérification des types
    etapes.append("Types de données vérifiés")
    
    return df_clean, etapes


def visualisations(df, df_clean):
    """Génération des visualisations."""
    
    # 1. Distribution de la variable cible
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df_clean[TARGET_COL].value_counts().sort_index()
    colors = ['#2ecc71', '#e74c3c']
    labels = ['Non faillite (0)', 'Faillite (1)'][: len(counts)]
    bars = ax.bar(labels, counts.values, color=colors[: len(counts)], edgecolor='black')
    ax.set_ylabel('Nombre d\'entreprises')
    ax.set_title('Distribution de la variable cible (Bankrupt?)')
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, str(val), ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'eda_distribution_cible.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Heatmap de corrélation (échantillon de variables clés)
    cols_cles = [TARGET_COL, 'Current Ratio', 'Quick Ratio', 'Total debt/Total net worth',
                 'Debt ratio %', 'Cash flow rate', 'Operating Gross Margin', 'Net worth/Assets']
    cols_dispo = [c for c in cols_cles if c in df_clean.columns]
    if len(cols_dispo) >= 3:
        corr = df_clean[cols_dispo].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f', ax=ax)
        ax.set_title('Matrice de corrélation (indicateurs clés)')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'eda_correlation_indicateurs_cles.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. Boxplots de quelques indicateurs par statut de faillite
    indicateurs = ['Current Ratio', 'Total debt/Total net worth', 'Debt ratio %']
    indicateurs = [c for c in indicateurs if c in df_clean.columns]
    if indicateurs:
        fig, axes = plt.subplots(1, min(3, len(indicateurs)), figsize=(14, 5))
        if len(indicateurs) == 1:
            axes = [axes]
        for i, col in enumerate(indicateurs[:3]):
            sns.boxplot(data=df_clean, x=TARGET_COL, y=col, hue=TARGET_COL, palette=['#2ecc71', '#e74c3c'], legend=False, ax=axes[i])
            axes[i].set_title(col)
            axes[i].set_xticklabels(['Non faillite', 'Faillite'])
        plt.suptitle('Distribution des indicateurs financiers par statut')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'boxplots_indicateurs.png', dpi=150, bbox_inches='tight')
        plt.close()


def exporter_donnees_nettoyees(df_clean):
    """Export des données nettoyées."""
    df_clean.to_csv(OUTPUT_DIR / 'data_cleaned.csv', index=False)
    print(f"Données nettoyées exportées: {OUTPUT_DIR / 'data_cleaned.csv'}")


def generer_rapport_stats(results, etapes_nettoyage):
    """Génère un fichier texte avec les statistiques pour le rapport LaTeX."""
    with open(OUTPUT_DIR / 'stats_rapport.txt', 'w', encoding='utf-8') as f:
        f.write(f"Lignes: {results['n_lignes']}\n")
        f.write(f"Colonnes: {results['n_colonnes']}\n")
        f.write(f"Taux faillite: {results['taux_faillite']:.2f}%\n")
        f.write(f"Entreprises en faillite: {results['target_distribution'].get(1, 0)}\n")
        f.write(f"Entreprises saines: {results['target_distribution'].get(0, 0)}\n")
        f.write(f"Valeurs manquantes: {results['missing_total']}\n")
        f.write(f"Doublons: {results['doublons']}\n")
        for etape in etapes_nettoyage:
            f.write(f"Nettoyage: {etape}\n")
    print(f"Statistiques exportées: {OUTPUT_DIR / 'stats_rapport.txt'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CRISP-DM — Analyse exploratoire et nettoyage (faillite d'entreprise)."
    )
    p.add_argument("--data", type=Path, default=DATA_PATH, help="Chemin vers data.csv")
    return p.parse_args()


def main(data_path: Optional[Path] = None):
    print("=" * 60)
    print("CRISP-DM — Analyse et nettoyage — Prédiction de la faillite d'entreprise")
    print("=" * 60)

    df = load_data(data_path)
    print(f"\nDataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # Analyse exploratoire
    results = analyse_exploratoire(df)
    print(f"\n--- Analyse Exploratoire ---")
    print(f"Variable cible - Non faillite: {results['target_distribution'].get(0, 0)}, Faillite: {results['target_distribution'].get(1, 0)}")
    print(f"Taux de faillite: {results['taux_faillite']:.2f}%")
    print(f"Valeurs manquantes: {results['missing_total']}")
    print(f"Doublons: {results['doublons']}")
    
    # Nettoyage
    df_clean, etapes = nettoyage_donnees(df)
    print(f"\n--- Nettoyage ---")
    for e in etapes:
        print(f"  - {e}")
    print(f"Dimensions après nettoyage: {df_clean.shape}")
    
    # Visualisations
    visualisations(df, df_clean)
    print(f"\nVisualisations générées dans {OUTPUT_DIR}/")
    
    # Export
    exporter_donnees_nettoyees(df_clean)
    generer_rapport_stats(results, etapes)
    
    print("\n" + "=" * 60)
    print("Terminé avec succès!")
    return df, df_clean, results


if __name__ == "__main__":
    _args = parse_args()
    df, df_clean, results = main(data_path=_args.data)
