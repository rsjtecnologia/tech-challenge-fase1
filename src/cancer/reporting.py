"""
Geracao de graficos e tabelas para o modulo de Cancer de Mama.

Seguindo o exemplo do notebook de referencia (FIAP Tech Challenge Fase 1),
estas funcoes produzem:
  - Tabelas: preview dos dados, descricao, distribuicao das classes,
    comparativo de metricas, top features.
  - Graficos (base64 PNG): distribuicao das classes, correlacao,
    boxplots, comparativo de modelos, feature importance/SHAP.

Tudo em formato JSON-serializavel para retornar diretamente na API.
"""
import base64
import io
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Paleta consistente com o notebook
PALETTE = ["#2ecc71", "#e74c3c"]  # B (verde), M (vermelho)

# Rótulos padrão (câncer de mama). Os gráficos preservam exatamente as strings
# originais "B (0)" / "M (1)" quando este padrão é usado.
_DEFAULT_CLASS_LABELS = ("B (benigno)", "M (maligno)")


def _fig_to_base64(fig) -> str:
    """Converte uma figura matplotlib para data-uri base64 (PNG)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------

def preview_table(df: pd.DataFrame, n: int = 5) -> list:
    """Primeiras n linhas do dataset (como X.head() no notebook)."""
    return df.head(n).to_dict(orient="records")


def describe_table(df: pd.DataFrame) -> list:
    """Estatisticas descritivas (como df.describe())."""
    feats = [c for c in df.columns if c not in ("diagnosis", "target")]
    stats = df[feats].describe().T.reset_index()
    stats = stats.rename(columns={"index": "feature"})
    return stats.round(4).to_dict(orient="records")


def class_distribution_table(df: pd.DataFrame, class_labels: tuple = _DEFAULT_CLASS_LABELS) -> list:
    """Contagem e proporcao das classes (como no notebook).

    Args:
        class_labels: nomes das classes (indice 0 e 1). Padrao: câncer de mama.
    """
    counts = df["target"].value_counts().sort_index()
    total = len(df)
    rows = []
    for label, name in enumerate(class_labels):
        rows.append({
            "classe": name,
            "label": label,
            "contagem": int(counts.get(label, 0)),
            "proporcao": round(int(counts.get(label, 0)) / total, 4),
        })
    return rows


def missing_table(df: pd.DataFrame) -> list:
    """Valores nulos por coluna."""
    missing = df.isnull().sum()
    rows = [{
        "coluna": col,
        "nulos": int(missing[col]),
        "percentual": round(float(missing[col]) / len(df) * 100, 2),
    } for col in df.columns if missing[col] > 0]
    return rows


def metrics_table(resultados: dict) -> list:
    """Tabela comparativa de metricas dos modelos (como no notebook)."""
    rows = []
    for nome, m in resultados.items():
        rows.append({
            "modelo": nome.replace("_", " ").title(),
            "acuracia": round(m["accuracy"] * 100, 2),
            "recall": round(m["recall"] * 100, 2),
            "precisao": round(m["precision"] * 100, 2),
            "f1": round(m["f1"] * 100, 2),
            "auc": round(m["auc"] * 100, 2),
        })
    return rows


def top_features_table(model, feature_names: list, n: int = 10, values: list = None) -> list:
    """
    Top n features mais relevantes de um modelo.

    Usa os valores SHAP da predição quando fornecidos (funciona para
    qualquer modelo, inclusive o linear), senão cai para
    `feature_importances_` (modelos tree-based).
    """
    if values is not None:
        vals = np.array(values)
        indices = np.argsort(np.abs(vals))[::-1][:n]
        return [
            {"posicao": i + 1, "feature": feature_names[idx],
             "importancia": round(float(vals[idx]), 4)}
            for i, idx in enumerate(indices)
        ]
    if not hasattr(model, "feature_importances_"):
        return []
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:n]
    return [
        {"posicao": i + 1, "feature": feature_names[idx],
         "importancia": round(float(importances[idx]), 4)}
        for i, idx in enumerate(indices)
    ]


def correlation_table(df: pd.DataFrame, n: int = 10) -> list:
    """Top n features mais correlacionadas com o target (M=1)."""
    feats = [c for c in df.columns if c not in ("diagnosis", "target")]
    corr = df[feats].corrwith(df["target"]).abs().sort_values(ascending=False)
    return [
        {"feature": idx, "correlacao": round(float(corr[idx]), 4)}
        for idx in corr.index[:n]
    ]


# ---------------------------------------------------------------------------
# Graficos (base64)
# ---------------------------------------------------------------------------

def _graph_class_names(class_labels: tuple) -> list:
    """Nomes curtos das classes para os eixos dos gráficos."""
    if class_labels == _DEFAULT_CLASS_LABELS:
        return ["B (0)", "M (1)"]
    return [f"{name} ({i})" for i, name in enumerate(class_labels)]


def class_distribution_graph(df: pd.DataFrame, class_labels: tuple = _DEFAULT_CLASS_LABELS) -> str:
    """Countplot + pie das classes (como no notebook).

    Args:
        class_labels: nomes das classes (indice 0 e 1). Padrao: câncer de mama.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    display_names = _graph_class_names(class_labels)

    counts = df["target"].value_counts().sort_index()
    sns.countplot(x="target", hue="target", data=df,
                  palette=PALETTE, legend=False, ax=axes[0])
    title = "Contagem por Classe (M=1, B=0)" if class_labels == _DEFAULT_CLASS_LABELS \
        else "Contagem por Classe"
    axes[0].set_title(title, fontsize=13, fontweight="bold")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(display_names)
    axes[0].set_xlabel("Classe")
    axes[0].set_ylabel("Contagem")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

    df["target"].value_counts().plot(
        kind="pie", autopct="%1.1f%%",
        colors=PALETTE, labels=display_names, ax=axes[1],
    )
    axes[1].set_title("Proporção das Classes", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("")
    plt.tight_layout()
    return _fig_to_base64(fig)


def correlation_graph(df: pd.DataFrame) -> str:
    """Heatmap de correlacao das features."""
    feats = [c for c in df.columns if c not in ("diagnosis", "target")]
    corr = df[feats].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                annot=False, square=True, linewidths=0.5, ax=ax)
    ax.set_title("Matriz de Correlação entre Features", fontsize=15, fontweight="bold")
    plt.tight_layout()
    return _fig_to_base64(fig)


def boxplot_graph(df: pd.DataFrame, n: int = 6) -> str:
    """Boxplots das top n features por classe."""
    feats = [c for c in df.columns if c not in ("diagnosis", "target")]
    corr = df[feats].corrwith(df["target"]).abs().sort_values(ascending=False)
    top = corr.index[:n]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for idx, feature in enumerate(top):
        sns.boxplot(x="target", y=df[feature], hue="target", data=df,
                    palette=PALETTE, legend=False, ax=axes[idx])
        axes[idx].set_title(feature, fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("B (0) / M (1)")
    for ax in axes[len(top):]:
        ax.axis("off")
    fig.suptitle("Distribuição das Top Features por Classe",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    return _fig_to_base64(fig)


def metrics_graph(resultados: dict) -> str:
    """Bar chart comparativo das metricas dos modelos."""
    df = pd.DataFrame(resultados).T * 100
    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot(kind="bar", ax=ax)
    ax.set_title("Comparativo de Métricas - Câncer de Mama",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("Modelo")
    ax.set_ylim(50, 100)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xticklabels([m.replace("_", " ").title() for m in df.index],
                       rotation=15, ha="right")
    plt.tight_layout()
    return _fig_to_base64(fig)


def shap_bar_graph(feature_names: list, values: list, title: str = "SHAP") -> str:
    """
    Bar chart horizontal de contribuicoes (ex.: SHAP de uma predicao).

    Args:
        feature_names: nomes das features
        values: valores de contribuicao (positivos empurram p/ M)
        title: titulo do grafico
    """
    vals = np.array(values)
    idx = np.argsort(np.abs(vals))[::-1][:10]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals[idx]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feature_names[i] for i in idx][::-1],
            [vals[i] for i in idx][::-1], color=colors[::-1])
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Contribuição (SHAP)")
    plt.tight_layout()
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Agregadores
# ---------------------------------------------------------------------------

def build_eda_payload(df: pd.DataFrame, class_labels: tuple = _DEFAULT_CLASS_LABELS) -> dict:
    """Tabelas + graficos de EDA (como o notebook de referencia)."""
    return {
        "tables": {
            "preview": preview_table(df),
            "describe": describe_table(df),
            "classes": class_distribution_table(df, class_labels=class_labels),
            "missing": missing_table(df),
            "correlacao_target": correlation_table(df),
        },
        "graphs": {
            "classes": class_distribution_graph(df, class_labels=class_labels),
            "correlacao": correlation_graph(df),
            "boxplots": boxplot_graph(df),
        },
    }


def build_metrics_payload(resultados: dict) -> dict:
    """Tabela + grafico comparativo dos modelos."""
    return {
        "tables": {"metricas": metrics_table(resultados)},
        "graphs": {"comparativo": metrics_graph(resultados)},
    }
