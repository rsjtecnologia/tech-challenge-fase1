"""
Rotas de predição e relatórios de Câncer de Mama (módulo src/cancer).

Seguindo o padrão das rotas de diabetes, expõe:
  - POST /predict/cancer/       -> predição com gráfico + tabela
  - GET  /predict/cancer/models -> modelos disponíveis
  - GET  /predict/cancer/info   -> informações do dataset
  - GET  /predict/cancer/eda    -> tabelas + gráficos de EDA (como o notebook)
  - GET  /predict/cancer/metrics-> tabela + gráfico comparativo dos modelos
"""
from fastapi import APIRouter, Depends, HTTPException

from src.api.schemas import (
    CancerPredictionRequest,
    CancerPredictionResponse,
    CancerEDAResponse,
    CancerMetricsResponse,
    CancerModelsListResponse,
    ErrorResponse,
)
from src.api.cancer_service import CancerService

router = APIRouter(
    prefix="/predict/cancer",
    tags=["cancer"],
)


@router.post(
    "/",
    response_model=CancerPredictionResponse,
    summary="Classificar tumor de mama (com gráfico e tabela)",
    description="""
    Prediz se um tumor de mama é benigno (B) ou maligno (M) com base em 30 features.

    **Encodings:** M → 1 (maligno), B → 0 (benigno).

    **Resposta inclui:**
    - **diagnosis**: M ou B
    - **probabilities**: probabilidades de cada classe
    - **graphs.shap**: gráfico de importância das features (base64 PNG)
    - **tables.top_features**: tabela com as 10 features mais relevantes

    **Modelos disponíveis:** logistic_regression, decision_tree, random_forest (padrão), gradient_boosting, svm, knn
    """,
    responses={
        200: {"description": "Predição realizada com sucesso", "model": CancerPredictionResponse},
        400: {"description": "Requisição inválida", "model": ErrorResponse},
        503: {"description": "Modelos não carregados", "model": ErrorResponse},
        500: {"description": "Erro interno", "model": ErrorResponse},
    },
)
def predict_cancer(
    req: CancerPredictionRequest,
    service: CancerService = Depends(),
):
    """Prediz o diagnóstico de câncer de mama e retorna gráfico + tabela."""
    if not service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Modelos de câncer não carregados. Execute o treinamento primeiro."
        )
    try:
        return service.predict(req.features, req.model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.get(
    "/models",
    response_model=CancerModelsListResponse,
    summary="Listar modelos de câncer",
    description="Retorna a lista de modelos de câncer de mama disponíveis para uso.",
)
def list_models(service: CancerService = Depends()):
    """Lista modelos de câncer carregados."""
    return {"available_models": service.available_models}


@router.get(
    "/samples",
    summary="Amostras da base de dados",
    description="Retorna amostras reais (benigna e maligna) do Breast Cancer Wisconsin para o frontend.",
)
def sample_rows(service: CancerService = Depends()):
    """Retorna amostras reais da base para preenchimento do frontend."""
    return service.get_samples()


@router.get(
    "/info",
    summary="Informações do dataset",
    description="Retorna informações detalhadas sobre o dataset Breast Cancer Wisconsin.",
)
def model_info():
    """Informações sobre o dataset de câncer de mama."""
    from src.cancer.dataset import get_cancer_info
    return get_cancer_info()


@router.get(
    "/eda",
    response_model=CancerEDAResponse,
    summary="Relatório EDA (gráficos e tabelas)",
    description="""
    Retorna tabelas e gráficos de análise exploratória do dataset,
    seguindo o notebook de referência do Tech Challenge:

    **Tabelas:** preview dos dados, estatísticas descritivas,
    distribuição das classes (M=1/B=0), valores nulos, correlação com target.

    **Gráficos (base64 PNG):** distribuição das classes, heatmap de
    correlação, boxplots das top features.
    """,
    responses={503: {"description": "Dataset não encontrado", "model": ErrorResponse}},
)
def eda_report(service: CancerService = Depends()):
    """Tabelas e gráficos de EDA do dataset de câncer de mama."""
    try:
        return service.get_eda()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"EDA indisponível: {str(e)}")


@router.get(
    "/metrics",
    response_model=CancerMetricsResponse,
    summary="Comparativo de métricas dos modelos",
    description="""
    Retorna tabela e gráfico comparando os modelos treinados
    (acuracia, recall, precisao, f1, auc), como o notebook de referência.
    """,
)
def metrics_report(service: CancerService = Depends()):
    """Tabela + gráfico comparativo dos modelos de câncer."""
    try:
        return service.get_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar métricas: {str(e)}")
