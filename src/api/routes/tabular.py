"""
Rotas de predição tabular (Câncer de Mama).

Classificação de tumores como benignos ou malignos
usando 30 features do dataset Breast Cancer Wisconsin.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from src.api.schemas import TabularPredictionRequest, TabularPredictionResponse, ErrorResponse
from src.api.tabular_service import TabularService

router = APIRouter(
    prefix="/predict/tabular",
    tags=["tabular"],
)


class ModelsListResponse(BaseModel):
    """Lista de modelos disponíveis."""
    available_models: list[str] = Field(
        ...,
        title="Modelos Disponíveis",
        description="Nomes dos modelos carregados e prontos para uso",
        examples=[["logistic_regression", "random_forest", "gradient_boosting"]],
    )


@router.post(
    "/",
    response_model=TabularPredictionResponse,
    summary="Classificar tumor de mama",
    description="""
    Prediz se um tumor de mama é benigno ou maligno com base em 30 features.

    **Features:** características dos núcleos celulares (radius, texture,
    perimeter, area, smoothness, compactness, concavity, concave points,
    symmetry, fractal dimension) para valores mean, se e worst.

    **Modelos disponíveis:**
    - logistic_regression: Regressão Logística
    - decision_tree: Árvore de Decisão
    - random_forest: Random Forest (padrão)
    - gradient_boosting: Gradient Boosting
    - svm: Support Vector Machine
    - knn: K-Nearest Neighbors
    - legacy_logistic_regression: Modelo legado (.pkl) migrado do projeto original

    **Explicabilidade:** Quando disponível, retorna as top 5 features
    mais importantes via SHAP.
    """,
    responses={
        200: {
            "description": "Predição realizada com sucesso",
            "model": TabularPredictionResponse,
        },
        400: {
            "description": "Requisição inválida (features ou modelo inválido)",
            "model": ErrorResponse,
        },
        503: {
            "description": "Modelos não carregados (necessário treinar primeiro)",
            "model": ErrorResponse,
        },
        500: {
            "description": "Erro interno no servidor",
            "model": ErrorResponse,
        },
    },
)
def predict_tabular(
    req: TabularPredictionRequest,
    service: TabularService = Depends(),
):
    """
    Prediz diagnóstico de câncer de mama a partir de 30 features.

    - **req**: 30 features clínicas + nome do modelo
    - **service**: Serviço de predição tabular (injetado automaticamente)

    Retorna o diagnóstico (M/B), probabilidades e explicabilidade SHAP.
    """
    if not service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Modelos tabulares não carregados. Execute o treinamento primeiro."
        )
    try:
        return service.predict(req.features, req.model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.get(
    "/models",
    response_model=ModelsListResponse,
    summary="Listar modelos tabulares",
    description="Retorna a lista de modelos de classificação tabular disponíveis para uso.",
)
def list_models(service: TabularService = Depends()):
    """
    Lista todos os modelos tabulares carregados e prontos para uso.

    Os modelos precisam ser treinados primeiro (via notebook ou script)
    e salvos no diretório `models/tabular/`.
    """
    return {"available_models": list(service.models.keys())}