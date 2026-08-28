"""
Rotas de predição de diabetes.

Predição de diabetes tipo 2 usando 8 features clínicas
do dataset Pima Indians Diabetes.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from src.api.schemas import DiabetesPredictionRequest, DiabetesPredictionResponse, ErrorResponse
from src.api.diabetes_service import DiabetesService

router = APIRouter(
    prefix="/predict/diabetes",
    tags=["diabetes"],
)


class DiabetesModelsListResponse(BaseModel):
    """Lista de modelos de diabetes disponíveis."""
    available_models: list[str] = Field(
        ...,
        title="Modelos Disponíveis",
        description="Nomes dos modelos de diabetes carregados",
        examples=[["logistic_regression", "random_forest", "gradient_boosting"]],
    )


@router.post(
    "/",
    response_model=DiabetesPredictionResponse,
    summary="Predizer risco de diabetes",
    description="""
    Avalia o risco de diabetes tipo 2 com base em 8 features clínicas.

    **Features:**
    - **pregnancies**: Número de gestações
    - **glucose**: Nível de glicose plasmática (mg/dL)
    - **blood_pressure**: Pressão arterial diastólica (mm Hg)
    - **skin_thickness**: Espessura da dobra cutânea (mm)
    - **insulin**: Insulina sérica (mu U/mL)
    - **bmi**: Índice de massa corporal (kg/m²)
    - **diabetes_pedigree**: Função de pedigree de diabetes
    - **age**: Idade (anos)

    **Modelos disponíveis:**
    - logistic_regression
    - decision_tree
    - random_forest (padrão)
    - gradient_boosting
    - svm
    - knn
    """,
    responses={
        200: {
            "description": "Predição realizada com sucesso",
            "model": DiabetesPredictionResponse,
        },
        400: {
            "description": "Requisição inválida",
            "model": ErrorResponse,
        },
        503: {
            "description": "Modelos não carregados",
            "model": ErrorResponse,
        },
        500: {
            "description": "Erro interno",
            "model": ErrorResponse,
        },
    },
)
def predict_diabetes(
    req: DiabetesPredictionRequest,
    service: DiabetesService = Depends(),
):
    """
    Prediz o risco de diabetes a partir de 8 features clínicas.

    - **req**: 8 features clínicas + modelo desejado
    - **service**: Serviço de predição (injetado)

    Retorna Positive (diabetes detectado), probabilidades e modelo usado.
    """
    if not service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Modelos de diabetes não carregados. Execute o treinamento primeiro."
        )
    try:
        return service.predict(req.features, req.model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@router.get(
    "/models",
    response_model=DiabetesModelsListResponse,
    summary="Listar modelos de diabetes",
    description="Retorna a lista de modelos de diabetes disponíveis para uso.",
)
def list_models(service: DiabetesService = Depends()):
    """
    Lista modelos de diabetes carregados.

    Os modelos precisam ser treinados primeiro via
    `train_diabetes_models()` ou pelos notebooks.
    """
    return {"available_models": service.available_models}


@router.get(
    "/info",
    summary="Informações do dataset",
    description="Retorna informações detalhadas sobre o dataset Pima Indians Diabetes Database.",
)
def model_info():
    """
    Informações sobre o dataset Pima Indians Diabetes.

    Retorna:
    - Nome e descrição do dataset
    - Número de amostras e features
    - Descrição de cada feature
    - Classes disponíveis
    """
    from src.diabetes.dataset import get_diabetes_info
    return get_diabetes_info()
