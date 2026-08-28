"""
Rota de health check.

Verifica o status da API e o carregamento dos modelos.
"""
from fastapi import APIRouter, Depends
from src.api.schemas import HealthResponse
from src.api.tabular_service import TabularService
from src.api.image_service import ImageService
from src.api.diabetes_service import DiabetesService
from src.api.cancer_service import CancerService
from src.api.pcos_service import PCOSService

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Verificar status da API",
    description="Retorna o estado atual da API e indica se cada modelo foi carregado com sucesso.",
    response_description="Status da API e flags de carregamento dos modelos",
)
def health(
    tabular: TabularService = Depends(),
    image_svc: ImageService = Depends(),
    diabetes_svc: DiabetesService = Depends(),
    cancer_svc: CancerService = Depends(),
    pcos_svc: PCOSService = Depends(),
):
    """
    Verifica a saúde da API e o status dos modelos.

    Retorna:
    - **status**: "ok" se a API estiver funcionando
    - **tabular_model_loaded**: modelos de câncer de mama carregados
    - **cnn_model_loaded**: modelo de mamografia carregado
    - **diabetes_model_loaded**: modelos de diabetes carregados
    - **cancer_model_loaded**: modelos de câncer (src/cancer) carregados
    - **version**: versão da API
    - **cache**: estatísticas do cache de predições
    """
    response = HealthResponse(
        status="ok",
        tabular_model_loaded=tabular.is_loaded,
        cnn_model_loaded=image_svc.is_loaded,
        diabetes_model_loaded=diabetes_svc.is_loaded,
        cancer_model_loaded=cancer_svc.is_loaded,
        pcos_model_loaded=pcos_svc.is_loaded,
    )
    # Adiciona estatísticas do cache
    cache_stats = {
        "tabular": tabular.cache.stats,
        "image": image_svc.cache.stats,
        "diabetes": diabetes_svc.cache.stats,
        "cancer": cancer_svc.cache.stats,
        "pcos": pcos_svc.cache.stats,
    }
    # Retorna como dict para incluir cache
    return response.model_dump() | {"cache": cache_stats}