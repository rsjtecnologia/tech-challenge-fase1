"""
Rotas administrativas para gerenciamento do cache.

Permite visualizar estatísticas e limpar o cache de predições.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from src.api.tabular_service import TabularService
from src.api.image_service import ImageService
from src.api.diabetes_service import DiabetesService
from src.api.cancer_service import CancerService
from src.api.pcos_service import PCOSService

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
)

SERVICES_NAMES = {
    "tabular": "Câncer de Mama (Tabular)",
    "image": "Mamografia (CNN)",
    "diabetes": "Diabetes",
    "cancer": "Câncer de Mama (módulo cancer)",
    "pcos": "Síndrome dos Ovários Policísticos",
}


class CacheStatusResponse(BaseModel):
    """Status detalhado de todos os caches."""
    services: dict = Field(
        ...,
        title="Serviços",
        description="Estatísticas de cache por serviço",
    )
    global_hit_rate: float = Field(
        ...,
        title="Hit Rate Global",
        description="Taxa de acerto combinada de todos os caches",
        ge=0.0,
        le=1.0,
    )
    total_entries: int = Field(
        ...,
        title="Total de Entradas",
        description="Total de entradas em todos os caches",
    )


class CacheClearResponse(BaseModel):
    """Resultado da limpeza do cache."""
    message: str = Field(
        ...,
        title="Mensagem",
        description="Resultado da operação",
    )
    cleared: list[str] = Field(
        ...,
        title="Caches Limpos",
        description="Lista dos caches que foram limpos",
    )


def _get_cache_stats(
    tabular: TabularService,
    image_svc: ImageService,
    diabetes_svc: DiabetesService,
    cancer_svc: CancerService,
    pcos_svc: PCOSService,
) -> dict:
    """Coleta estatísticas de todos os caches."""
    caches = {
        "tabular": tabular.cache,
        "image": image_svc.cache,
        "diabetes": diabetes_svc.cache,
        "cancer": cancer_svc.cache,
        "pcos": pcos_svc.cache,
    }

    services_data = {}
    total_entries = 0
    total_global = 0
    total_hits = 0

    for name, cache in caches.items():
        stats = cache.stats
        services_data[name] = {
            "display_name": SERVICES_NAMES.get(name, name),
            "stats": stats,
        }
        total_entries += stats["current_size"]
        total_global += stats["total_requests"]
        total_hits += stats["hits"]

    global_hit_rate = total_hits / total_global if total_global > 0 else 0.0

    return {
        "services": services_data,
        "global_hit_rate": round(global_hit_rate, 4),
        "total_entries": total_entries,
    }


@router.get(
    "/cache",
    response_model=CacheStatusResponse,
    summary="Status do cache",
    description="Retorna estatísticas detalhadas de todos os caches de predição.",
)
def cache_status(
    tabular: TabularService = Depends(),
    image_svc: ImageService = Depends(),
    diabetes_svc: DiabetesService = Depends(),
    cancer_svc: CancerService = Depends(),
    pcos_svc: PCOSService = Depends(),
):
    """
    Retorna status de todos os caches.

    Inclui:
    - **services**: Estatísticas individuais por serviço
    - **global_hit_rate**: Taxa de acerto combinada
    - **total_entries**: Total de entradas cacheadas

    Use `DELETE /admin/cache` para limpar todos os caches.
    Use `DELETE /admin/cache/{service}` para limpar um cache específico.
    """
    return _get_cache_stats(tabular, image_svc, diabetes_svc, cancer_svc, pcos_svc)


@router.delete(
    "/cache",
    response_model=CacheClearResponse,
    summary="Limpar todos os caches",
    description="Remove todas as entradas de todos os caches de predição.",
)
def clear_all_caches(
    tabular: TabularService = Depends(),
    image_svc: ImageService = Depends(),
    diabetes_svc: DiabetesService = Depends(),
    cancer_svc: CancerService = Depends(),
    pcos_svc: PCOSService = Depends(),
):
    """
    Limpa todos os caches de predição.

    Para limpar apenas um cache específico, use:
    `DELETE /admin/cache/{tabular|image|diabetes|cancer}`
    """
    caches = {
        "tabular": tabular.cache,
        "image": image_svc.cache,
        "diabetes": diabetes_svc.cache,
        "cancer": cancer_svc.cache,
        "pcos": pcos_svc.cache,
    }

    cleared = []
    for name, cache in caches.items():
        cache.clear()
        cleared.append(name)

    return CacheClearResponse(
        message=f"{len(cleared)} caches limpos com sucesso",
        cleared=cleared,
    )


@router.delete(
    "/cache/{service_name}",
    response_model=CacheClearResponse,
    summary="Limpar cache específico",
    description="Remove todas as entradas do cache de um serviço específico.",
    responses={
        404: {
            "description": "Serviço não encontrado",
        },
    },
)
def clear_service_cache(
    service_name: str,
    tabular: TabularService = Depends(),
    image_svc: ImageService = Depends(),
    diabetes_svc: DiabetesService = Depends(),
    cancer_svc: CancerService = Depends(),
    pcos_svc: PCOSService = Depends(),
):
    """
    Limpa o cache de um serviço específico.

    **Serviços disponíveis:**
    - `tabular`: Câncer de Mama
    - `image`: Mamografia (CNN)
    - `diabetes`: Diabetes
    - `cancer`: Câncer (módulo src/cancer)
    """
    cache_map = {
        "tabular": tabular.cache,
        "image": image_svc.cache,
        "diabetes": diabetes_svc.cache,
        "cancer": cancer_svc.cache,
        "pcos": pcos_svc.cache,
    }

    if service_name not in cache_map:
        available = list(cache_map.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Serviço '{service_name}' não encontrado. Disponíveis: {available}",
        )

    cache_map[service_name].clear()
    return CacheClearResponse(
        message=f"Cache '{service_name}' limpo com sucesso",
        cleared=[service_name],
    )
