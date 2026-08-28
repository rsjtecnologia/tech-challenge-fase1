"""Rotas de classificação de síndrome dos ovários policísticos."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.pcos_service import PCOSService
from src.api.schemas import PCOSPredictionRequest, PCOSPredictionResponse
from src.pcos.dataset import get_pcos_info, get_pcos_samples

router = APIRouter(prefix="/predict/pcos", tags=["pcos"])


class PCOSModelsListResponse(BaseModel):
    """Lista de modelos de SOP disponíveis."""

    available_models: list[str]


@router.post("/", response_model=PCOSPredictionResponse, summary="Predizer risco de SOP")
def predict_pcos(request: PCOSPredictionRequest, service: PCOSService = Depends()):
    """Classifica a presença de SOP a partir de 41 medidas clínicas."""
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Modelos de SOP não carregados.")
    try:
        return service.predict(request.features, request.model_name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/models", response_model=PCOSModelsListResponse, summary="Listar modelos de SOP")
def list_models(service: PCOSService = Depends()):
    return {"available_models": service.available_models}


@router.get("/info", summary="Informações do dataset de SOP")
def model_info():
    return get_pcos_info()


@router.get("/samples", summary="Amostras reais do dataset de SOP")
def samples():
    return get_pcos_samples()


@router.get("/eda", summary="Relatório EDA de SOP")
def eda_report(service: PCOSService = Depends()):
    """Retorna tabelas e gráficos exploratórios como no módulo de câncer."""
    try:
        return service.get_eda()
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"EDA indisponível: {error}") from error


@router.get("/metrics", summary="Comparativo de métricas dos modelos de SOP")
def metrics_report(service: PCOSService = Depends()):
    """Retorna métricas de teste dos seis modelos de SOP."""
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Modelos de SOP não carregados.")
    return service.get_metrics()
