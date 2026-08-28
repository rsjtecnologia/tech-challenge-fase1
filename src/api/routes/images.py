"""
Rotas de predição por imagem (Mamografias).

Classificação de mamografias usando CNN (MobileNetV2),
visualização com Grad-CAM, EDA, métricas e galeria de amostras.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from src.api.schemas import ImagePredictionResponse, GradCAMResponse, ErrorResponse
from src.api.image_service import ImageService
from src.cnn.dataset import index_cbis_images

router = APIRouter(
    prefix="/predict/image",
    tags=["images"],
)

ALLOWED_FORMATS = ["image/png", "image/jpeg", "image/jpg"]
ALLOWED_EXTENSIONS = "PNG, JPG/JPEG"


@router.get("/samples", summary="Listar mamografias locais (galeria)")
def list_local_samples():
    """Retorna as mamografias CBIS-DDSM preparadas (train/val/test) para o frontend."""
    samples = index_cbis_images()
    split_counts = {split: 0 for split in ("train", "val", "test")}
    for item in samples.values():
        split_counts[item["split"]] = split_counts.get(item["split"], 0) + 1
    return {
        "samples": [
            {
                "id": image_id,
                "label": item["label"],
                "split": item["split"],
                "image_url": f"/predict/image/samples/{image_id}",
            }
            for image_id, item in samples.items()
        ],
        "total": len(samples),
        "split_counts": split_counts,
    }


@router.get("/samples/{sample_id}", summary="Exibir mamografia local")
def get_local_sample_image(sample_id: str):
    """Serve uma imagem local previamente catalogada (miniatura/visualização)."""
    item = index_cbis_images().get(sample_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Amostra de mamografia não encontrada.")
    media_type = (
        "image/png" if item["path"].suffix.lower() == ".png" else "image/jpeg"
    )
    return FileResponse(item["path"], media_type=media_type)


@router.post("/mammography/sample/{sample_id}", response_model=ImagePredictionResponse)
def predict_local_sample(sample_id: str, service: ImageService = Depends()):
    """Prediz uma das mamografias já presentes no dataset local (qualquer split)."""
    item = index_cbis_images().get(sample_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Amostra de mamografia não encontrada.")
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Modelo CNN não carregado. Execute o notebook de treinamento.")
    try:
        return service.predict_from_bytes(item["path"].read_bytes())
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {error}") from error


@router.get("/info", summary="Informações do dataset de mamografias")
def image_info(service: ImageService = Depends()):
    """Retorna informações do CBIS-DDSM preparado e do modelo treinado."""
    try:
        return service.get_info()
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar informações: {error}") from error


@router.get("/metrics", summary="Métricas de teste do modelo CNN")
def image_metrics(service: ImageService = Depends()):
    """Retorna tabela + gráfico das métricas de teste salvas no checkpoint."""
    try:
        return service.get_metrics()
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar métricas: {error}") from error


@router.get("/eda", summary="Relatório EDA das mamografias")
def image_eda(service: ImageService = Depends()):
    """Retorna tabelas e gráficos exploratórios (distribuição por split/classe)."""
    try:
        return service.get_eda()
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar EDA: {error}") from error


@router.post(
    "/mammography",
    response_model=ImagePredictionResponse,
    summary="Classificar mamografia",
    description="""
    Envie uma mamografia (PNG ou JPG) e receba a classificação
    entre benigno e maligno com nível de confiança.

    **Formato aceito:**
    - PNG, JPG/JPEG
    - RGB ou grayscale (automáticamente convertido)
    - Qualquer resolução (redimensionada para 224x224)

    **Modelo:** MobileNetV2 (PyTorch) com fine-tuning em CBIS-DDSM

    **Exemplo:**
    ```bash
    curl -X POST http://localhost:8000/predict/image/mammography \\
      -F "file=@mamografia.png"
    ```
    """,
    responses={
        200: {
            "description": "Classificação realizada com sucesso",
            "model": ImagePredictionResponse,
        },
        400: {
            "description": f"Formato não suportado. Use: {ALLOWED_EXTENSIONS}",
            "model": ErrorResponse,
        },
        503: {
            "description": "Modelo CNN não carregado (necessário treinar primeiro)",
            "model": ErrorResponse,
        },
        500: {
            "description": "Erro interno no processamento da imagem",
            "model": ErrorResponse,
        },
    },
)
async def predict_mammography(
    file: UploadFile = File(
        ...,
        title="Arquivo de Mamografia",
        description="Imagem de mamografia nos formatos PNG ou JPG",
    ),
    service: ImageService = Depends(),
):
    """
    Classifica uma mamografia como benigna ou maligna.

    Parâmetros:
    - **file**: Arquivo de imagem (PNG/JPG) contendo a mamografia

    Retorna:
    - **prediction**: Classe predita (benign/malignant)
    - **probabilities**: Probabilidades para cada classe
    - **confidence**: Nível de confiança da predição
    - **model_used**: Arquitetura do modelo utilizado
    """
    # Validação de formato
    if file.content_type not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato '{file.content_type}' não suportado. Use: {ALLOWED_EXTENSIONS}"
        )

    # Verifica modelo carregado
    if not service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Modelo CNN não carregado. Treine o modelo primeiro."
        )

    try:
        contents = await file.read()
        return service.predict_from_bytes(contents)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro na predição: {str(e)}"
        )


@router.post(
    "/mammography/gradcam",
    response_model=GradCAMResponse,
    summary="Classificar mamografia com Grad-CAM",
    description="""
    Envie uma mamografia e receba a classificação acompanhada de
    um mapa de calor Grad-CAM explicando a decisão do modelo.

    O heatmap destaca as regiões da imagem que mais contribuíram
    para a decisão final (última camada convolucional da MobileNetV2).
    """,
    responses={
        200: {
            "description": "Classificação + Grad-CAM gerados com sucesso",
            "model": GradCAMResponse,
        },
        400: {
            "description": f"Formato não suportado. Use: {ALLOWED_EXTENSIONS}",
            "model": ErrorResponse,
        },
        503: {
            "description": "Modelo CNN não carregado",
            "model": ErrorResponse,
        },
        500: {
            "description": "Erro ao gerar Grad-CAM",
            "model": ErrorResponse,
        },
    },
)
async def predict_mammography_gradcam(
    file: UploadFile = File(
        ...,
        title="Arquivo de Mamografia",
        description="Imagem de mamografia (PNG ou JPG) para análise com Grad-CAM",
    ),
    service: ImageService = Depends(),
):
    """
    Classifica mamografia e retorna Grad-CAM heatmap.

    Parâmetros:
    - **file**: Arquivo de imagem (PNG/JPG)

    Retorna:
    - **prediction**: Classe predita
    - **probabilities**: Probabilidades
    - **heatmap**: Matriz 2D do mapa de calor (224x224)
    - **heatmap_shape**: Dimensões do heatmap

    **Interpretação do heatmap:**
    Valores mais altos (próximos de 1) indicam regiões
    que mais influenciaram a decisão do modelo.
    """
    if file.content_type not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato '{file.content_type}' não suportado. Use: {ALLOWED_EXTENSIONS}"
        )

    if not service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Modelo CNN não carregado. Treine o modelo primeiro."
        )

    try:
        contents = await file.read()
        return service.compute_gradcam_from_bytes(contents)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro no Grad-CAM: {str(e)}"
        )