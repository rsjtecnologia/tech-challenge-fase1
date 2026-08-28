"""
FastAPI application principal.

Health AI — Sistema de IA para saude feminina
Oferece endpoints para diagnostico de cancer de mama,
classificacao de mamografias e predicao de diabetes.
"""
import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from src.api.tabular_service import TabularService
from src.api.image_service import ImageService
from src.api.diabetes_service import DiabetesService
from src.api.cancer_service import CancerService
from src.api.pcos_service import PCOSService
from src.api.routes import health, tabular, images, diabetes, cancer, pcos, admin

# Configuração global de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação.

    **Startup:**
    - Carrega modelos tabulares (Random Forest, Logistic Regression, Gradient Boosting)
    - Carrega modelo CNN (MobileNetV2) para mamografias
    - Carrega modelos de diabetes (quando disponíveis)
    - Carrega modelos de câncer de mama (módulo src/cancer)
    - Gera relatórios EDA/métricas com gráficos e tabelas

    **Shutdown:**
    - Libera recursos
    """
    app.state.tabular_service = TabularService()
    app.state.image_service = ImageService()
    app.state.diabetes_service = DiabetesService()
    app.state.cancer_service = CancerService()
    app.state.pcos_service = PCOSService()

    # Override das dependências (permite usar Depends() nas rotas)
    app.dependency_overrides[TabularService] = lambda: app.state.tabular_service
    app.dependency_overrides[ImageService] = lambda: app.state.image_service
    app.dependency_overrides[DiabetesService] = lambda: app.state.diabetes_service
    app.dependency_overrides[CancerService] = lambda: app.state.cancer_service
    app.dependency_overrides[PCOSService] = lambda: app.state.pcos_service

    logger.info("API Health AI iniciada")
    yield
    logger.info("Encerrando API...")


app = FastAPI(
    title="Health AI API — Tech Challenge Fase 1",
    description="""
    # \U0001F3E5 Health AI — Sistema de Intelig\u00eancia Artificial para Sa\u00fade Feminina

    Esta API disponibiliza modelos de machine learning e deep learning para
    **diagn\u00f3stico e triagem** de condi\u00e7\u00f5es de sa\u00fade feminina.

    ## \U0001F4CC Endpoints Dispon\u00edveis

    | Grupo | Endpoint | Descri\u00e7\u00e3o |
    |-------|----------|-------------|
    | \U0001F31F **Sa\u00fade** | `GET /health` | Status dos servi\u00e7os e modelos |
    | \U0001f9ec **C\u00e2ncer de Mama** | `POST /predict/tabular/` | Classifica\u00e7\u00e3o por 30 features cl\u00ednicas |
    | \U0001f9ec **C\u00e2ncer de Mama** | `GET /predict/tabular/models` | Lista modelos tabulares dispon\u00edveis |
    | \U0001f9ec **C\u00e2ncer de Mama** | `POST /predict/cancer/` | Predi\u00e7\u00e3o com gr\u00e1fico + tabela (SHAP) |
    | \U0001f9ec **C\u00e2ncer de Mama** | `GET /predict/cancer/models` | Lista modelos de c\u00e2ncer |
    | \U0001f9ec **C\u00e2ncer de Mama** | `GET /predict/cancer/info` | Informa\u00e7\u00f5es do dataset |
    | \U0001f9ec **C\u00e2ncer de Mama** | `GET /predict/cancer/eda` | Gr\u00e1ficos e tabelas de EDA |
    | \U0001f9ec **C\u00e2ncer de Mama** | `GET /predict/cancer/metrics` | Comparativo de m\u00e9tricas + gr\u00e1fico |
    | \U0001f9ec **C\u00e2ncer de Mama** | `POST /predict/image/mammography` | Classifica\u00e7\u00e3o por mamografia |
    | \U0001f9ec **C\u00e2ncer de Mama** | `POST /predict/image/mammography/gradcam` | Mamografia + mapa de calor explicativo |
    | \U0001f9ec **Diabetes** | `POST /predict/diabetes/` | Predi\u00e7\u00e3o de diabetes (8 features) |
    | \U0001f9ec **Diabetes** | `GET /predict/diabetes/models` | Lista modelos diabetes dispon\u00edveis |
    | \U0001f9ec **Diabetes** | `GET /predict/diabetes/info` | Informa\u00e7\u00f5es do dataset |

    ## \U0001f916 Modelos Dispon\u00edveis

    - **Tabular (C\u00e2ncer de Mama):** Logistic Regression, Random Forest, Gradient Boosting
    - **Imagem (Mamografia):** MobileNetV2 (PyTorch) com Fine-tuning
    - **Diabetes:** Logistic Regression, Random Forest, Gradient Boosting

    ## \u26a0\ufe0f Notas

    - Os modelos precisam ser treinados antes do uso
    - Consulte `/health` para verificar o status de carregamento
    - SHAP explainability dispon\u00edvel para predi\u00e7\u00f5es tabulares
    - Grad-CAM dispon\u00edvel para visualiza\u00e7\u00e3o de mamografias
    """,
    summary="API de diagn\u00f3stico de sa\u00fade feminina com IA",
    version="2.0.0",
    terms_of_service="https://github.com/ricoi/tech-challenge-fase1",
    contact={
        "name": "Tech Challenge Fase 1",
        "url": "https://github.com/ricoi/tech-challenge-fase1",
        "email": "techchallenge@fase1.com",
    },
    license_info={
        "name": "MIT",
        "identifier": "MIT",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ordenar tags no Swagger
# Serve arquivos estaticos (frontend.html, etc.)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

app.include_router(health.router)
app.include_router(tabular.router)
app.include_router(images.router)
app.include_router(diabetes.router)
app.include_router(cancer.router)
app.include_router(pcos.router)
app.include_router(admin.router)


@app.get("/", tags=["root"])
async def root():
    """
    Raiz da API — Retorna links para documentação.

    Fornece acesso r\u00e1pido \u00e0 documenta\u00e7\u00e3o interativa.
    """
    return {
        "message": "\U0001f3e5 Tech Challenge Fase 1 — Health AI API",
        "version": "2.0.0",
        "status": "running",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
        },
        "endpoints": {
            "health": "/health",
            "tabular_predict": "POST /predict/tabular/",
            "tabular_models": "GET /predict/tabular/models",
            "mammography": "POST /predict/image/mammography",
            "mammography_gradcam": "POST /predict/image/mammography/gradcam",
            "diabetes_predict": "POST /predict/diabetes/",
            "diabetes_models": "GET /predict/diabetes/models",
            "diabetes_info": "GET /predict/diabetes/info",
            "pcos_predict": "POST /predict/pcos/",
            "pcos_models": "GET /predict/pcos/models",
            "pcos_info": "GET /predict/pcos/info",
            "cancer_predict": "POST /predict/cancer/",
            "cancer_models": "GET /predict/cancer/models",
            "cancer_info": "GET /predict/cancer/info",
            "cancer_eda": "GET /predict/cancer/eda",
            "cancer_metrics": "GET /predict/cancer/metrics",
        }
    }