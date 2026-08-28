"""
Pydantic schemas para validação da API.

Define os modelos de request/response com validações,
descrições e exemplos para documentação Swagger.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional


class TabularPredictionRequest(BaseModel):
    """
    Requisição para predição de câncer de mama.

    Recebe 30 features do dataset Breast Cancer Wisconsin
    e o nome do modelo a ser utilizado.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": [
                    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
                    0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
                    0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119,
                    0.2654, 0.4601, 0.1189
                ],
                "model_name": "random_forest"
            }
        }
    )

    features: List[float] = Field(
        ...,
        title="Features Clínicas",
        description="30 features do dataset Breast Cancer Wisconsin",
        min_length=30,
        max_length=30,
        examples=[[
            17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
            0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
            0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
            25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119,
            0.2654, 0.4601, 0.1189
        ]],
    )
    model_name: str = Field(
        default="random_forest",
        title="Nome do Modelo",
        description="Modelo ML para classificação",
        examples=["random_forest"],
        pattern="^(logistic_regression|decision_tree|random_forest|gradient_boosting|svm|knn|legacy_logistic_regression)$",
    )


class TabularPredictionResponse(BaseModel):
    """Resposta da predição de câncer de mama."""
    diagnosis: str = Field(
        ...,
        title="Diagnóstico",
        description="Resultado: M (maligno) ou B (benigno)",
        examples=["M"],
    )
    probability_malignant: float = Field(
        ...,
        title="Probabilidade Maligno",
        description="Probabilidade do tumor ser maligno (0 a 1)",
        ge=0.0,
        le=1.0,
        examples=[0.95],
    )
    probability_benign: float = Field(
        ...,
        title="Probabilidade Benigno",
        description="Probabilidade do tumor ser benigno (0 a 1)",
        ge=0.0,
        le=1.0,
        examples=[0.05],
    )
    model_used: str = Field(
        ...,
        title="Modelo Utilizado",
        description="Nome do modelo que gerou a predição",
        examples=["random_forest"],
    )
    shap_top_features: Optional[dict] = Field(
        None,
        title="Features SHAP",
        description="Top 5 features mais importantes (SHAP explainability)",
        examples=[{"worst radius": 0.45, "worst area": 0.32}],
    )


class ImagePredictionResponse(BaseModel):
    """Resposta da classificação de mamografia."""
    prediction: str = Field(
        ...,
        title="Predição",
        description="Classe predita: benign ou malignant",
        examples=["malignant"],
    )
    probability_malignant: float = Field(
        ...,
        title="Probabilidade Maligno",
        description="Probabilidade de ser maligno",
        ge=0.0,
        le=1.0,
    )
    probability_benign: float = Field(
        ...,
        title="Probabilidade Benigno",
        description="Probabilidade de ser benigno",
        ge=0.0,
        le=1.0,
    )
    confidence: float = Field(
        ...,
        title="Confiança",
        description="Nível de confiança da predição",
        ge=0.0,
        le=1.0,
        examples=[0.95],
    )
    model_used: str = Field(
        default="mobilenet_mammo_pytorch",
        title="Modelo",
        description="Arquitetura do modelo utilizado",
        examples=["mobilenet_mammo_pytorch"],
    )


class HealthResponse(BaseModel):
    """Status de saúde da API."""
    status: str = Field(
        ...,
        title="Status da API",
        description="Estado atual da API",
        examples=["ok"],
    )
    tabular_model_loaded: bool = Field(
        ...,
        title="Modelo Tabular Carregado",
        description="Indica se os modelos tabulares foram carregados",
    )
    cnn_model_loaded: bool = Field(
        ...,
        title="Modelo CNN Carregado",
        description="Indica se o modelo de mamografia foi carregado",
    )
    diabetes_model_loaded: bool = Field(
        False,
        title="Modelo Diabetes Carregado",
        description="Indica se os modelos de diabetes foram carregados",
    )
    cancer_model_loaded: bool = Field(
        False,
        title="Modelo Câncer Carregado",
        description="Indica se os modelos de câncer (src/cancer) foram carregados",
    )
    pcos_model_loaded: bool = Field(
        False,
        title="Modelo de SOP Carregado",
        description="Indica se os modelos de síndrome dos ovários policísticos foram carregados",
    )
    version: str = Field(
        default="2.0.0",
        title="Versão da API",
        description="Versão atual da API",
        examples=["2.0.0"],
    )
    cache: Optional[dict] = Field(
        None,
        title="Cache de Predições",
        description="Estatísticas do cache de predições (tabular e diabetes)",
    )


class DiabetesPredictionRequest(BaseModel):
    """
    Requisição para predição de diabetes.

    Recebe 8 features clínicas do dataset Pima Indians Diabetes
    e o nome do modelo a ser utilizado.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": [6, 148, 72, 35, 0, 33.6, 0.627, 50],
                "model_name": "random_forest"
            }
        }
    )

    features: List[float] = Field(
        ...,
        title="Features Clínicas",
        description="8 features: pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age",
        min_length=8,
        max_length=8,
        examples=[[6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0]],
    )
    model_name: str = Field(
        default="random_forest",
        title="Nome do Modelo",
        description="Modelo ML para classificação de diabetes",
        examples=["random_forest"],
        pattern="^(logistic_regression|decision_tree|random_forest|gradient_boosting|svm|knn)$",
    )


class DiabetesPredictionResponse(BaseModel):
    """Resposta da predição de diabetes."""
    prediction: str = Field(
        ...,
        title="Predição",
        description="Positive (diabetes detectado) ou Negative",
        examples=["Positive"],
    )
    probability_positive: float = Field(
        ...,
        title="Probabilidade Positivo",
        description="Probabilidade de diabetes (0 a 1)",
        ge=0.0,
        le=1.0,
    )
    probability_negative: float = Field(
        ...,
        title="Probabilidade Negativo",
        description="Probabilidade de não ter diabetes (0 a 1)",
        ge=0.0,
        le=1.0,
    )
    model_used: str = Field(
        ...,
        title="Modelo Utilizado",
        description="Nome do modelo que gerou a predição",
    )
    features: List[float] = Field(
        ...,
        title="Features de Entrada",
        description="Features enviadas na requisição (eco)",
    )


class PCOSPredictionRequest(BaseModel):
    """Requisição de classificação de síndrome dos ovários policísticos."""
    features: List[float] = Field(
        ...,
        title="Features Clínicas",
        description="41 medidas clínicas do dataset de SOP",
        min_length=41,
        max_length=41,
    )
    model_name: str = Field(
        default="random_forest",
        pattern="^(logistic_regression|decision_tree|random_forest|gradient_boosting|svm|knn)$",
    )


class PCOSPredictionResponse(BaseModel):
    """Resposta da classificação de SOP."""
    prediction: str = Field(..., description="PCOS ou No PCOS")
    probability_positive: float = Field(..., ge=0.0, le=1.0)
    probability_negative: float = Field(..., ge=0.0, le=1.0)
    model_used: str
    features: List[float]
    graphs: Optional[dict] = Field(
        None, description="Gráfico das variáveis clínicas mais relevantes"
    )
    tables: Optional[dict] = Field(
        None, description="Tabela das variáveis clínicas mais relevantes"
    )


class GradCAMResponse(BaseModel):
    """
    Resposta com Grad-CAM heatmap.

    Inclui a predição da mamografia e o mapa de calor
    indicando as regiões que mais influenciaram a decisão.
    """
    prediction: str = Field(
        ...,
        title="Predição",
        description="benign ou malignant",
        examples=["malignant"],
    )
    probability_malignant: float = Field(
        ..., title="Probabilidade Maligno", ge=0.0, le=1.0
    )
    probability_benign: float = Field(
        ..., title="Probabilidade Benigno", ge=0.0, le=1.0
    )
    confidence: float = Field(
        ..., title="Confiança", ge=0.0, le=1.0, examples=[0.95]
    )
    model_used: str = Field(
        ..., title="Modelo", description="Arquitetura do modelo"
    )
    heatmap: List[List[float]] = Field(
        ...,
        title="Heatmap Grad-CAM",
        description="Matriz 2D (HxW) com valores de 0 a 1 indicando importância dos pixels",
    )
    heatmap_shape: List[int] = Field(
        ...,
        title="Dimensões do Heatmap",
        description="Altura e largura da matriz heatmap [H, W]",
        min_length=2,
        max_length=2,
        examples=[[224, 224]],
    )


class CancerPredictionRequest(BaseModel):
    """
    Requisição para predição de câncer de mama (módulo src/cancer).

    Mesmas 30 features do dataset Breast Cancer Wisconsin.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": [
                    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
                    0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
                    0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119,
                    0.2654, 0.4601, 0.1189
                ],
                "model_name": "random_forest"
            }
        }
    )

    features: List[float] = Field(
        ...,
        title="Features Clínicas",
        description="30 features do dataset Breast Cancer Wisconsin",
        min_length=30,
        max_length=30,
        examples=[[
            17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
            0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4,
            0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
            25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119,
            0.2654, 0.4601, 0.1189
        ]],
    )
    model_name: str = Field(
        default="random_forest",
        title="Nome do Modelo",
        description="Modelo ML para classificação",
        examples=["random_forest"],
        pattern="^(logistic_regression|decision_tree|random_forest|gradient_boosting|svm|knn)$",
    )


class CancerPredictionResponse(BaseModel):
    """
    Resposta da predição de câncer de mama com gráficos e tabelas.

    Inclui a predição (M/B), probabilidades, explicabilidade (SHAP) e,
    quando disponível, um gráfico e uma tabela das features mais relevantes.
    """
    diagnosis: str = Field(
        ...,
        title="Diagnóstico",
        description="M (maligno) ou B (benigno)",
        examples=["M"],
    )
    probability_malignant: float = Field(
        ...,
        title="Probabilidade Maligno",
        description="Probabilidade do tumor ser maligno (0 a 1)",
        ge=0.0,
        le=1.0,
    )
    probability_benign: float = Field(
        ...,
        title="Probabilidade Benigno",
        description="Probabilidade do tumor ser benigno (0 a 1)",
        ge=0.0,
        le=1.0,
    )
    model_used: str = Field(
        ...,
        title="Modelo Utilizado",
        description="Nome do modelo que gerou a predição",
        examples=["random_forest"],
    )
    features: List[float] = Field(
        ...,
        title="Features de Entrada",
        description="Features enviadas na requisição (eco)",
    )
    graphs: Optional[dict] = Field(
        None,
        title="Gráficos",
        description="Gráficos em base64 (ex.: importância das features/SHAP)",
        examples=[{"shap": "data:image/png;base64,..."}],
    )
    tables: Optional[dict] = Field(
        None,
        title="Tabelas",
        description="Tabelas (ex.: top features da predição)",
        examples=[{"top_features": [{"feature": "worst radius", "importancia": 0.2}]}],
    )


class CancerEDAResponse(BaseModel):
    """
    Relatório EDA do dataset de câncer de mama.

    Retorna tabelas (preview, describe, classes, missing, correlação)
    e gráficos em base64 (classes, correlação, boxplots).
    """
    info: dict = Field(..., title="Informações do Dataset")
    tables: dict = Field(..., title="Tabelas de EDA")
    graphs: dict = Field(..., title="Gráficos de EDA (base64)")


class CancerMetricsResponse(BaseModel):
    """Comparativo de métricas dos modelos treinados."""
    tables: dict = Field(..., title="Tabelas", description="Tabela de métricas por modelo")
    graphs: dict = Field(..., title="Gráficos", description="Gráfico comparativo (base64)")


class CancerModelsListResponse(BaseModel):
    """Lista de modelos de câncer disponíveis."""
    available_models: list[str] = Field(
        ...,
        title="Modelos Disponíveis",
        description="Nomes dos modelos de câncer carregados",
        examples=[["logistic_regression", "random_forest", "gradient_boosting"]],
    )


class ErrorResponse(BaseModel):
    """Resposta padronizada de erro."""
    error: str = Field(
        ...,
        title="Código do Erro",
        description="Tipo do erro ocorrido",
        examples=["Formato não suportado"],
    )
    detail: Optional[str] = Field(
        None,
        title="Detalhes do Erro",
        description="Mensagem detalhada sobre o erro",
        examples=["Formato não suportado. Use PNG ou JPG."],
    )