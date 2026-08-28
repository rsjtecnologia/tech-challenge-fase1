# Relatório Técnico — Tech Challenge Fase 1

## Health AI: Sistema de Apoio à Triagem e Diagnóstico em Saúde Feminina

| Campo | Descrição |
|---|---|
| **Projeto** | Tech Challenge Fase 1 — Health AI |
| **Autor** | Aluno Pós-Tech (Engenharia de Machine Learning / IA) |
| **Versão do relatório** | 1.0 |
| **Data** | Agosto de 2026 |
| **Repositório** | `tech-challenge-fase1` |
| **Stack** | Python 3.10+, FastAPI, scikit-learn, PyTorch, SHAP, MLflow |

---

## 1. Resumo Executivo

Este trabalho apresenta a Fase 1 de um sistema inteligente de **apoio à triagem e identificação precoce de riscos em saúde feminina**, cobrindo quatro frentes:

1. **Câncer de mama (tabular)** — classificação binária (benigno/maligno) a partir de 30 características morfológicas do núcleo celular (*Breast Cancer Wisconsin*).
2. **Mamografia (imagens)** — classificação binária com **CNN MobileNetV2** + **Grad-CAM** (CBIS-DDSM).
3. **Diabetes** — classificação binária com 8 features clínicas (*Pima Indians Diabetes*).
4. **Síndrome dos Ovários Policísticos (SOP)** — classificação binária com 41 features clínicas.

Cada frente tabular compara **6 algoritmos** (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM e KNN). O melhor desempenho no conjunto de teste foi:

| Problema | Melhor modelo | Acurácia | AUC-ROC |
|---|---|---|---|
| Câncer de mama | Random Forest / SVM | **97,37%** | **99,77%** |
| Diabetes | Decision Tree | **77,27%** | 74,52% |
| SOP | Random Forest | **93,58%** | **95,13%** |

Além dos modelos, o projeto entrega uma **API REST (FastAPI)**, um **dashboard frontend** (HTML/Tailwind) com explicabilidade **SHAP** e **Grad-CAM**, **cache de predições**, **tracking com MLflow** e **CI via GitHub Actions**.

> **Nota clínica:** o sistema é uma ferramenta de **suporte à decisão** e não substitui o diagnóstico médico.

---

## 2. Introdução e Contexto

### 2.1 O problema

Doenças como câncer de mama, diabetes tipo 2 e SOP possuem alto impacto na saúde feminina e são mais tratáveis quando detectadas precocemente. A triagem, porém, depende de exames e avaliações clínicas especializadas, nem sempre acessíveis. Modelos de machine learning podem atuar como **segundo par de olhos**, apontando riscos de forma rápida, consistente e explicável.

### 2.2 Objetivos

- Construir a base de um sistema inteligente de apoio à triagem em saúde feminina;
- Aplicar o ciclo completo de ML: **EDA → pré-processamento → modelagem → avaliação → explicabilidade**;
- Comparar múltiplos algoritmos de classificação de forma **reprodutível** (seed fixa, split estratificado);
- Oferecer **explicabilidade** das predições (SHAP, feature importance e Grad-CAM);
- Entregar uma **API** e um **frontend** funcionais;
- Registrar experimentos em **MLflow** e validar o código com **testes automatizados e CI**.

---

## 3. Metodologia

O projeto seguiu um fluxo inspirado no **CRISP-DM**, com as etapas:

1. **Entendimento do negócio/dados** — análise exploratória (EDA) em notebooks Jupyter;
2. **Preparação dos dados** — limpeza, imputação e padronização;
3. **Modelagem** — treino de 6 classificadores por problema tabular + 1 CNN;
4. **Avaliação** — acurácia, AUC-ROC, recall, precisão e F1-score sobre split de teste *held-out* (80/20 estratificado, `random_state=42`);
5. **Implantação** — API FastAPI, dashboard e cache;
6. **Monitoramento/versionamento** — MLflow e GitHub Actions.

---

## 4. Dados e Análise Exploratória

### 4.1 Câncer de Mama — *Breast Cancer Wisconsin (Diagnostic)*

| Atributo | Valor |
|---|---|
| Fonte | UCI Machine Learning Repository |
| Amostras | 569 |
| Features | 30 (média, erro padrão e pior valor de 10 medidas nucleares) |
| Classes | Benigno (B=0): 357 (62,7%) · Maligno (M=1): 212 (37,3%) |
| Valores nulos | Nenhum |
| Split | 455 treino / 114 teste |

As features medem características do núcleo celular (raio, textura, perímetro, área, suavidade, compacidade, concavidade, pontos côncavos, simetria e dimensão fractal). A EDA revela forte separabilidade entre classes, especialmente nas features `worst area`, `worst perimeter` e `mean area`, com alta correlação com o alvo.

### 4.2 Diabetes — *Pima Indians Diabetes Database*

| Atributo | Valor |
|---|---|
| Fonte | UCI / repositório público (jbrownlee) |
| Amostras | 768 |
| Features | 8 (glicose, pressão arterial, IMC, idade, etc.) |
| Classes | Sem diabetes (0): 500 (65,1%) · Com diabetes (1): 268 (34,9%) |
| Valores nulos | Nenhum (defesa `nan_to_num` no loader) |
| Split | 614 treino / 154 teste |

### 4.3 SOP — *Polycystic Ovary Syndrome Dataset*

| Atributo | Valor |
|---|---|
| Fonte | Dataset clínico (workbook `PCOS_data_without_infertility.xlsx`) |
| Amostras | 541 |
| Features | 41 (após remoção de identificadores e colunas não numéricas) |
| Classes | Sem SOP (0): 364 (67,3%) · Com SOP (1): 177 (32,7%) |
| Valores nulos | 4 (imputados com mediana) |
| Split | 432 treino / 109 teste |

### 4.4 Mamografias — *CBIS-DDSM*

| Atributo | Valor |
|---|---|
| Fonte | CBIS-DDSM (Kaggle) |
| Estrutura | `train/`, `val/` e `test/` com subpastas `benign/` e `malignant/` |
| Formato | JPGs convertidos, redimensionados para 224×224 |
| Preparação | Scripts `prepare_cbis_ddsm_dataset` / `prepare_extracted_cbis_ddsm_dataset` |

Os gráficos de EDA gerados (distribuição de classes, correlação, boxplots) estão disponíveis em `outputs/` e nos endpoints `/predict/{cancer,pcos}/eda` e `/predict/image/eda`.

---

## 5. Pré-processamento

- **Padronização:** `StandardScaler` ajustado apenas no treino para câncer e diabetes; para SOP o pré-processamento é encapsulado em um `Pipeline` (`SimpleImputer(median)` + `StandardScaler`).
- **Codificação do alvo:** câncer `M→1 / B→0`; demais datasets já numéricos.
- **Imputação:** SOP usa mediana (4 valores ausentes); diabetes aplica `nan_to_num` como defesa.
- **Imagens:** resize 224×224, normalização ImageNet (mean/std), *data augmentation* (rotação, flip, affine, color jitter) apenas no treino, e **WeightedRandomSampler** para balancear classes.
- **Reprodutibilidade:** splits estratificados com `random_state=42`; modelos com seeds fixas.

---

## 6. Modelagem

### 6.1 Modelos tabulares (câncer, diabetes e SOP)

Seis classificadores foram treinados por problema, com hiperparâmetros fixos:

| Modelo | Hiperparâmetros principais |
|---|---|
| Logistic Regression | `max_iter=1000` |
| Decision Tree | `max_depth=6` |
| Random Forest | `n_estimators=200`, `max_depth=10`, `class_weight="balanced"` |
| Gradient Boosting | `n_estimators=200`, `max_depth=5` |
| SVM | `probability=True` (RBF) |
| KNN | `n_neighbors=5` |

Todos são persistidos em `.joblib` (modelos + scaler/preprocessor + nomes de features) e registrados no **MLflow** (métricas, parâmetros e artefatos de gráficos).

### 6.2 CNN para mamografias

- **Arquitetura:** MobileNetV2 (backbone ImageNet) + cabeçalho customizado (Dropout + Linear + BatchNorm + ReLU) com saída sigmoide binária;
- **Estratégia:** *feature extraction* (backbone congelado) → *fine-tuning* das últimas 20 camadas com LR reduzida;
- **Função de perda:** **Focal Loss** (`alpha=0.75`, `gamma=2.0`) para lidar com desbalanceamento;
- **Otimização:** Adam com `CosineAnnealingLR`, *gradient clipping* (1.0), *mixed precision* (CUDA), *early stopping* (patience 7);
- **Checkpoint:** `models/cnn/mobilenet_mammo.pth` (estado, arquitetura, img_size e métricas de teste).

> **Observação:** o checkpoint atual foi gerado em modo de teste rápido (README orienta `pretrained=False` e poucas épocas para validação), o que explica métricas de teste mais conservadoras. O treino completo com pesos ImageNet é custoso e deve ser executado com GPU.

---

## 7. Resultados

> Métricas calculadas sobre o **split de teste held-out** (nunca visto no treino), com os modelos persistidos em `models/`. CM = matriz de confusão `[[TN, FP], [FN, TP]]`.

### 7.1 Câncer de Mama

| Modelo | Acurácia | AUC-ROC | Recall | Precisão | F1 | CM |
|---|---|---|---|---|---|---|
| Random Forest | **97,37%** | **99,77%** | 92,86% | **100,00%** | **96,30%** | [[72,0],[3,39]] |
| SVM | **97,37%** | 99,47% | 92,86% | **100,00%** | **96,30%** | [[72,0],[3,39]] |
| Logistic Regression | 96,49% | 99,60% | 92,86% | 97,50% | 95,12% | [[71,1],[3,39]] |
| KNN | 95,61% | 98,23% | 90,48% | 97,44% | 93,83% | [[71,1],[4,38]] |
| Gradient Boosting | 92,98% | 98,64% | 88,10% | 92,50% | 90,24% | [[69,3],[5,37]] |
| Decision Tree | 92,11% | 89,48% | 85,71% | 92,31% | 88,89% | [[69,3],[6,36]] |

**Análise:** Random Forest e SVM empatam na melhor acurácia com **precisão de 100%** (nenhum benigno classificado como maligno) e apenas **3 falsos negativos**. O alto recall de ~93% é adequado para triagem, priorizando a captura de casos malignos. Gráficos em `outputs/cancer/comparativo.png` e `outputs/cancer/confusion_matrices.png`.

### 7.2 Diabetes

| Modelo | Acurácia | AUC-ROC | Recall | Precisão | F1 | CM |
|---|---|---|---|---|---|---|
| Decision Tree | **77,27%** | 74,52% | 64,81% | **68,63%** | **66,67%** | [[84,16],[19,35]] |
| SVM | 75,32% | 79,24% | 61,11% | 66,00% | 63,46% | [[83,17],[21,33]] |
| Gradient Boosting | 73,38% | 79,43% | 57,41% | 63,27% | 60,19% | [[82,18],[23,31]] |
| Random Forest | 72,73% | **83,46%** | **70,37%** | 59,38% | 64,41% | [[74,26],[16,38]] |
| Logistic Regression | 71,43% | 82,30% | 51,85% | 60,87% | 56,00% | [[82,18],[26,28]] |
| KNN | 70,13% | 74,05% | 51,85% | 58,33% | 54,90% | [[80,20],[26,28]] |

**Análise:** o diabetes é o problema mais desafiador dos três (dados com sobreposição de classes). A Decision Tree lidera em acurácia/F1, mas o **Random Forest tem o melhor AUC-ROC (83,46%) e o maior recall (70,37%)**, o que o torna a opção mais adequada para triagem (minimizar falsos negativos).

### 7.3 SOP

| Modelo | Acurácia | AUC-ROC | Recall | Precisão | F1 | CM |
|---|---|---|---|---|---|---|
| Random Forest | **93,58%** | **95,13%** | **88,89%** | 91,43% | **90,14%** | [[70,3],[4,32]] |
| Gradient Boosting | 89,91% | 94,94% | 80,56% | 87,88% | 84,06% | [[69,4],[7,29]] |
| KNN | 89,91% | 93,74% | 72,22% | **96,30%** | 82,54% | [[72,1],[10,26]] |
| SVM | 89,91% | 93,82% | 75,00% | 93,10% | 83,08% | [[71,2],[9,27]] |
| Logistic Regression | 88,99% | 95,09% | 86,11% | 81,58% | 83,78% | [[66,7],[5,31]] |
| Decision Tree | 88,99% | 91,38% | 80,56% | 85,29% | 82,86% | [[68,5],[7,29]] |

**Análise:** o **Random Forest domina em praticamente todas as métricas**, com recall de 88,89% e apenas 4 falsos negativos — ótimo resultado para um problema clínico com 41 features e leve desbalanceamento.

### 7.4 Mamografia (CNN)

| Métrica | Valor |
|---|---|
| Acurácia | 56,67% |
| AUC-ROC | 65,92% |
| Recall | 91,43% |
| Precisão | 47,06% |
| F1-Score | 62,14% |

**Análise:** o recall alto indica boa sensibilidade para malignidade, mas a baixa precisão revela muitos falsos positivos — típico de treino rápido/demonstrativo em amostra limitada (ver nota na Seção 6.2). O treino completo com pesos ImageNet, mais épocas e dados balanceados deve elevar acurácia e precisão. O histórico de treino está em `outputs/cnn/training_history.png` e no TensorBoard (`outputs/cnn/tensorboard`).

---

## 8. Explicabilidade

A transparência das predições é tratada em três camadas:

1. **SHAP (câncer e SOP):** a API calcula valores SHAP por predição (`TreeExplainer` para modelos baseados em árvore e `LinearExplainer` para a regressão logística) e devolve gráfico de barras + tabela *top-10 features* — mesmo padrão do notebook de referência.
2. **Feature Importance:** tabelas de importância por feature para todos os modelos que a expõem.
3. **Grad-CAM (mamografia):** mapa de calor sobre a imagem destacando as regiões que mais influenciaram a decisão da CNN (`POST /predict/image/mammography/gradcam`).

Esses artefatos permitem que o profissional entenda **por que** o modelo chegou àquela predição.

---

## 9. Sistema Entregue

### 9.1 Arquitetura

```
frontend (HTML/Tailwind)
        |
        v
FastAPI (src/api) ---> Servicos (cancer, tabular, image, diabetes, pcos)
        |                   |
        |                   |-- Cache TTL (cachetools)
        |                   +-- Modelos (joblib / PyTorch)
        v
MLflow (sqlite:///mlflow.db) . TensorBoard (outputs/cnn/tensorboard)
```

### 9.2 API REST (FastAPI)

Principais grupos de endpoints:

| Grupo | Endpoints |
|---|---|
| Saúde | `GET /health`, `GET /` |
| Câncer | `POST /predict/cancer/` (SHAP), `GET /predict/cancer/{models,info,eda,metrics,samples}` |
| Tabular | `POST /predict/tabular/`, `GET /predict/tabular/models` |
| Mamografia | `POST /predict/image/mammography`, `.../gradcam`, `GET /predict/image/{samples,info,metrics,eda}` |
| Diabetes | `POST /predict/diabetes/`, `GET /predict/diabetes/{models,info}` |
| SOP | `POST /predict/pcos/`, `GET /predict/pcos/{models,info,eda,metrics,samples}` |
| Admin | `GET/DELETE /admin/cache*` |

Documentação interativa em `/docs` (Swagger) e `/redoc`.

### 9.3 Frontend

Dashboard em HTML/Tailwind (`frontend/frontend.html`) com abas para **Câncer, Mamografia, Diabetes, SOP e Cache**. Recursos:
- diagnóstico destacado, probabilidades e confiança;
- gráficos (EDA, comparativo de métricas) e tabelas de explicabilidade;
- galeria de mamografias com **paginação (15 por página)**, filtro por split e **classificação com um clique**;
- visualização de Grad-CAM.

### 9.4 Cache de predições

Cache em memória (`TTLCache`) com chave por hash das features+modelo (ou MD5 de bytes para imagens), métricas de hit/miss e endpoints administrativos para inspeção e limpeza.

### 9.5 Rastreabilidade

- **MLflow:** experimentos `cancer_prediction`, `diabetes_prediction`, `mammo_cnn` (e SOP) com métricas, parâmetros, artefatos e modelos registrados;
- **TensorBoard:** gráficos de loss/AUC por época da CNN.

---

## 10. Testes e Validação

| Camada | Ferramenta | Cobertura |
|---|---|---|
| Testes unitários/API | `pytest` (`tests/`) | API, serviços de câncer, diabetes, SOP e CNN |
| Frontend | `python test_frontend.py` | UTF-8, estilos, renderização, predições, gráficos, paleta |
| JS inline | `scripts/validate_frontend_js.py` | Sintaxe do JavaScript do frontend |
| Galeria | `node scripts/validate_gallery_logic.js` | Lógica de paginação do frontend |
| Treino | `scripts/validate_training.py`, `scripts/validate_cnn_training.py` | Pipelines de treino (rápidos) |
| CI | GitHub Actions (`.github/workflows/ci.yml`) | Instala deps, treina modelos dummy + reais e roda `pytest` |

---

## 11. Como Reproduzir

```bash
# 1. Dependencias
pip install -r requirements.txt

# 2. Treinar modelos tabulares
python -c "from src.cancer.train import train_cancer_models; train_cancer_models()"
python -c "from src.diabetes.train import train_diabetes_models; train_diabetes_models()"
python -c "from src.pcos.train import train_pcos_models; train_pcos_models()"

# 3. API
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# 4. Frontend / Docs
# http://127.0.0.1:8000/static/frontend.html  .  http://127.0.0.1:8000/docs

# 5. Testes
python -m pytest tests/ -v
python test_frontend.py

# 6. Docker (opcional)
docker-compose up --build
```

---

## 12. Conclusões

A Fase 1 cumpriu os objetivos de construir uma base sólida de um sistema inteligente para saúde feminina:

- **Modelos tabulares com excelente desempenho** em câncer (97,4% acurácia, 99,8% AUC) e SOP (93,6% / 95,1%), com o Random Forest como melhor candidato geral;
- **Fluxo completo e reprodutível** de ML com explicabilidade (SHAP, importance e Grad-CAM);
- **Aplicação funcional** (API + dashboard + cache + MLflow + CI) pronta para evolução;
- A frente de **mamografia (CNN)** está operacional como prova de conceito, com espaço claro de melhoria via treino completo e mais dados.

**Recomendações:** para produção, treinar a CNN completa com GPU e validação externa, refinar hiperparâmetros do diabetes (ex.: SMOTE ou otimização de limiar), e submeter os modelos a avaliação clínica e regulatória.

---

## 13. Trabalhos Futuros (Fase 2)

- Deploy da API em nuvem (Railway, Render, AWS) com Docker;
- Autenticação e autorização (RBAC para profissionais de saúde);
- Refinamento da CNN (treino completo, mais épocas, validação cruzada);
- Otimização de hiperparâmetros (Optuna) e busca de limiar ótimo;
- Monitoramento de drift e re-treino agendado;
- Novos datasets e integração com prontuário eletrônico.

---

## 14. Referências

- Breast Cancer Wisconsin (Diagnostic) — UCI ML Repository: `archive.ics.uci.edu`
- Pima Indians Diabetes Database — UCI/Kaggle
- CBIS-DDSM (Curated Breast Imaging Subset of DDSM) — Kaggle
- SOP Dataset — Polycystic Ovary Syndrome (workbook clínico)
- Lundberg, S. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions* (SHAP)
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks*
- Sandler, M. et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*
- Lin, T.-Y. et al. (2017). *Focal Loss for Dense Object Detection*
