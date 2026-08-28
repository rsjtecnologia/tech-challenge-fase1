# Tech Challenge Fase 1 — Health AI

Projeto em Python para apoio ao diagnóstico e análise de risco em saúde feminina, com:
- classificação tabular para câncer de mama
- explicabilidade com SHAP
- extra opcional com mamografia via CNN + Grad-CAM
- modelo de diabetes
- frontend HTML/Tailwind
- API FastAPI

---

## 1. Objetivo da Fase 1

Atender ao desafio de criar a base de um sistema inteligente para apoiar a triagem e a identificação precoce de riscos, usando:
- Machine Learning
- análise exploratória
- pré-processamento
- avaliação com métricas
- explicabilidade
- visão computacional opcional

Dataset principal:
- `data/breast_cancer_wisconsin.csv`

Extras do projeto:
- mamografia com CNN (CBIS-DDSM + Grad-CAM)
- diabetes com múltiplos modelos
- SOP com múltiplos modelos

---

## 2. O que este projeto entrega

- API REST com FastAPI
- frontend visual com dashboard
- predição tabular para câncer de mama
- comparação de modelos
- explicabilidade com SHAP e feature importance
- predição de diabetes
- predição de SOP
- predição de mamografia com Grad-CAM
- galeria de mamografias com paginação no frontend
- cache de predições com endpoints administrativos
- CI via GitHub Actions
- documentação de apoio em Markdown

---

## 3. Como rodar a aplicação

> Este guia traz os comandos para os três terminais mais comuns:
> - **bash** — Linux, macOS ou Git Bash no Windows
> - **CMD** — Prompt de Comando do Windows
> - **PowerShell** — Windows PowerShell / PowerShell 7
>
> Escolha o bloco correspondente ao seu terminal. Quando o comando é
> idêntico nos três, isso é indicado no texto.

### 3.1 Pré-requisitos

- Python 3.10 ou superior
- pip
- Git
- opcional: Docker e Docker Compose

> **Windows:** ao instalar o Python pelo [python.org](https://www.python.org),
> marque a opção **"Add python.exe to PATH"** para usar `python` e `pip`
> diretamente no CMD e no PowerShell.

### 3.2 Instalar dependências

Comando idêntico em todos os terminais:

```bash
pip install -r requirements.txt
```

#### 3.2.1 (Recomendado) Ambiente virtual

**bash (Git Bash / Linux / macOS)**

```bash
python -m venv .venv
source .venv/Scripts/activate
```

**CMD**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> Se o PowerShell bloquear a ativação do script, libere a execução de
> scripts uma vez com:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### 3.3 Rodar o backend

Comando idêntico em todos os terminais:

```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

### 3.4 Abrir o frontend

No navegador:

```text
http://127.0.0.1:8000/static/frontend.html
```

### 3.5 Abrir a documentação da API

```text
http://127.0.0.1:8000/docs
```

---

## 4. Rodar com Docker

```bash
docker-compose up --build
```

> Comando idêntico no CMD e no PowerShell.

Depois abra:

```text
http://127.0.0.1:8000/static/frontend.html
```

---

## 5. Como rodar os notebooks

Todos os notebooks seguem o mesmo padrão do câncer de mama (EDA completo +
modelagem completa, usando os módulos `src/`):

```text
01_eda_tabular.ipynb           -> EDA do câncer de mama
02_modelagem_tabular.ipynb     -> Modelagem do câncer de mama
03_eda_imagens.ipynb           -> EDA das mamografias CBIS-DDSM
04_cnn_mamografias.ipynb       -> CNN: amostras, predição e Grad-CAM
05_eda_diabetes.ipynb          -> EDA do diabetes (Pima)
06_modelagem_diabetes.ipynb    -> Modelagem do diabetes (com SHAP)
06b_eda_sop.ipynb              -> EDA do SOP
07_modelagem_sop.ipynb         -> Modelagem do SOP (com SHAP)
08_preparacao_mamografias.ipynb-> Preparação das mamografias
09_treino_teste_mamografias.ipynb -> Treino/teste da CNN
10_FIAP_Tech_Challenge_Fase1.ipynb -> Notebook legado autônomo (câncer)
```

Os notebooks de modelagem `06_modelagem_diabetes` e `07_modelagem_sop`
incluem, além de treino/métricas/feature importance, uma análise de
explicabilidade **SHAP** (summary plot usando a `logistic_regression` como
modelo de referência, no mesmo padrão do notebook FIAP).

### 5.1 Abrir Jupyter

```bash
jupyter notebook
```

ou

```bash
jupyter lab
```

> Os comandos acima são idênticos no CMD e no PowerShell.

### 5.2 O que o notebook faz

- carrega a base
- faz EDA
- trata dados
- treina múltiplos modelos
- avalia métricas
- gera explicabilidade
- exporta artefatos

### 5.3 Treinar os modelos fora do notebook

Os mesmos pipelines dos notebooks podem ser executados via módulos Python:

**bash e CMD** — os mesmos comandos funcionam nos dois terminais:

```bash
python -c "from src.cancer.train import train_cancer_models; train_cancer_models()"
python -c "from src.diabetes.train import train_diabetes_models; train_diabetes_models()"
python -c "from src.pcos.train import train_pcos_models; train_pcos_models()"
```

**PowerShell** — use **aspas simples** para o código Python:

```powershell
python -c 'from src.cancer.train import train_cancer_models; train_cancer_models()'
python -c 'from src.diabetes.train import train_diabetes_models; train_diabetes_models()'
python -c 'from src.pcos.train import train_pcos_models; train_pcos_models()'
```

A CNN (mamografia) é treinada pelo notebook `09_treino_teste_mamografias.ipynb`
(chama `train_cnn()`), que prepara os splits CBIS-DDSM, treina com feature
extraction + fine-tuning, salva `models/cnn/mobilenet_mammo.pth` e registra no
MLflow. O treino completo com pesos ImageNet é pesado; para um teste rápido,
use `pretrained=False` e poucas épocas.

### 5.4 Acompanhar experimentos no MLflow

O MLflow usa a base `mlflow.db` do projeto (ou o servidor Docker).
Defina a variável de ambiente e abra a UI em `http://127.0.0.1:5000`:

**bash**

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
mlflow ui
```

**CMD**

```cmd
set MLFLOW_TRACKING_URI=sqlite:///mlflow.db
mlflow ui
```

**PowerShell**

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow ui
```

---

## 6. Estrutura do projeto

```text
src/
  api/        -> FastAPI, rotas, schemas e serviços
  cancer/     -> treino e predição tabular (câncer de mama)
  diabetes/   -> treino e predição de diabetes
  pcos/       -> treino e predição de SOP
  cnn/        -> visão computacional para mamografias
  tracking/   -> logging/MLflow
frontend/     -> interface HTML (dashboard)
notebooks/    -> notebooks Jupyter
data/         -> datasets (cancer, diabetes, images, polycystic)
models/       -> modelos treinados (cancer, cnn, diabetes, legacy, pcos, tabular)
outputs/      -> artefatos de treino (tensorboard, gráficos)
scripts/      -> scripts de validação
reports/      -> relatório técnico
tests/        -> testes automatizados (pytest)
.github/      -> CI (GitHub Actions)
```

---

## 7. Datasets usados

### Principal
- Breast Cancer Wisconsin
- arquivo local: `data/breast_cancer_wisconsin.csv`
- cópia adicional: `data/cancer/breast_cancer.csv`

### Complementares
- Pima Indians Diabetes: `data/diabetes/pima-indians-diabetes.csv`
- CBIS-DDSM (mamografias): `data/images/Breast_cancer_Image/` (splits preparados em `data/images/cbis-ddsm/`)
- SOP (Síndrome dos Ovários Policísticos): `data/polycystic/polycystic_ovary_syndrome/`
  (`PCOS_data_without_infertility.xlsx` + `PCOS_infertility.csv`)

---

## 8. Endpoints principais

### Raiz
- `GET /` — links de documentação e versão da API

### Saúde
- `GET /health` — status da API e modelos carregados (+ estatísticas de cache)

### Câncer tabular
- `POST /predict/tabular/`
- `GET /predict/tabular/models`

### Câncer com gráficos/SHAP
- `POST /predict/cancer/`
- `GET /predict/cancer/models`
- `GET /predict/cancer/samples`
- `GET /predict/cancer/info`
- `GET /predict/cancer/eda`
- `GET /predict/cancer/metrics`

### Mamografia
- `POST /predict/image/mammography`
- `POST /predict/image/mammography/gradcam`
- `GET /predict/image/samples`
- `GET /predict/image/samples/{sample_id}`
- `POST /predict/image/mammography/sample/{sample_id}`
- `GET /predict/image/info`
- `GET /predict/image/metrics`
- `GET /predict/image/eda`

### Diabetes
- `POST /predict/diabetes/`
- `GET /predict/diabetes/models`
- `GET /predict/diabetes/info`

### SOP (Síndrome dos Ovários Policísticos)
- `POST /predict/pcos/`
- `GET /predict/pcos/models`
- `GET /predict/pcos/info`
- `GET /predict/pcos/samples`
- `GET /predict/pcos/eda`
- `GET /predict/pcos/metrics`

### Administração (cache)
- `GET /admin/cache` — status detalhado de todos os caches
- `DELETE /admin/cache` — limpa todos os caches
- `DELETE /admin/cache/{service}` — limpa o cache de um serviço

---

## 9. Como usar o frontend

Abra o frontend e escolha uma aba:
- 🩺 Câncer
- 🖼️ Mamografia
- 💉 Diabetes
- 🌸 SOP
- ⚙️ Cache

Cada resultado mostra:
- diagnóstico destacado
- métricas
- gráfico
- tabela/explicabilidade

A aba **Mamografia** inclui uma galeria com miniaturas das imagens do
dataset local (`data/images/cbis-ddsm`), com **5 miniaturas por linha e 3
linhas por página (15 imagens)**, paginação ‹ ›, filtro por split
(Todos/Teste/Validação/Treino) e classificação com um clique, além dos botões
**Ver EDA** e **Métricas** (como na aba de Câncer).

---

## 10. Testes e validação

### Frontend

```bash
python test_frontend.py
```

> Comando idêntico em bash, CMD e PowerShell.

Esse script valida:
- UTF-8
- estilos
- funções de renderização
- predições
- gráficos
- paleta de cores

### Scripts de validação (`scripts/`)

```bash
python scripts/validate_frontend_js.py
python scripts/validate_training.py
python scripts/validate_cnn_training.py
node scripts/validate_gallery_logic.js
```

> ⚠️ **CMD:** evite comentários `# ...` no fim das linhas — o CMD os
> interpreta como argumentos extras e o comando falha. Os scripts acima
> devem ser executados **sem comentários** e funcionam da mesma forma em
> bash, CMD e PowerShell.

### Testes automatizados (pytest)

```bash
python -m pytest tests/ -v
```

> Comando idêntico em CMD e PowerShell.

---

## 11. Modelos

### Câncer (módulo `src/cancer`)
- logistic_regression
- decision_tree
- random_forest
- gradient_boosting
- svm
- knn

### Câncer tabular (endpoint `/predict/tabular/`)
- os mesmos 6 modelos acima (salvos em `models/tabular/`)
- `legacy_logistic_regression`: modelo legado (.pkl) migrado do projeto original
  (carregado de `models/legacy/`)

### Diabetes
- logistic_regression
- decision_tree
- random_forest
- gradient_boosting
- svm
- knn

### SOP
- logistic_regression
- decision_tree
- random_forest
- gradient_boosting
- svm
- knn

### Mamografia (CNN)
- MobileNetV2 (PyTorch) com fine-tuning em CBIS-DDSM
- Focal Loss + WeightedRandomSampler para classes desbalanceadas

---

## 12. Relatório e entrega

Para a entrega da Fase 1, inclua:
- código-fonte
- este README
- Dockerfile, se usado
- dataset ou link
- prints, gráficos e análises
- relatório técnico
- vídeo de demonstração

Sugestão de relatório técnico:
- `reports/relatorio_tecnico.md`

---

## 13. Passo a passo resumido

1. Instale dependências
2. Rode a API
3. Abra o frontend
4. Execute o notebook
5. Valide com `test_frontend.py`
6. Gere os resultados para o relatório PDF

---

## 14. Links úteis

- Frontend: `http://127.0.0.1:8000/static/frontend.html`
- Swagger: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- Health: `http://127.0.0.1:8000/health`
- MLflow UI (via Docker): `http://127.0.0.1:5000`

---

## 15. Observações

- O médico sempre deve ter a palavra final no diagnóstico.
- O modelo é suporte à decisão, não substituto clínico.
- Se o frontend mostrar dados estranhos, faça reload completo do navegador.

---

## 16. Estado atual

- Frontend com dashboard visual
- Diagnóstico destacado
- Gráficos maiores
- UTF-8 corrigido
- Galeria de mamografias com paginação, filtro por split e classificação com um clique
- Cache de predições com endpoints administrativos (`/admin/cache`)
- CI com GitHub Actions (`.github/workflows/ci.yml`)
- Testes automatizados e scripts de validação OK

---

## 17. Roadmap / Fase 2 (sugestões)

- Deploy da API em nuvem (ex.: Railway, Render, AWS)
- Autenticação e autorização
- Novos datasets e modelos
- Monitoramento dos modelos em produção

