"""Logger centralizado do MLflow."""
import logging
import os

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_URI)


class ExperimentLogger:
    """Wrapper para registrar experimentos no MLflow.

    O logging é sempre opcional: se o servidor/banco do MLflow estiver
    indisponível ou com schema desatualizado, o treinamento continua
    normalmente (apenas emite um aviso). Isso garante que os pipelines de
    treino dos notebooks nunca falhem por causa de tracking.
    """

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self._available = True
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as error:  # noqa: BLE001 - tracking é opcional
            self._available = False
            logger.warning(
                "MLflow indisponível para '%s' (seguindo sem tracking): %s",
                experiment_name, error,
            )

    def log_sklearn_model(self, model, metrics: dict, params: dict,
                          artifacts: list = None, model_name: str = "sklearn_model"):
        """Registra modelo scikit-learn + metricas + parametros (tracking opcional)."""
        if not self._available:
            return
        try:
            with mlflow.start_run(run_name=model_name):
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

                if artifacts:
                    for art in artifacts:
                        if os.path.exists(art):
                            mlflow.log_artifact(art)

                try:
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        registered_model_name=f"{self.experiment_name}_{model_name}"
                    )
                    print(f"Run registrada: {mlflow.active_run().info.run_id}")
                except Exception as e:
                    # Fallback para modelos que não conseguem ser serializados
                    # (ex: KNeighborsClassifier com KDTree não-confiável)
                    print(f"⚠️  Aviso ao registrar {model_name}: {str(e)}")
                    print(f"   Continuando... Modelo salvo como .joblib, MLflow com metadados apenas.")
        except Exception as error:  # noqa: BLE001 - tracking é opcional
            logger.warning(
                "MLflow falhou ao registrar '%s' (seguindo sem tracking): %s",
                model_name, error,
            )

    def log_pytorch_model(self, model, metrics: dict, params: dict,
                          artifacts: list = None, model_name: str = "pytorch_model"):
        """Registra modelo PyTorch + metricas + parametros (tracking opcional)."""
        if not self._available:
            return
        try:
            with mlflow.start_run(run_name=model_name):
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

                if artifacts:
                    for art in artifacts:
                        if os.path.exists(art):
                            mlflow.log_artifact(art)

                try:
                    mlflow.pytorch.log_model(
                        pytorch_model=model,
                        artifact_path="model",
                        registered_model_name=f"{self.experiment_name}_{model_name}"
                    )
                    print(f"Run PyTorch registrada: {mlflow.active_run().info.run_id}")
                except MlflowException as error:
                    print(f"MLflow não registrou o modelo PyTorch: {error}")
        except Exception as error:  # noqa: BLE001 - tracking é opcional
            logger.warning(
                "MLflow falhou ao registrar '%s' (seguindo sem tracking): %s",
                model_name, error,
            )

    @staticmethod
    def get_best_run(experiment_name: str, metric: str = "test_auc"):
        """Recupera a melhor run de um experimento por metrica."""
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return None
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1
        )
        return runs[0] if runs else None
