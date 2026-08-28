"""Validação do treinamento de SOP (mesmo fluxo do notebook 07_modelagem_sop)."""
import sys
import time

sys.path.insert(0, ".")

from src.pcos.train import train_pcos_models

if __name__ == "__main__":
    t0 = time.time()
    models, results, preprocessor, feature_names = train_pcos_models()
    print("Modelos treinados:", list(models.keys()))
    print("Features:", len(feature_names))
    for name, metrics in results.items():
        print(
            f"  {name}: acc={metrics['accuracy']:.3f} auc={metrics['auc']:.3f} "
            f"f1={metrics['f1']:.3f}"
        )
    print(f"Treino SOP OK em {round(time.time() - t0, 1)}s")
