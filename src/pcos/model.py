"""Modelos de machine learning para classificação de SOP."""
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def create_pcos_models(random_state: int = 42) -> dict:
    """Cria os seis classificadores comparados no projeto."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "decision_tree": DecisionTreeClassifier(max_depth=6, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced", random_state=random_state
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, random_state=random_state
        ),
        "svm": SVC(probability=True, random_state=random_state),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }
