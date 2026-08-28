"""Modelos ML para predição de câncer de mama."""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def create_cancer_models(random_state: int = 42) -> dict:
    """Retorna dicionario de modelos para câncer de mama."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=random_state, max_depth=6
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=10,
            random_state=random_state, class_weight="balanced",
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5,
            random_state=random_state,
        ),
        "svm": SVC(probability=True, random_state=random_state),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }
