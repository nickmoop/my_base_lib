from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy


DECISION_TREE_CLASSIFIER = 'Decision Tree Classifier'
RANDOM_FOREST_CLASSIFIER = 'Random Forest Classifier'
ADA_BOOST_CLASSIFIER = 'Ada Boost Classsifier'
EXTRA_TREES_CLASSIFIER = 'Extra Trees classifier'
GRADIENT_BOOSTING_CLASSIFIER = 'Gradient Boosting Classifier'
SVM_CLASSIFIER = 'Support Vector Machine Classifier'
K_NEIGHBORS_CLASSIFIER = 'Nearest Neighbors Classifier'
LINEAR_DISCRIMINANT_ANALYSIS = 'Linear Discriminant Analysis'
LOGISTIC_REGRESSION = 'Logistic Regression'
MLP_CLASSIFIER = 'MLP CLASSIFIER'

MODELS_METHODS = {
    DECISION_TREE_CLASSIFIER: DecisionTreeClassifier,
    RANDOM_FOREST_CLASSIFIER: RandomForestClassifier,
    ADA_BOOST_CLASSIFIER: AdaBoostClassifier,
    EXTRA_TREES_CLASSIFIER: ExtraTreesClassifier,
    GRADIENT_BOOSTING_CLASSIFIER: GradientBoostingClassifier,
    SVM_CLASSIFIER: SVC,
    K_NEIGHBORS_CLASSIFIER: KNeighborsClassifier,
    LINEAR_DISCRIMINANT_ANALYSIS: LinearDiscriminantAnalysis,
    LOGISTIC_REGRESSION: LogisticRegression,
    MLP_CLASSIFIER: MLPClassifier
}

MODELS_HYPERPARAMETERS_GRID = {
    DECISION_TREE_CLASSIFIER: {
        'max_leaf_nodes': [5, 50, 100, 500, 200, 5000],
        'min_samples_leaf': [0.1, 0.5, 5],
        'min_samples_split': [0.1, 1.0, 10],
        'max_depth': list(range(1, 33))

    },
    RANDOM_FOREST_CLASSIFIER: {
        'max_depth': [None],
        'max_features': [1, 3, 10],
        'min_samples_split': [2, 3, 10],
        'min_samples_leaf': [1, 3, 10],
        'bootstrap': [False],
        'n_estimators': [100, 300],
        'criterion': ['gini']
    },
    ADA_BOOST_CLASSIFIER: {
        'base_estimator__criterion': ['gini', 'entropy'],
        'base_estimator__splitter':   ['best', 'random'],
        'algorithm': ['SAMME', 'SAMME.R'],
        'n_estimators': [1, 2],
        'learning_rate':  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5]
    },
    EXTRA_TREES_CLASSIFIER: {
        'max_depth': [None],
        'max_features': [1, 3, 10],
        'min_samples_split': [2, 3, 10],
        'min_samples_leaf': [1, 3, 10],
        'bootstrap': [False],
        'n_estimators': [100, 300],
        'criterion': ['gini']
    },
    GRADIENT_BOOSTING_CLASSIFIER: {
        'loss': ['deviance'],
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': [4, 8],
        'min_samples_leaf': [100, 150],
        'max_features': [0.3, 0.1]
    },
    SVM_CLASSIFIER: {
        'kernel': ['rbf'],
        'gamma': [0.001, 0.01, 0.1, 1],
        'C': [1, 10, 50, 100, 200, 300, 1000]
    },
    K_NEIGHBORS_CLASSIFIER: {
        'n_neighbors': list(range(1, 30)),
        'p': [1, 2, 3, 4, 5]
    },
    LOGISTIC_REGRESSION: {
        'penalty': ['l1', 'l2'],
        'C': numpy.logspace(0, 4, 10)
    },
    MLP_CLASSIFIER: {
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam']

    },
    LINEAR_DISCRIMINANT_ANALYSIS: {
        'solver': ['svd', 'lsqr']
    },
}

MODELS_ADDITIONAL_PARAMETERS = {
    DECISION_TREE_CLASSIFIER: dict(),
    RANDOM_FOREST_CLASSIFIER: dict(),
    ADA_BOOST_CLASSIFIER: {'base_estimator': DecisionTreeClassifier()},
    EXTRA_TREES_CLASSIFIER: dict(),
    GRADIENT_BOOSTING_CLASSIFIER: dict(),
    SVM_CLASSIFIER: {'probability': True},
    K_NEIGHBORS_CLASSIFIER: dict(),
    LINEAR_DISCRIMINANT_ANALYSIS: dict(),
    LOGISTIC_REGRESSION: dict(),
    MLP_CLASSIFIER: dict()
}
