from collections import OrderedDict

from sklearn.metrics import (
    accuracy_score, auc, cohen_kappa_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_curve
)


def calculate_classification_metrics(true_values, predicted_values):
    fpr_tr, tpr_tr, thresholds_tr = roc_curve(true_values, predicted_values)

    return OrderedDict(
        accuracy=accuracy_score(true_values, predicted_values),
        f1_score=f1_score(true_values, predicted_values),
        roc_auc=auc(fpr_tr, tpr_tr),
        cohen_cappa=cohen_kappa_score(true_values, predicted_values),
        matthews_correlation=matthews_corrcoef(true_values, predicted_values),
        precision_score=precision_score(true_values, predicted_values),
        recall_score=recall_score(true_values, predicted_values)
    )