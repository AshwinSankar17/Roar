from torchmetrics import (
    Accuracy,
    AveragePrecision,
    F1Score,
    MatthewsCorrCoef,
    PearsonCorrCoef,
    SpearmanCorrCoef,
)
from torchmetrics.text.rouge import ROUGEScore

from roar.collections.common.metrics.classification_accuracy import (
    ExactStringMatchMetric,
    TokenF1Score,
)

__all__ = ["MetricStringToTorchMetric"]

# Dictionary that maps a metric string name to its corresponding torchmetric class.

MetricStringToTorchMetric = {
    "accuracy": Accuracy,
    "average_precision": AveragePrecision,
    "f1": F1Score,
    "token_f1": TokenF1Score,
    "pearson_corr_coef": PearsonCorrCoef,
    "spearman_corr_coef": SpearmanCorrCoef,
    "matthews_corr_coef": MatthewsCorrCoef,
    "exact_string_match": ExactStringMatchMetric,
    "rouge": ROUGEScore,
}
