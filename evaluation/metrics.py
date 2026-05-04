from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.stats import chi2


def precision_recall_f1(y_true: List[str], y_pred: List[str], positive_class: str = "contradiction") -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive_class and p == positive_class)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive_class and p == positive_class)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive_class and p != positive_class)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    classes = sorted(set(y_true) | set(y_pred))
    f1s = []
    for cls in classes:
        metrics = precision_recall_f1(y_true, y_pred, positive_class=cls)
        f1s.append(metrics["f1"])
    return float(np.mean(f1s)) if f1s else 0.0


def bootstrap_ci(
    y_true: List[str],
    y_pred: List[str],
    n_iterations: int = 1000,
    metric_fn: Callable[[List[str], List[str]], float] = None,
) -> Tuple[float, float, float]:
    if metric_fn is None:
        metric_fn = lambda yt, yp: precision_recall_f1(yt, yp)["f1"]

    scores = []
    indices = np.arange(len(y_true))
    for _ in range(n_iterations):
        sample = np.random.choice(indices, size=len(indices), replace=True)
        yt = [y_true[i] for i in sample]
        yp = [y_pred[i] for i in sample]
        scores.append(metric_fn(yt, yp))

    scores = np.array(scores)
    return float(scores.mean()), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def mcnemar_test(y_true: List[str], pred_a: List[str], pred_b: List[str]) -> float:
    b = sum(1 for t, a, b in zip(y_true, pred_a, pred_b) if a == t and b != t)
    c = sum(1 for t, a, b in zip(y_true, pred_a, pred_b) if a != t and b == t)
    if b + c == 0:
        return 1.0
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    return float(1 - chi2.cdf(chi2_stat, 1))
