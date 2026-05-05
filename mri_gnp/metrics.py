from typing import List

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def ordinal_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    targets = torch.zeros((labels.shape[0], num_classes - 1), dtype=torch.float32, device=labels.device)
    for threshold_id in range(num_classes - 1):
        targets[:, threshold_id] = (labels > threshold_id).to(torch.float32)
    return targets


def ordinal_pred_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(logits) > 0.5).to(torch.int64).sum(dim=1)


def ordinal_probabilities_tensor(logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    scores = torch.sigmoid(logits.float())
    if num_classes == 2:
        return torch.stack([1.0 - scores[:, 0], scores[:, 0]], dim=1)
    probs = torch.zeros((scores.shape[0], num_classes), dtype=torch.float32, device=logits.device)
    cumulative = torch.ones((scores.shape[0],), dtype=torch.float32, device=logits.device)
    for class_id in range(num_classes - 1):
        score = scores[:, class_id]
        if class_id == 0:
            probs[:, 0] = 1.0 - score
            cumulative = score
        else:
            probs[:, class_id] = cumulative * (1.0 - score)
            cumulative = cumulative * score
    probs[:, num_classes - 1] = cumulative
    return probs / torch.clamp(probs.sum(dim=1, keepdim=True), min=1e-8)


def finite_row_mask(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim <= 1:
        return torch.isfinite(tensor)
    finite = torch.isfinite(tensor)
    return finite.reshape(finite.shape[0], -1).all(dim=1)


def _compute_auc_from_probabilities(y_true: List[int], y_probs: np.ndarray, num_classes: int):
    if len(y_true) == 0:
        return None, None
    y_true_np = np.asarray(y_true, dtype="int64")
    try:
        if num_classes == 2:
            if len(np.unique(y_true_np)) > 1:
                return float(roc_auc_score(y_true_np, y_probs[:, 1])), None
            return 0.0, None
        auc_per_class = []
        for class_id in range(num_classes):
            y_true_binary = (y_true_np == class_id).astype(int)
            if len(np.unique(y_true_binary)) > 1:
                try:
                    auc_per_class.append(float(roc_auc_score(y_true_binary, y_probs[:, class_id])))
                except Exception:
                    auc_per_class.append(0.0)
            else:
                auc_per_class.append(0.0)
        valid = [score for score in auc_per_class if score > 0]
        auc_value = float(np.mean(valid)) if valid else 0.0
        return auc_value, auc_per_class
    except Exception:
        return None, None


def confusion_and_basic_metrics(y_true: List[int], y_pred: List[int], num_classes: int):
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true_value, pred_value in zip(y_true, y_pred):
        confusion[int(true_value)][int(pred_value)] += 1
    recalls = []
    correct = 0
    for class_id in range(num_classes):
        row_sum = sum(confusion[class_id])
        correct += confusion[class_id][class_id]
        recalls.append((confusion[class_id][class_id] / row_sum) if row_sum > 0 else 0.0)
    accuracy = correct / max(1, len(y_true))
    balanced_accuracy = float(sum(recalls) / max(1, len(recalls)))
    return confusion, accuracy, balanced_accuracy


def compute_classification_metrics(total_loss, total_count, y_true, y_pred, num_classes, y_probs=None):
    confusion, accuracy, balanced_accuracy = confusion_and_basic_metrics(y_true, y_pred, num_classes)
    precision = recall = f1 = None
    precision_per_class = recall_per_class = f1_per_class = None
    specificity = specificity_per_class = None
    auc = auc_per_class = None

    if y_true and y_pred:
        precision_raw, recall_raw, f1_raw, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            zero_division=0,
            average=None,
        )
        precision_per_class = [float(item) for item in precision_raw]
        recall_per_class = [float(item) for item in recall_raw]
        f1_per_class = [float(item) for item in f1_raw]
        precision = float(np.mean(precision_per_class))
        recall = float(np.mean(recall_per_class))
        f1 = float(np.mean(f1_per_class))

        specificity_per_class = []
        for class_id in range(num_classes):
            tn = sum(
                confusion[i][j]
                for i in range(num_classes)
                if i != class_id
                for j in range(num_classes)
                if j != class_id
            )
            fp = sum(confusion[i][class_id] for i in range(num_classes) if i != class_id)
            denom = tn + fp
            specificity_per_class.append(float(tn / denom) if denom > 0 else 0.0)
        specificity = float(np.mean(specificity_per_class))

    if y_probs is not None:
        auc, auc_per_class = _compute_auc_from_probabilities(y_true, y_probs, num_classes)

    return {
        "loss": float(total_loss) / max(1, int(total_count)),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "auc": auc,
        "auc_per_class": auc_per_class,
        "precision": precision,
        "precision_per_class": precision_per_class,
        "recall": recall,
        "recall_per_class": recall_per_class,
        "f1": f1,
        "f1_per_class": f1_per_class,
        "specificity": specificity,
        "specificity_per_class": specificity_per_class,
        "confusion": confusion,
        "num_samples": int(total_count),
    }
