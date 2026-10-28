"""Perdidas con reduccion media exclusivamente sobre etiquetas conocidas."""
import math
import torch
from torch.nn import functional as F
from .training_engine import supervision


def masked_bce(logits, targets, label_mask):
    safe = supervision(logits, targets, label_mask)
    return F.binary_cross_entropy_with_logits(logits[label_mask], safe[label_mask])


def masked_focal(logits, targets, label_mask, gamma=2.0, alpha=None):
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not math.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma debe ser finito y no negativo.")
    if alpha is not None and (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
                              or not math.isfinite(alpha) or not 0 <= alpha <= 1):
        raise ValueError("alpha debe ser None o un numero entre cero y uno.")
    safe = supervision(logits, targets, label_mask)
    x, y = logits[label_mask], safe[label_mask]
    bce = F.binary_cross_entropy_with_logits(x, y, reduction="none")
    # log(p_t)=-BCE. expm1 evita perder precision cerca de p_t=1.
    one_minus_pt = -torch.expm1(-bce)
    loss = one_minus_pt.pow(gamma)*bce
    if alpha is not None:
        loss = loss*torch.where(y == 1, alpha, 1-alpha)
    return loss.mean()
