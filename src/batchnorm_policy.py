"""Politica BN por estudio v1. No cambia entrenamiento ni buffers persistentes."""
from contextlib import contextmanager
import torch
from torch import nn

POLICY_ID = "BN_por_estudio_v1"


def validate_policy(policy):
    if policy not in (None, POLICY_ID):
        raise ValueError("Politica de evaluacion desconocida.")


@contextmanager
def per_study_batchnorm(model, images, slice_mask):
    """Un unico estudio por forward; BN usa sus cortes validos sin acumularlos.

    El llamador debe haber usado model.eval() e inference_mode/no_grad.
    Se restauran los flags incluso si el forward falla. Dropout sigue en eval.
    """
    if model.training or torch.is_grad_enabled():
        raise ValueError("Requiere model.eval() y gradientes desactivados.")
    if (images.ndim != 5 or images.shape[0] != 1 or images.shape[1] != 8
            or images.shape[2] != 1 or tuple(slice_mask.shape) != (1, 8)):
        raise ValueError("Se requiere exactamente un estudio [1,8,1,H,W].")
    if slice_mask.dtype != torch.bool or not bool(slice_mask.any()):
        raise ValueError("Se requiere mascara booleana con cortes validos.")
    layers = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not layers or any(not isinstance(m, nn.BatchNorm2d) for m in layers):
        raise ValueError("Esta politica requiere un modelo con BatchNorm2d.")
    if any(m.training for m in model.modules()):
        raise ValueError("Todas las capas deben entrar inicialmente en eval.")
    previous = [(m, m.training, m.track_running_stats) for m in layers]
    try:
        for m, _, _ in previous:
            m.track_running_stats = False
            m.training = True
        yield
    finally:
        for m, training, tracking in previous:
            m.training = training
            m.track_running_stats = tracking


def evaluation_forward(model, images, slice_mask, policy=None):
    validate_policy(policy)
    if policy is None:
        return model(images, slice_mask)
    with per_study_batchnorm(model, images, slice_mask):
        return model(images, slice_mask)
