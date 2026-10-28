"""Grad-CAM del logit por estudio; no cambia el modulo ni el motor BN comun."""
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from .batchnorm_policy import POLICY_ID, evaluation_forward

VERSION = 'GradCAM_estudio_BNv1_v1'


@contextmanager
def explanation_forward(model, images, slice_mask, policy):
    """Misma modalidad numerica BN con derivadas; parametros deben estar congelados.

    Contexto exclusivo de explicacion. La politica comun exige no_grad y NO se
    modifica ni se evade: se usa aparte para comparar todos los logits.
    """
    if policy != POLICY_ID or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Explicacion requiere politica comun y todos los parametros congelados.')
    if any(m.training for m in model.modules()):
        raise ValueError('Todas las capas deben entrar en eval.')
    if images.ndim != 5 or tuple(images.shape[:3]) != (1, 8, 1) or tuple(slice_mask.shape) != (1, 8):
        raise ValueError('Un estudio de ocho posiciones por forward.')
    if slice_mask.dtype != torch.bool or not bool(slice_mask.any()):
        raise ValueError('Mascara invalida.')
    layers = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not layers or any(not isinstance(m, nn.BatchNorm2d) for m in layers):
        raise ValueError('Se requiere BatchNorm2d.')
    flags = [(m, m.training, m.track_running_stats) for m in layers]
    try:
        for m, _, _ in flags:
            m.training = True; m.track_running_stats = False
        yield
    finally:
        for m, training, tracking in flags:
            m.training = training; m.track_running_stats = tracking


def gradcam(model, target_layer, images, slice_mask, class_index, policy=POLICY_ID):
    """Un mapa por corte valido para el logit objetivo del estudio completo.

    Promedio espacial de derivadas por canal, suma ponderada, ReLU y bilineal.
    Escala unica compartida por los ocho cortes del mismo estudio/condicion.
    No aplicar como evidencia de una lesion presente en cada corte.
    """
    if torch.is_inference_mode_enabled():
        raise ValueError('Grad-CAM requiere autograd; no envolver en inference_mode.')
    if any(m.training for m in model.modules()):
        raise ValueError('Usar model.eval().')
    if type(class_index) is not int or not 0 <= class_index < 12:
        raise ValueError('Indice de condicion invalido.')
    if not any(target_layer is m for m in model.modules()):
        raise ValueError('Capa objetivo fuera del modelo.')
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Se requiere modelo sin gradientes acumulados previos.')
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    previous = [(m, m.training, getattr(m, 'track_running_stats', None)) for m in model.modules()]
    requires_grad = [(p, p.requires_grad) for p in model.parameters()]
    captured = []
    handle = None
    try:
        with torch.no_grad():
            reference = evaluation_forward(model, images, slice_mask, policy).detach().clone()
        for p, _ in requires_grad:
            p.requires_grad_(False)
        handle = target_layer.register_forward_hook(lambda module, args, output: captured.append(output))
        with torch.enable_grad(), explanation_forward(model, images, slice_mask, policy):
            x = images.detach().clone().requires_grad_(True)
            logits = model(x, slice_mask)
            if logits.shape != (1, 12) or not bool(torch.isfinite(logits).all()):
                raise ValueError('Logits invalidos.')
            torch.testing.assert_close(logits.detach(), reference, atol=1e-6, rtol=1e-5)
            if len(captured) != 1 or captured[0].ndim != 4 or captured[0].shape[0] != int(slice_mask.sum()):
                raise ValueError('La capa debe emitir un tensor espacial por corte valido una sola vez.')
            activations = captured[0]
            gradients, = torch.autograd.grad(logits[0, class_index], activations, create_graph=False)
            if not bool(torch.isfinite(gradients).all()):
                raise ValueError('Gradientes no finitos.')
            weights = gradients.mean(dim=(-2, -1), keepdim=True)
            coarse = torch.relu((weights * activations).sum(dim=1)).detach()
            raw = F.interpolate(coarse[:, None], size=images.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
            scale = float(raw.max())
            normalized = raw / scale if scale > 0 else torch.zeros_like(raw)
            maps = images.new_zeros((8, *images.shape[-2:]))
            maps[slice_mask[0]] = normalized
            raw_maps = torch.zeros_like(maps); raw_maps[slice_mask[0]] = raw
            output = {'cam': maps.cpu(), 'cam_raw': raw_maps.cpu(), 'logits': logits.detach().cpu(),
                'probabilities': torch.sigmoid(logits.detach()).cpu(), 'class_index': class_index,
                'raw_positive_scale': scale, 'zero_positive_map': scale == 0,
                'activation_shape': list(activations.shape), 'policy': policy, 'version': VERSION,
                'numerical_forward_matches_common_policy': True}
    finally:
        if handle is not None:
            handle.remove()
        for p, flag in requires_grad:
            p.requires_grad_(flag)
        for m, training, tracking in previous:
            if m.training != training or getattr(m, 'track_running_stats', None) != tracking:
                raise RuntimeError('Modo de modelo alterado por la explicacion.')
        if not all(torch.equal(v, model.state_dict()[k].detach().cpu()) for k, v in before.items()):
            raise RuntimeError('Pesos o buffers cambiaron durante Grad-CAM.')
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Gradientes de parametros acumulados inesperadamente.')
    return output
