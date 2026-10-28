"""Deriva configuracion focal del presupuesto comun sin reescribirlo."""
from dataclasses import replace
import math

from .run_budget import require_run_budget


def require_focal_plan(spec, budget, *, release_sha256, implementation_sha256):
    base = require_run_budget(budget, effective_release_sha256=release_sha256, model='CNN_BCE')
    if 'ResNet18_FOCAL' not in budget.get('configuraciones', []):
        raise ValueError('ResNet18 focal no esta incluida en el presupuesto.')
    if spec.get('estado') not in ('acordado', 'comun_recibida_para_validar') or not spec.get('fuente_acuerdo_parametros'):
        raise ValueError('Funcion y parametros focal pendientes de version comun; no iniciar la corrida academica.')
    if spec.get('funcion_sha256') != implementation_sha256:
        raise ValueError('La funcion focal no corresponde a la version registrada.')
    if spec.get('reduccion') != 'mean_solo_etiquetas_conocidas' or spec.get('seleccion_checkpoint') != 'min_dev_masked_bce':
        raise ValueError('Focal requiere media de conocidas y seleccion por BCE dev.')
    gamma, alpha = spec['gamma'], spec['alpha']
    if type(gamma) not in (int, float) or not math.isfinite(gamma) or gamma < 0:
        raise ValueError('gamma focal invalido.')
    if alpha is not None and (type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 <= alpha <= 1):
        raise ValueError('alpha focal invalido.')
    return replace(base, loss_name='focal'), {'gamma': gamma, 'alpha': alpha}
