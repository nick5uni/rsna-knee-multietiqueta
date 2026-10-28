"""Comprobacion del presupuesto para el lanzador CNN BCE, separada del release."""
from .training_contract import TrainConfig


def require_run_budget(budget, *, effective_release_sha256, model):
    if budget.get('estado') != 'aceptado':
        raise ValueError('Presupuesto de corridas todavia propuesto: registrar acuerdo comun antes de entrenar.')
    if budget.get('aceptado_por') != {'Nicolas': True, 'Victor': True} or not budget.get('fuente_acuerdo'):
        raise ValueError('Falta el registro del acuerdo comun sobre el presupuesto.')
    if budget.get('release_sha256') != effective_release_sha256:
        raise ValueError('El presupuesto corresponde a otro release de supervision.')
    if model not in budget.get('configuraciones', []):
        raise ValueError('Configuracion no incluida en el presupuesto.')
    if model != 'CNN_BCE':
        raise ValueError('Este lanzador inicial solo admite CNN BCE; integrar las otras configuraciones por separado.')
    if budget.get('selection') != 'min_dev_masked_bce':
        raise ValueError('El criterio comun de seleccion requiere BCE de desarrollo.')
    if budget.get('cache') != {'tipo': 'RAM_CPU', 'max_cache_studies_por_dataset': 300, 'precargar_train_dev': True}:
        raise ValueError('Politica de cache diferente de la propuesta comun.')
    if budget.get('amp') is not False or budget.get('aumentos') is not False:
        raise ValueError('Cambios de precision o aumentos requieren una version explicita del protocolo.')
    values = dict(budget['train_config'])
    if values.get('loss_name') != 'bce':
        raise ValueError('El presupuesto de CNN BCE debe declarar perdida bce.')
    config = TrainConfig(**values)
    config.validate()
    return config
