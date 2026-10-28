"""Métricas comunes sobre supervisión conocida; no entrena ni realiza inferencia.

NumPy es la única dependencia. Entrada [estudios, condiciones]. Desconocidos
excluidos por máscara, incluso si su etiqueta es NaN. Umbral fijo >= 0.5.
Sin ejemplos conocidos: métricas None (JSON null), excluidas del macro.
AUROC requiere positivos y negativos; AP requiere al menos un positivo.
Precisión y F1 con datos conocidos usan zero_division=0; se informa si
el denominador de precisión es cero para no confundirlo con una medida definida.
"""
import csv
import json
from pathlib import Path

import numpy as np

VERSION = 'evaluador_comun_v2_20261017'
UMBRAL = 0.5
EPSILON_BCE = 1e-7


def ranked_curves(y, p):
    """Curvas por umbrales agrupados (empates), sin curvas ficticias."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape:
        raise ValueError('Curvas: se esperan dos vectores del mismo tamaño.')
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Curvas: etiquetas conocidas binarias y finitas.')
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Curvas: probabilidades finitas entre 0 y 1.')
    empty = {'FPR': [], 'TPR': [], 'recall': [], 'precision': [], 'AP': None}
    if y.size == 0:
        return empty
    npos = int(y.sum())
    nneg = len(y) - npos
    order = np.argsort(-p, kind='stable')
    yy, pp = y[order], p[order]
    ends = np.r_[np.flatnonzero(np.diff(pp)), len(pp) - 1]
    tp = np.cumsum(yy)[ends]
    fp = ends + 1 - tp
    out = dict(empty)
    if npos:
        recall = tp / npos
        precision = tp / (tp + fp)
        out.update(recall=recall.tolist(), precision=precision.tolist(),
                   AP=float(np.sum(np.diff(np.r_[0., recall]) * precision)))
    if npos and nneg:
        out.update(FPR=np.r_[0., fp / nneg].tolist(),
                   TPR=np.r_[0., tp / npos].tolist())
    return out


def _mean_defined(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _bce(y, p):
    if not len(y):
        return None
    p = np.clip(p, EPSILON_BCE, 1 - EPSILON_BCE)
    return float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p)))


def metrics(y, p, mask, conditions):
    """Evalúa una matriz de predicciones con una máscara común explícita."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    raw_mask = np.asarray(mask)
    conditions = list(conditions)
    if y.ndim != 2 or p.shape != y.shape or raw_mask.shape != y.shape:
        raise ValueError('y, p y mask deben tener la misma forma [estudios, condiciones].')
    if len(conditions) != y.shape[1] or not conditions or len(set(conditions)) != len(conditions):
        raise ValueError('Orden de condiciones vacío, duplicado o incompatible con las matrices.')
    if not all(isinstance(c, str) and c.strip() for c in conditions):
        raise ValueError('Los nombres de condición deben ser texto no vacío.')
    if not np.isin(raw_mask, [0, 1]).all():
        raise ValueError('La máscara debe contener solamente 0/1 o booleanos.')
    mask = raw_mask.astype(bool)
    if not np.isfinite(y[mask]).all() or not np.isin(y[mask], [0, 1]).all():
        raise ValueError('Las etiquetas conocidas deben ser binarias y finitas.')
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Todas las probabilidades deben ser finitas y estar entre 0 y 1.')
    per = []
    for j, condition in enumerate(conditions):
        yy, pp = y[mask[:, j], j], p[mask[:, j], j]
        pos, neg = pp[yy == 1], pp[yy == 0]
        curve = ranked_curves(yy, pp)
        pred = pp >= UMBRAL
        tp = int(((yy == 1) & pred).sum())
        tn = int(((yy == 0) & ~pred).sum())
        fp = int(((yy == 0) & pred).sum())
        fn = int(((yy == 1) & ~pred).sum())
        known = len(yy)
        auc = float(((pos[:, None] > neg[None, :]) +
                     .5 * (pos[:, None] == neg[None, :])).mean()) if len(pos) and len(neg) else None
        precision = (tp / (tp + fp) if tp + fp else 0.) if known else None
        f1 = (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.) if known else None
        per.append({
            'condicion': condition, 'positivos': len(pos), 'negativos': len(neg),
            'conocidas': known, 'BCE': _bce(yy, pp), 'AUROC': auc, 'AP': curve['AP'],
            'Precision_umbral_05': precision, 'F1_umbral_05': f1,
            'sensibilidad_umbral_05': tp / len(pos) if len(pos) else None,
            'especificidad_umbral_05': tn / len(neg) if len(neg) else None,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'probabilidad_media': float(pp.mean()) if known else None,
            'probabilidad_media_negativos': float(neg.mean()) if len(neg) else None,
            'precision_denominador_definido': bool(tp + fp),
            'motivo_AUROC_no_disponible': ('sin_etiquetas_conocidas' if not known else 'falta_positivo_o_negativo') if auc is None else None,
            'motivo_AP_no_disponible': ('sin_etiquetas_conocidas' if not known else 'sin_positivos') if curve['AP'] is None else None,
            'curvas': curve,
        })
    macro_fields = {'AUROC_macro': 'AUROC', 'AP_macro': 'AP',
                    'F1_macro_umbral_05': 'F1_umbral_05',
                    'Precision_macro_umbral_05': 'Precision_umbral_05'}
    result = {'version_evaluador': VERSION, 'umbral': UMBRAL,
              'BCE': _bce(y[mask], p[mask]), 'conocidas': int(mask.sum()),
              'desconocidas': int(mask.size - mask.sum()),
              'estudios': len(y), 'por_condicion': per,
              'convencion_denominador_cero': 'precision_y_F1=0_si_hay_datos_conocidos; sin_datos=null',
              'denominadores_macro': {}, 'condiciones_incluidas_macro': {}}
    for name, key in macro_fields.items():
        result[name] = _mean_defined([r[key] for r in per])
        included = [r['condicion'] for r in per if r[key] is not None]
        result['denominadores_macro'][name] = len(included)
        result['condiciones_incluidas_macro'][name] = included
    result['condiciones_precision_denominador_definido'] = sum(r['precision_denominador_definido'] for r in per)
    return result


def evaluate_common(y, predictions_by_model, mask, conditions):
    """Todos los modelos se evalúan con los mismos estudios/etiquetas/máscara.

El llamador debe alinear filas por identificador antes de suministrar matrices.
Para CSV se proporciona load_predictions, que comprueba ese enlace.
"""
    if not predictions_by_model:
        raise ValueError('Debe existir al menos un modelo.')
    conditions = list(conditions)
    return {name: metrics(y, p, mask, conditions) for name, p in predictions_by_model.items()}


def load_predictions(path, expected_uids, conditions, expected_split):
    """Alinea por UID y rechaza duplicados, cohortes distintas y split incorrecto."""
    expected_uids = list(expected_uids)
    if len(expected_uids) != len(set(expected_uids)):
        raise ValueError('IDs esperados duplicados.')
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'StudyInstanceUID', 'split', *conditions}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Faltan identificador, split o columnas de condiciones.')
        rows = list(reader)
    by_uid = {r['StudyInstanceUID']: r for r in rows}
    if len(rows) != len(by_uid) or set(by_uid) != set(expected_uids):
        raise ValueError('Predicciones duplicadas, faltantes o fuera de la cohorte.')
    if any(r['split'] != expected_split for r in rows):
        raise ValueError('Partición de predicciones incorrecta.')
    out = np.array([[float(by_uid[u][c]) for c in conditions] for u in expected_uids], dtype=np.float64)
    return out.reshape(len(expected_uids), len(conditions))


def save_json(path, value):
    """None se conserva como null; prohíbe NaN/Infinity en los resultados."""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
