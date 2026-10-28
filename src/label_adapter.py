"""Enlace de propuestas por identificador; nunca aprueba ni convierte silencio en ausencia."""
from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from .data_checks import LABELS

APPROVED = 'aprobada_equipo'  # Propuesta de estado operativo, no validacion clinica.


def read_table(path, required):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)) or not set(required).issubset(headers):
            raise ValueError(f'Columnas faltantes o duplicadas en {Path(path).name}.')
        rows = list(reader)
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError('Fila CSV incompleta o con campos de mas.')
    return rows


def indexed(rows, name):
    result = {}
    for r in rows:
        uid = r['StudyInstanceUID']
        if not uid or uid in result:
            raise ValueError(f'Estudio vacio o duplicado en {name}.')
        result[uid] = r
    return result


@dataclass
class LabelBundle:
    study_uids: list
    series_uids: list
    targets: np.ndarray
    available_mask: np.ndarray
    approved_mask: np.ndarray
    splits: dict
    groups_checked: bool
    provenance: dict

    def summary(self):
        available = self.available_mask
        return {'estudios': len(self.study_uids), 'orden_condiciones': list(LABELS),
                'propuestas_positivas': int(((self.targets == 1) & available).sum()),
                'propuestas_negativas': int(((self.targets == 0) & available).sum()),
                'desconocidas': int((~available).sum()),
                'propuestas_disponibles': int(available.sum()),
                'etiquetas_aprobadas_equipo': int(self.approved_mask.sum()),
                'propuestas_disponibles_sin_aprobar': int((available & ~self.approved_mask).sum()),
                'particiones_recibidas': bool(self.splits), 'grupos_texto_comprobados': self.groups_checked,
                'puede_iniciar_entrenamiento': self.groups_checked and all(
                    any(self.splits.get(uid) == split and self.approved_mask[i].any()
                        for i, uid in enumerate(self.study_uids)) for split in ('train', 'dev')),
                'relleno_targets': 'Cero numerico en desconocidos, siempre con mascara False; no es etiqueta negativa.',
                'procedencia_sha256': self.provenance}

    def training_view(self, split):
        """Solo aprobadas; excluye y registra estudios sin ninguna etiqueta aprobada."""
        if split not in ('train', 'dev') or not self.splits or not self.groups_checked:
            raise ValueError('Faltan particiones train/dev y comprobacion de grupos; entrenamiento bloqueado.')
        chosen = [i for i, uid in enumerate(self.study_uids) if self.splits[uid] == split]
        usable = [i for i in chosen if self.approved_mask[i].any()]
        skipped = [self.study_uids[i] for i in chosen if not self.approved_mask[i].any()]
        if not usable:
            raise ValueError('No hay estudios con etiquetas aprobadas en esta particion; entrenamiento bloqueado.')
        return {'indices': usable, 'StudyInstanceUID': [self.study_uids[i] for i in usable],
                'targets': self.targets[usable].copy(), 'label_mask': self.approved_mask[usable].copy(),
                'estudios_omitidos_sin_etiquetas_aprobadas': skipped}


def load_label_bundle(labels_path, mask_path, annotations_path, manifest, *,
                      gold_uids, excluded_uids, partitions_path=None, groups_by_study=None):
    """Matrices de la misma muestra. El orden de filas de origen no controla el enlace."""
    required = ['StudyInstanceUID', 'SeriesInstanceUID', *LABELS]
    values = indexed(read_table(labels_path, required), 'etiquetas')
    masks = indexed(read_table(mask_path, required), 'mascaras')
    expected = indexed(manifest, 'manifiesto')
    if not expected or values.keys() != expected.keys() or masks.keys() != expected.keys():
        raise ValueError('La muestra de etiquetas/mascaras debe coincidir exactamente con el manifiesto.')
    if set(expected) & (set(gold_uids) | set(excluded_uids)):
        raise ValueError('Se encontro un estudio gold o excluido en la muestra.')
    notes = read_table(annotations_path, ['StudyInstanceUID', 'condicion', 'validacion',
                                        'valor_propuesto', 'mascara_propuesta', 'evidencia_textual'])
    annotations = {}
    for a in notes:
        key = (a['StudyInstanceUID'], a['condicion'])
        if key in annotations:
            raise ValueError('Anotacion estudio-condicion duplicada.')
        annotations[key] = a
    if set(annotations) != {(uid, c) for uid in expected for c in LABELS}:
        raise ValueError('Las anotaciones no cubren exactamente las doce condiciones de la muestra.')
    uids = list(expected)
    targets = np.zeros((len(uids), len(LABELS)), dtype=np.float32)
    available = np.zeros(targets.shape, dtype=bool)
    approved = np.zeros(targets.shape, dtype=bool)
    for i, uid in enumerate(uids):
        sid = expected[uid]['SeriesInstanceUID']
        if not sid or values[uid]['SeriesInstanceUID'] != sid or masks[uid]['SeriesInstanceUID'] != sid:
            raise ValueError('SeriesInstanceUID diferente del manifiesto.')
        for j, c in enumerate(LABELS):
            value, mask = values[uid][c].strip(), masks[uid][c].strip()
            if value not in ('', '0', '1') or mask not in ('0', '1'):
                raise ValueError('Etiqueta o mascara no binaria; desconocido debe ser vacio.')
            if (value != '') != (mask == '1'):
                raise ValueError('Mascara inconsistente: desconocidos requieren vacio y mascara 0.')
            a = annotations[(uid, c)]
            if a['valor_propuesto'].strip() != value or a['mascara_propuesta'].strip() != mask:
                raise ValueError('Anotacion diferente de las matrices.')
            if mask == '1':
                if not a['evidencia_textual'].strip():
                    raise ValueError('Propuesta binaria sin evidencia textual.')
                targets[i, j] = int(value)
                available[i, j] = True
                approved[i, j] = a['validacion'] == APPROVED
    splits = {}
    checked = False
    if partitions_path is not None:
        rows = json.loads(Path(partitions_path).read_text(encoding='utf-8-sig'))
        partitions = indexed(rows, 'particiones')
        if partitions.keys() != expected.keys() or any(r.get('split') not in ('train', 'dev') for r in rows):
            raise ValueError('Particiones incompletas o diferentes de train/dev para la muestra debil.')
        splits = {uid: partitions[uid]['split'] for uid in uids}
        if set(splits.values()) != {'train', 'dev'}:
            raise ValueError('Se requieren tanto train como dev.')
        if groups_by_study is not None:
            if not set(uids).issubset(groups_by_study) or any(not groups_by_study[uid] for uid in uids):
                raise ValueError('Faltan grupos de texto para verificar las particiones.')
            assignments = {}
            for uid in uids:
                group = groups_by_study[uid]
                if group in assignments and assignments[group] != splits[uid]:
                    raise ValueError('Fuga: un grupo de texto repetido cruza train y dev.')
                assignments[group] = splits[uid]
            checked = True
    paths = {'etiquetas': labels_path, 'mascaras': mask_path, 'anotaciones': annotations_path}
    if partitions_path is not None:
        paths['particiones'] = partitions_path
    provenance = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in paths.items()}
    return LabelBundle(uids, [expected[u]['SeriesInstanceUID'] for u in uids],
                       targets, available, approved, splits, checked, provenance)
