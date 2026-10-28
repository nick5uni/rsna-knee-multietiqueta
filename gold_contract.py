"""Controles de cierre gold independientes de PyTorch y de etiquetas médicas."""
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def validate_gold(manifest, references, control, closure, *, gold_uids, canonical_groups,
                  canonical_patient_hashes, expected_files):
    if closure.get('estado') != 'manifiesto_y_evaluador_confirmados':
        raise ValueError('Falta cierre gold final; la preparación anterior no lo sustituye.')
    if closure.get('protocolo_aceptado') is not True or closure.get('fallback_coronal_aceptado') is not True:
        raise ValueError('Falta registrar el acuerdo del protocolo y del fallback coronal.')
    if not closure.get('fuente_acuerdo'):
        raise ValueError('Falta procedencia del acuerdo de cierre.')
    if closure.get('archivos_sha256') != expected_files:
        raise ValueError('El cierre no corresponde a estos archivos finales.')
    if not canonical_patient_hashes:
        raise ValueError('Falta referencia PatientID train/dev; no comparar contra un conjunto vacio.')
    gold_uids = set(gold_uids)
    rows = {r['StudyInstanceUID']: r for r in manifest}
    refs = {r['StudyInstanceUID']: r for r in references}
    if len(manifest) != len(rows) or len(rows) != 58 or set(rows) != gold_uids:
        raise ValueError('El manifiesto debe cubrir exactamente los 58 gold históricos, sin duplicados.')
    if len(references) != len(refs) or set(refs) != gold_uids:
        raise ValueError('Faltan referencias de lectura para los 58 gold, incluidas ambas coronales.')
    if set(rows) & set(canonical_groups):
        raise ValueError('Hay identificadores gold presentes en train/dev.')
    if (control.get('control_grupos_correcto') is not True or
            control.get('grupos_canonicos_reproducidos') != 300 or
            control.get('gold_controlados') != 58 or
            control.get('cruces_texto_gold_train_dev') != []):
        raise ValueError('El control exacto de grupos no está cerrado sin cruces.')
    if set(control.get('grupos_gold', {})) != gold_uids:
        raise ValueError('Los grupos del control no cubren exactamente estos gold.')
    if set(control['grupos_gold'].values()) & set(canonical_groups.values()):
        raise ValueError('El propio registro contiene grupos compartidos con train/dev.')
    planes = Counter()
    for uid, row in rows.items():
        sid = row.get('SeriesInstanceUID')
        ref = refs[uid]
        if not sid or row.get('split') != 'gold' or ref.get('SeriesInstanceUID') != sid:
            raise ValueError('Serie, partición o referencia inconsistente.')
        if row.get('estado') != 'correcto_entrada_8_cortes' or ref.get('estado') != 'correcto_entrada_8_cortes':
            raise ValueError('Hay estudios pendientes de lectura o con incidencias.')
        if row.get('Fluid_Sensitive') != '1' or row.get('Fat_Suppression') != '1':
            raise ValueError('Serie fuera del criterio de fluido y supresión grasa.')
        plane = row.get('Anatomical_Plane')
        if plane not in ('Sagittal', 'Coronal'):
            raise ValueError('Plano no incluido en el acuerdo.')
        planes[plane] += 1
        if ref.get('entrada_shape') != [8, 1, 224, 224] or ref.get('mascara_cortes') != [True] * 8:
            raise ValueError('La referencia no contiene ocho cortes válidos 224x224.')
        if ref.get('error') or ref.get('advertencias'):
            raise ValueError('La lectura conserva errores o advertencias pendientes.')
        selected = ref.get('archivos_seleccionados', [])
        if len(selected) != 8 or len({s['member'] for s in selected}) != 8:
            raise ValueError('Referencias de cortes faltantes o repetidas.')
        for cut in selected:
            parts = PurePosixPath(cut['member']).parts
            if len(parts) != 4 or parts[:3] != ('train_series', uid, sid) or '..' in parts:
                raise ValueError('Ruta DICOM fuera de la serie elegida.')
            if len(cut.get('sha256', '')) != 64 or cut.get('bytes', 0) <= 0:
                raise ValueError('Falta integridad de un DICOM seleccionado.')
        if row.get('entrada_sha256') != ref.get('entrada_sha256') or len(ref.get('entrada_sha256', '')) != 64:
            raise ValueError('Hash de entrada ausente o inconsistente.')
        patient_hash = ref.get('patient_id_hash')
        if not patient_hash or ref.get('patient_id_consistente') is not True:
            raise ValueError('Falta resolver el control de PatientID observado.')
        if patient_hash in canonical_patient_hashes:
            raise ValueError('PatientID observado coincide con train/dev; revisar antes de inferencia.')
    if dict(planes) != {'Sagittal': 56, 'Coronal': 2}:
        raise ValueError('El escenario esperado es 56 sagitales y dos coronales; cualquier cambio requiere cierre revisado.')
    return {'gold': 58, 'planos': dict(planes), 'cruces_UID': 0, 'cruces_texto': 0,
            'cruces_patient_id_observado': 0, 'nota': 'PatientID observado no acredita identidad clínica real.'}
