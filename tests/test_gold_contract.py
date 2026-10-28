import copy
import unittest
from gold_contract import validate_gold


def fixture():
    rows, refs = [], []
    gold = {f'gold{i}' for i in range(58)}
    for i in range(58):
        uid, sid = f'gold{i}', f'series{i}'
        rows.append({'StudyInstanceUID': uid, 'SeriesInstanceUID': sid, 'split': 'gold',
            'estado': 'correcto_entrada_8_cortes', 'Anatomical_Plane': 'Sagittal' if i < 56 else 'Coronal',
            'Fluid_Sensitive': '1', 'Fat_Suppression': '1', 'entrada_sha256': 'a'*64})
        refs.append({'StudyInstanceUID': uid, 'SeriesInstanceUID': sid,
            'estado': 'correcto_entrada_8_cortes', 'entrada_shape': [8, 1, 224, 224],
            'mascara_cortes': [True]*8, 'error': '', 'advertencias': [],
            'archivos_seleccionados': [{'member': f'train_series/{uid}/{sid}/{j}.dcm',
                'bytes': 100, 'sha256': 'b'*64} for j in range(8)],
            'entrada_sha256': 'a'*64, 'patient_id_hash': f'patient{i}', 'patient_id_consistente': True})
    files = {'manifest.json': 'x', 'refs.json': 'y', 'control.json': 'z'}
    control = {'control_grupos_correcto': True, 'grupos_canonicos_reproducidos': 300,
        'gold_controlados': 58, 'cruces_texto_gold_train_dev': [],
        'grupos_gold': {u: u for u in gold}}
    closure = {'estado': 'manifiesto_y_evaluador_confirmados', 'protocolo_aceptado': True,
        'fallback_coronal_aceptado': True, 'fuente_acuerdo': 'fixture artificial',
        'archivos_sha256': files}
    kw = {'gold_uids': gold, 'canonical_groups': {'train1': 'texto_train'},
        'canonical_patient_hashes': {'patient_train'}, 'expected_files': files}
    return rows, refs, control, closure, kw


class GoldContractChecks(unittest.TestCase):
    def check(self, f): return validate_gold(*f[:4], **f[4])
    def test_referencia_patient_vacia_bloquea(self):
        f=fixture(); f[4]['canonical_patient_hashes']=set()
        with self.assertRaises(ValueError): self.check(f)
    def test_cierre_completo(self):
        self.assertEqual(self.check(fixture())['gold'], 58)
    def test_preparacion_anterior_no_habilita(self):
        f=fixture(); f[3]['estado']='pendiente'
        with self.assertRaises(ValueError): self.check(f)
    def test_faltan_las_dos_referencias_coronales(self):
        f=fixture(); f[1].pop(); f[1].pop()
        with self.assertRaises(ValueError): self.check(f)
    def test_cruce_texto_bloquea(self):
        f=fixture(); f[2]['grupos_gold']['gold0']='texto_train'
        with self.assertRaises(ValueError): self.check(f)
    def test_cruce_patient_observado_bloquea(self):
        f=fixture(); f[1][0]['patient_id_hash']='patient_train'
        with self.assertRaises(ValueError): self.check(f)
    def test_serie_cambiada_bloquea(self):
        f=fixture(); f[0][0]['SeriesInstanceUID']='otra'
        with self.assertRaises(ValueError): self.check(f)
    def test_cortes_incompletos_bloquean(self):
        f=fixture(); f[1][0]['archivos_seleccionados'].pop()
        with self.assertRaises(ValueError): self.check(f)
    def test_archivo_no_corresponde_cierre(self):
        f=fixture(); f[3]['archivos_sha256']={'otro':'hash'}
        with self.assertRaises(ValueError): self.check(f)
    def test_fallback_sin_acuerdo_bloquea(self):
        f=fixture(); f[3]['fallback_coronal_aceptado']=False
        with self.assertRaises(ValueError): self.check(f)


if __name__=='__main__': unittest.main(verbosity=2)
