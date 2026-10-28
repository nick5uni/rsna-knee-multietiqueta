"""Pruebas de la entrada autónoma; no ejecutan PyTorch, RSNA ni entrenamiento."""
import ast,contextlib,hashlib,io,json,subprocess,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parent
SOURCE=(R/'bootstrap_kaggle_v2.py').read_text()
ORIGINAL=(R/'pipeline_antes_ajuste_gradcam.txt').read_bytes()
FINAL_SHA='ab93fb0781403a9fe7d1da1911693dd9d5d42e96ea368bc13ec48ce5c46de3d1'
ZIP_SHA='b4897e57edf9b426ce6040ab3791f2ac0e8a63831467881b773aed172825160e'
BUNDLE='Paquete_Privado_RSNA_REV2_ComandoUnico.zip'

def fixture(bad_frozen=False):
 prefix='rsna-knee-multietiqueta/'
 frozen=b'# fixture de control; no es codigo academico\n'
 files={prefix+'pipeline.py':ORIGINAL,prefix+'src/control.py':frozen,
        prefix+'ejecutar_kaggle.py':(R/'ejecutar_kaggle.py').read_bytes(),
        prefix+'codigo_congelado_sha256.json':json.dumps({'src/control.py':('0'*64 if bad_frozen else hashlib.sha256(frozen).hexdigest())}).encode()}
 manifest={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
 buf=io.BytesIO()
 with zipfile.ZipFile(buf,'w') as z:
  for n,b in files.items():z.writestr(n,b)
  z.writestr('PAQUETE_SHA256.json',json.dumps(manifest))
 return buf.getvalue()

def adapted(root,zipsha):
 class Adapt(ast.NodeTransformer):
  def visit_Constant(self,node):
   if isinstance(node.value,str):
    if node.value.startswith('/kaggle/'):
     return ast.copy_location(ast.Constant(str(root/node.value.removeprefix('/kaggle/'))),node)
    if node.value==ZIP_SHA:return ast.copy_location(ast.Constant(zipsha),node)
   return node
 return compile(ast.fix_missing_locations(Adapt().visit(ast.parse(SOURCE))),'celda_autonoma','exec')

class Consolidacion(unittest.TestCase):
 def run_case(self,source,location='input',download=None,expected_error=None):
  data=fixture() if source is None else source
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);(root/'input').mkdir();(root/'working').mkdir()
   if location=='input':(root/'input'/BUNDLE).write_bytes(data)
   elif location=='cache':
    cache=root/'working/paquete_original_verificado';cache.mkdir();(cache/BUNDLE).write_bytes(data)
   sha=hashlib.sha256(data).hexdigest()
   payload=data if download is None else download
   namespace={'__name__':'__main__'}
   with patch('urllib.request.urlopen',return_value=io.BytesIO(payload)) as get, \
        patch('subprocess.run',return_value=subprocess.CompletedProcess(['test'],0)) as launch, \
        contextlib.redirect_stdout(io.StringIO()):
    if expected_error:
     with self.assertRaises(expected_error):exec(adapted(root,sha),namespace)
     launch.assert_not_called();return
    exec(adapted(root,sha),namespace)
   launch.assert_called_once()
   self.assertEqual(launch.call_args.args[0][-2:],['--mode','train'])
   repo=launch.call_args.kwargs['cwd']
   self.assertEqual(hashlib.sha256((repo/'pipeline.py').read_bytes()).hexdigest(),FINAL_SHA)
   self.assertEqual((repo/'pipeline_antes_ajuste_gradcam.txt').read_bytes(),ORIGINAL)
   self.assertTrue((repo/'ajuste_gradcam.json').is_file())
   self.assertTrue((repo/'preparacion_comando_unico_v2.json').is_file())
   if location=='download':get.assert_called_once()
   else:get.assert_not_called()
 def test_sesion_limpia_con_zip_integro_en_input(self):self.run_case(None)
 def test_sesion_limpia_sin_zip_descarga_y_verifica(self):self.run_case(None,'download')
 def test_reutiliza_zip_integro_en_cache(self):self.run_case(None,'cache')
 def test_descarga_alterada_bloquea_antes_de_lanzar(self):self.run_case(None,'download',download=b'archivo incompleto',expected_error=AssertionError)
 def test_modulo_congelado_incorrecto_bloquea(self):self.run_case(fixture(bad_frozen=True),expected_error=AssertionError)

if __name__=='__main__':unittest.main(verbosity=2)
