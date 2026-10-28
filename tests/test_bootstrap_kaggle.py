import hashlib,json,tempfile,unittest,zipfile
from pathlib import Path
import bootstrap_kaggle as bootstrap

class BootstrapChecks(unittest.TestCase):
    def test_unsafe_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ['../escape','/absolute','C:/escape','a\\b']:
                with self.assertRaises(ValueError):bootstrap.safe_path(root,name)
    def test_flat_input_is_verified_and_copied(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'input';source.mkdir()
            content=b'# synthetic launcher';name='rsna-knee-multietiqueta/ejecutar_kaggle.py'
            file=source/name;file.parent.mkdir();file.write_bytes(content)
            (source/'PAQUETE_SHA256.json').write_text(json.dumps({name:hashlib.sha256(content).hexdigest()}))
            repo=bootstrap.prepare(source,root/'working')
            self.assertEqual((repo/'ejecutar_kaggle.py').read_bytes(),content)
            self.assertEqual(file.read_bytes(),content)
    def test_tampered_zip_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'input';source.mkdir()
            name='rsna-knee-multietiqueta/ejecutar_kaggle.py'
            with zipfile.ZipFile(source/bootstrap.BUNDLE,'w') as z:
                z.writestr(name,'changed');z.writestr('PAQUETE_SHA256.json',json.dumps({name:'0'*64}))
            with self.assertRaisesRegex(ValueError,'Hash incorrecto'):bootstrap.prepare(source,root/'working')

if __name__=='__main__':unittest.main()
