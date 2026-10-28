"""Copia y verifica el paquete privado antes de lanzar el comando completo."""
from pathlib import Path,PurePosixPath
import datetime,hashlib,json,os,shutil,subprocess,sys,zipfile

BUNDLE='Paquete_Privado_RSNA_REV2_ComandoUnico.zip'

def find_bounded(root,name):
    root=Path(root)
    for folder,dirs,files in os.walk(root):
        depth=len(Path(folder).relative_to(root).parts)
        dirs[:]=[] if depth>=5 else [d for d in dirs if d not in {'train_series','.git','__pycache__'}]
        if name in files:yield Path(folder)/name

def safe_path(root,name):
    part=PurePosixPath(name)
    if part.is_absolute() or '..' in part.parts or '\\' in name or ':' in name:raise ValueError('Ruta no permitida en el paquete.')
    result=(Path(root)/name).resolve()
    if not result.is_relative_to(Path(root).resolve()):raise ValueError('Ruta fuera de destino.')
    return result

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def prepare(input_root,working_root):
    input_root=Path(input_root);working_root=Path(working_root)
    if not input_root.is_dir():raise RuntimeError('Se requiere Kaggle con competencia y paquete privado adjuntos como Input.')
    archives=list(find_bounded(input_root,BUNDLE))
    manifests=list(find_bounded(input_root,'PAQUETE_SHA256.json')) if not archives else []
    if len(archives)+len(manifests)!=1:raise RuntimeError('Adjuntar una sola copia del paquete privado como Input; no se encontro una fuente unica.')
    dest=working_root/('comando_unico_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    dest.mkdir(parents=True,exist_ok=False)
    if archives:
        with zipfile.ZipFile(archives[0]) as z:
            expected=json.loads(z.read('PAQUETE_SHA256.json').decode('utf-8'))
            names=z.namelist()
            if len(names)!=len(set(names)) or set(names)!=set(expected)|{'PAQUETE_SHA256.json'}:raise ValueError('Contenido del ZIP diferente del manifiesto.')
            for info in z.infolist():
                target=safe_path(dest,info.filename)
                if (info.external_attr>>16)&0o170000==0o120000:raise ValueError('Enlace simbolico no permitido.')
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(info) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst,4*1024*1024)
    else:
        origin=manifests[0].parent
        expected=json.loads(manifests[0].read_text(encoding='utf-8'))
        for name in expected:
            source=safe_path(origin,name);target=safe_path(dest,name)
            if source.is_symlink() or not source.is_file():raise ValueError('Falta un archivo regular del paquete.')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    for name,sha in expected.items():
        if digest(safe_path(dest,name))!=sha:raise ValueError('Hash incorrecto: '+name)
    repo=dest/'rsna-knee-multietiqueta'
    if not (repo/'ejecutar_kaggle.py').is_file():raise RuntimeError('Paquete sin lanzador.')
    return repo

def main():
    repo=prepare('/kaggle/input','/kaggle/working')
    print('Paquete privado copiado y hashes correctos:',repo,flush=True)
    command=[sys.executable,str(repo/'ejecutar_kaggle.py'),'--mode','train']
    print('Comando unico: python ejecutar_kaggle.py --mode train',flush=True)
    subprocess.run(command,cwd=repo,check=True)

if __name__=='__main__':main()
