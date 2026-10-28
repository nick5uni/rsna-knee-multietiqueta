# Entrada autónoma consolidada a partir de los bloques utilizados en T4.
# Conserva el ZIP original y los módulos congelados; ajusta solo el orquestador.
# Un único lanzamiento del entrenamiento completo al final de esta celda.
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

# Descargar o reutilizar el ZIP íntegro, sin descompresión de Kaggle.
from pathlib import Path
import urllib.request
import hashlib
import shutil
import subprocess
import sys

carpeta = Path("/kaggle/working/paquete_original_verificado")
carpeta.mkdir(exist_ok=True)

nombre = "Paquete_Privado_RSNA_REV2_ComandoUnico.zip"
archivo = carpeta / nombre

# Aceptar también un ZIP íntegro adjunto, sin descargarlo nuevamente.
if not archivo.is_file():
    fuentes_zip = list(find_bounded("/kaggle/input", nombre))
    assert len(fuentes_zip) <= 1, "Hay varias copias del ZIP original como Input."
    if fuentes_zip:
        shutil.copyfile(fuentes_zip[0], archivo)

url = (
    "https://github.com/nick5uni/rsna-knee-multietiqueta/"
    "releases/download/rev2-reproduccion-v1/" + nombre
)

hash_esperado = (
    "b4897e57edf9b426ce6040ab3791f2ac"
    "0e8a63831467881b773aed172825160e"
)

if not archivo.is_file():
    print("Descargando el ZIP original en Kaggle...", flush=True)
    temporal = carpeta / (nombre + ".descargando")
    solicitud = urllib.request.Request(
        url, headers={"User-Agent": "RSNA-Reproduccion"}
    )

    with urllib.request.urlopen(solicitud, timeout=60) as entrada:
        with temporal.open("wb") as salida:
            shutil.copyfileobj(entrada, salida, 8 * 1024 * 1024)

    with temporal.open("rb") as entrada:
        obtenido = hashlib.file_digest(entrada, "sha256").hexdigest()

    assert obtenido == hash_esperado, "El ZIP descargado no coincide."
    temporal.replace(archivo)

with archivo.open("rb") as entrada:
    obtenido = hashlib.file_digest(entrada, "sha256").hexdigest()

assert obtenido == hash_esperado, "El ZIP no supera la comprobación."
print("CORRECTO: ZIP original íntegro.", flush=True)

# Misma preparación y mismo lanzador del notebook de Nicolás.
repo = prepare(str(carpeta), "/kaggle/working")
print("Paquete copiado y hashes correctos:", repo, flush=True)

# Aplicar el mismo ajuste Grad-CAM de la ejecución T4 completada.
from pathlib import Path
import hashlib
import json
import subprocess
import sys

repo = Path(repo)
archivo = repo / "pipeline.py"
texto = archivo.read_text(encoding="utf-8")

# Mantener intactos los módulos y criterios congelados.
congelados = json.loads(
    (repo / "codigo_congelado_sha256.json").read_text(encoding="utf-8")
)
for nombre, esperado in congelados.items():
    obtenido = hashlib.sha256((repo / nombre).read_bytes()).hexdigest()
    assert obtenido == esperado, f"Archivo congelado diferente: {nombre}"

lineas = [
    linea for linea in texto.splitlines(keepends=True)
    if linea.startswith("                result=gradcam(model,")
]
assert len(lineas) == 1, "No se identificó inequívocamente la llamada Grad-CAM."

linea = lineas[0]
limpieza = "                model.zero_grad(set_to_none=True)\n"

if limpieza + linea not in texto:
    respaldo = repo / "pipeline_antes_ajuste_gradcam.txt"
    assert not respaldo.exists(), "Ya existe un respaldo; revisar antes de modificar."
    respaldo.write_bytes(archivo.read_bytes())

    corregido = texto.replace(linea, limpieza + linea, 1)
    compile(corregido, str(archivo), "exec")
    archivo.write_text(corregido, encoding="utf-8")

    registro = {
        "ajuste": "Limpiar gradientes antes de Grad-CAM",
        "sha256_anterior": hashlib.sha256(respaldo.read_bytes()).hexdigest(),
        "sha256_corregido": hashlib.sha256(archivo.read_bytes()).hexdigest(),
        "modulos_congelados": "sin cambios"
    }
    (repo / "ajuste_gradcam.json").write_text(
        json.dumps(registro, indent=2), encoding="utf-8"
    )

print("Ajuste verificado. Iniciando el flujo completo.", flush=True)

# Registro adicional: identifica la preparación autónoma utilizada.
(repo / "preparacion_comando_unico_v2.json").write_text(
    json.dumps({
        "version": "entrada_autonoma_v2_28Oct2026",
        "paquete_sha256": hash_esperado,
        "pipeline_sha256": hashlib.sha256((repo / "pipeline.py").read_bytes()).hexdigest(),
        "modo": "train",
        "nota": "Preparación consolidada; mismo lanzador y ajuste usados en T4."
    }, indent=2), encoding="utf-8"
)

subprocess.run(
    [sys.executable, str(repo / "ejecutar_kaggle.py"), "--mode", "train"],
    cwd=repo,
    check=True
)
