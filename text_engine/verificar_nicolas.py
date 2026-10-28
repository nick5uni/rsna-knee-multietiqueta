"""Ejecuta los comprobadores originales de Nicolás sin editarlos."""
from pathlib import Path
import sys,subprocess,tempfile,shutil

def verificar(carpeta):
    carpeta=Path(carpeta).resolve();pruebas=carpeta/'pruebas_Nicolas';salidas={}
    with tempfile.TemporaryDirectory() as d:
        puente=Path(d)
        shutil.copyfile(carpeta/'adaptador_bilingue.py',puente/'adaptador_bilingue.py')
        texto=("import importlib.util,sys\nfrom pathlib import Path\n"+f"P=Path({str(carpeta)!r})\nsys.path.insert(0,str(P))\n"+
         "s=importlib.util.spec_from_file_location('motor_puente_Nicolas',P/'motor_textual_comun.py')\nm=importlib.util.module_from_spec(s);s.loader.exec_module(m)\nVERSION=m.VERSION\nmotor=m.MotorTextualComun(P)\nextraer_es=motor.extraer_es\n")
        (puente/'extractor_es_candidato.py').write_text(texto,encoding='utf-8')
        r=subprocess.run([sys.executable,'-X','utf8',str(pruebas/'comprobar_regresiones_Nicolas.py'),str(puente)],capture_output=True,text=True,encoding='utf-8')
        if r.returncode:raise AssertionError(r.stdout+r.stderr)
        salidas['regresiones_originales_6_y_traza']=r.stdout+r.stderr
    r=subprocess.run([sys.executable,'-X','utf8',str(pruebas/'comprobar_OA_compartida_Nicolas.py'),str(carpeta)],capture_output=True,text=True,encoding='utf-8')
    if r.returncode:raise AssertionError(r.stdout+r.stderr)
    salidas['OA_compartida_4']=r.stdout+r.stderr
    return salidas
