"""Detecta insumos de Kaggle y ejecuta el mismo pipeline documentado."""
from pathlib import Path
import argparse, datetime, hashlib, importlib.metadata, json, os, subprocess, sys, zipfile

HERE=Path(__file__).resolve().parent
CSV_SHA='8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33'
SERIES_SHA='573c1d80772bf41211c91b149c95677385a1c22d63f485c347f1b46c0177aef3'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def bounded_find(root,filename,max_depth=4):
    root=Path(root)
    for folder,dirs,files in os.walk(root):
        depth=len(Path(folder).relative_to(root).parts)
        dirs[:]=[] if depth>=max_depth else [n for n in dirs if n not in {'train_series','__pycache__','.git'}]
        if filename in files:yield Path(folder)/filename

def discover_data(root):
    matches=[]
    for csv in bounded_find(root,'train.csv'):
        folder=csv.parent;series=folder/'train_series.csv'
        if series.is_file() and (folder/'train_series').is_dir() and sha(csv)==CSV_SHA and sha(series)==SERIES_SHA:
            matches.append(folder)
    if len(matches)!=1:raise RuntimeError(f'Se esperaba una fuente RSNA compatible; encontradas {len(matches)}. Adjuntar la competencia y conservar la version de los CSV.')
    return matches[0]

def gpu_and_python():
    if sys.version_info[:2]!=(3,13):raise RuntimeError('La referencia requiere Python 3.13; se conserva el entorno acordado.')
    probe="import torch,json; print(json.dumps({'version':str(torch.__version__),'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}))"
    state=json.loads(subprocess.check_output([sys.executable,'-c',probe],text=True))
    if not state['cuda'] or 'T4' not in state['gpu']:raise RuntimeError('Activar GPU Tesla T4 en Settings. No se inicia entrenamiento en CPU ni se cambia el protocolo.')
    if state['version'].split('+')[0]!='2.11.0':raise RuntimeError('La referencia requiere torch 2.11.0 con CUDA; instalar requirements.txt con el indice cu128 antes de ejecutar.')
    return state

def ensure_light_dependencies():
    required={'torchvision':'0.26.0','numpy':'2.1.3','pydicom':'3.0.2','Pillow':'12.3.0','matplotlib':'3.10.8'}
    missing=[]
    for package,version in required.items():
        try:found=importlib.metadata.version(package).split('+')[0]
        except importlib.metadata.PackageNotFoundError:found=None
        if found!=version:missing.append(f'{package}=={version}')
    if any(x.startswith('torchvision') for x in missing):raise RuntimeError('Se requiere torchvision 0.26.0 compatible con CUDA; no sustituir el runtime silenciosamente.')
    if missing:
        print('Preparando dependencias documentadas:',missing,flush=True)
        subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check',*missing],check=True)

def make_config(data_root,working_root):
    return {'data_csv':str(data_root/'train.csv'),'data_series_csv':str(data_root/'train_series.csv'),
            'dataset_zip':str(working_root/'no_necesario_con_datos_montados.zip'),'data_root':str(data_root),
            'private_inputs':str(HERE/'private'),'cache_root':str(working_root/'cache'),
            'output_root':str(working_root/'resultados'),'cpu_threads':4,'device':'cuda:0'}

def check_complete(output,mode):
    state=json.loads((output/'ejecucion.json').read_text(encoding='utf-8'))
    if state['estado']!='OK' or state['mode']!=mode:raise RuntimeError('No hay una ejecucion completa correcta.')
    if mode=='train':
        if state.get('nuevo_entrenamiento') is not True:raise RuntimeError('La evidencia no acredita entrenamiento nuevo.')
        summaries=json.loads((output/'entrenamiento_resumen.json').read_text(encoding='utf-8'))
        for name in ['CNN_BCE','ResNet18_BCE','ResNet18_FOCAL','ResNet18_BCE_FILTRO']:
            if name not in summaries:raise RuntimeError('Falta configuracion: '+name)
            for filename in ['best.pt','last.pt','history.json','run_summary.json']:
                p=output/'entrenamiento'/name/filename
                if not p.is_file():raise RuntimeError('Falta archivo: '+str(p))
            folder=output/'entrenamiento'/name
            summary=json.loads((folder/'run_summary.json').read_text(encoding='utf-8'))
            history=json.loads((folder/'history.json').read_text(encoding='utf-8'))
            if not history or min(history,key=lambda row:row['dev']['masked_bce'])['epoch']!=summary['selected_epoch']:
                raise RuntimeError('La epoca seleccionada no coincide con el historial DEV: '+name)
            for filename in ['best.pt','last.pt']:
                if sha(folder/filename)!=summary['sha256'][filename]:raise RuntimeError('Checkpoint alterado: '+name+'/'+filename)
            for filename in [f'predictions_dev_{name}.csv',f'predictions_gold_{name}.csv']:
                if not (output/filename).is_file():raise RuntimeError('Faltan predicciones: '+name)
    for filename in ['metrics_gold.json','gradcam/verificacion.json','texto/metricas.json']:
        if not (output/filename).is_file():raise RuntimeError('Falta evidencia: '+filename)
    metrics=json.loads((output/'metrics_gold.json').read_text(encoding='utf-8'))
    if set(metrics)!={'CNN_BCE','ResNet18_BCE','ResNet18_FOCAL','ResNet18_BCE_FILTRO'}:raise RuntimeError('Metricas GOLD incompletas.')
    maps=json.loads((output/'gradcam/verificacion.json').read_text(encoding='utf-8'))
    if len(maps)!=6 or len({row['case_id'] for row in maps})!=6 or not all(row['state_unchanged'] for row in maps):raise RuntimeError('Verificacion de los seis casos Grad-CAM incompleta.')
    for row in maps:
        if not (output/'gradcam'/(row['case_id']+'_mapas.npz')).is_file():raise RuntimeError('Faltan los mapas guardados: '+row['case_id'])
    return state

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['train','reproduce'],default='train')
    parser.add_argument('--input-root',type=Path,default=Path('/kaggle/input'))
    parser.add_argument('--working-root',type=Path,default=Path('/kaggle/working'))
    args=parser.parse_args()
    args.working_root.mkdir(parents=True,exist_ok=True)
    gpu=gpu_and_python();ensure_light_dependencies()
    if not (HERE/'private/SHA256SUMS.json').is_file():raise RuntimeError('Falta el paquete privado junto al codigo; utilizar el ZIP completo proporcionado.')
    data=discover_data(args.input_root)
    config=make_config(data,args.working_root)
    config_path=HERE/'config.kaggle.local.json';config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    os.environ['TEMP']=os.environ['TMP']=str(args.working_root/'temporales_rsna')
    Path(os.environ['TEMP']).mkdir(exist_ok=True)
    os.environ['TORCH_HOME']=str(args.working_root/'torch_cache')
    print('GPU:',gpu,'| datos y configuracion detectados; iniciando comprobaciones.',flush=True)
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=HERE,check=True)
    subprocess.run([sys.executable,'ejecutar_pruebas.py'],cwd=HERE/'text_engine',check=True)
    subprocess.run([sys.executable,'-m','unittest','test_evaluador_textual','-v'],cwd=HERE/'text_engine',check=True)
    results=Path(config['output_root']);before=set(results.iterdir()) if results.exists() else set()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    log=args.working_root/f'comando_unico_{stamp}.log'
    command=[sys.executable,'-u','pipeline.py','--config',str(config_path),'--mode',args.mode]
    with log.open('w',encoding='utf-8') as capture:
        process=subprocess.Popen(command,cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:print(line,end='',flush=True);capture.write(line);capture.flush()
        code=process.wait()
    if code:raise RuntimeError(f'El pipeline termino con codigo {code}; conservar el registro {log}.')
    added=set(results.iterdir())-before
    if len(added)!=1:raise RuntimeError('No se pudo identificar una unica salida nueva.')
    output=added.pop();state=check_complete(output,args.mode)
    (output/'comando_unico.log').write_bytes(log.read_bytes())
    (output/'configuracion_kaggle.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    archive=args.working_root/f'Resultados_Comando_Unico_REV2_{args.mode}_{stamp}.zip'
    print('Entrenamientos y evaluaciones completos; empaquetando evidencias y checkpoints...',flush=True)
    files=sorted(p for p in output.rglob('*') if p.is_file())
    manifest={p.relative_to(output).as_posix():sha(p) for p in files}
    (output/'SHA256SUMS_entrega.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in [*files,output/'SHA256SUMS_entrega.json']:z.write(p,p.relative_to(output).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise RuntimeError('Error de integridad del ZIP final.')
    print('COMPROBACION COMPLETA:',state['estado'],'| nuevo entrenamiento:',state['nuevo_entrenamiento'],flush=True)
    print('DESCARGAR:',archive,flush=True)

if __name__=='__main__':main()
