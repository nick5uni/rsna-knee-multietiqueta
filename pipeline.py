"""Punto de entrada: extracción textual, matrices, imágenes, modelos y evaluación.

reproduce: usa checkpoints congelados, sin entrenar ni ajustar con GOLD.
train: repite las cuatro configuraciones con el motor y presupuesto acordados.
audit: comprueba contratos y recalcula métricas de CSV; no es inferencia DICOM.
"""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,platform,sys,zipfile
import numpy as np

HERE=Path(__file__).resolve().parent
LABELS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
KEYS=['CNN_BCE','ResNet18_BCE','ResNet18_FOCAL','ResNet18_BCE_FILTRO']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def resolve(config,key):
    p=Path(config[key]);return p if p.is_absolute() else HERE/p
def emit(message):print(message,flush=True)
def table(path,uids,pred,series,split):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['StudyInstanceUID','SeriesInstanceUID','split',*LABELS]);w.writeheader()
        for u,p in zip(uids,pred):w.writerow(dict(StudyInstanceUID=u,SeriesInstanceUID=series[u],split=split,**dict(zip(LABELS,map(float,p)))))

def verify(config):
    private=resolve(config,'private_inputs')
    for n,h in read(HERE/'codigo_congelado_sha256.json').items():
        if sha(HERE/n)!=h:raise ValueError('Código congelado modificado: '+n)
    for n,h in read(private/'SHA256SUMS.json').items():
        if sha(private/n)!=h:raise ValueError('Insumo privado modificado: '+n)
    from src.rev2_integration import review_bundles,activate
    source=private/'insumos_REV2';filt=private/'insumos_filtro_REV2'
    closure=read(private/'cierre_comun.json')
    assert closure['estado'].startswith('cierre_comun_registrado')
    excluded='1.2.826.0.1.3680043.8.498.52010826440192227660457107121818389761'
    base,filtered,manifest,gold,parent,policy=review_bundles(source,filt,excluded_uid=excluded)
    release,filtered_release=activate(source,filt,base,filtered,parent,common_confirmed=True)
    reports={r['StudyInstanceUID']:r for r in rows(resolve(config,'data_csv'))}
    assert sha(resolve(config,'data_csv'))=='8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33'
    assert sha(resolve(config,'data_series_csv'))=='573c1d80772bf41211c91b149c95677385a1c22d63f485c347f1b46c0177aef3'
    assert not set(base.study_uids)&gold
    return private,base,filtered,manifest,gold,release,filtered_release,reports

def textual(private,reports,out):
    sys.path.insert(0,str(HERE/'text_engine'))
    from motor_textual_comun import MotorTextualComun,VERSION,CONDICIONES
    from adaptador_bilingue import AdaptadorBilingue
    from evaluador_textual import evaluar_texto
    folder=private/'text_gold';closure=read(folder/'cierre_utilizado.json')
    for n,h in closure['artefactos_sha256'].items():assert sha(HERE/'text_engine'/n)==h
    motor=MotorTextualComun(HERE/'text_engine');adapter=AdaptadorBilingue()
    for lang in ['en','es']:adapter.registrar(lang,VERSION,getattr(motor,'extraer_'+lang))
    references=[];annotations=[]
    for row in rows(folder/'Confirmacion_Idiomas_GOLD.csv'):
        uid=row['StudyInstanceUID'];lang=row['idioma_confirmado'];text=reports[uid]['Report']
        assert hashlib.sha256(text.encode()).hexdigest()==row['reporte_sha256']
        references.extend(dict(StudyInstanceUID=uid,condicion=c,gold=int(float(reports[uid][c])),idioma=lang) for c in CONDICIONES)
        if lang in ['en','es']:annotations.extend(adapter.procesar(uid,text,lang))
    assert annotations==read(folder/'anotaciones_textuales_GOLD.json')
    result=evaluar_texto(references,annotations,closure)
    assert result==read(folder/'resultados_textuales_GOLD.json')
    save(out/'texto/anotaciones.json',annotations);save(out/'texto/metricas.json',result)
    emit('Texto EN/ES: 456 pares, 197 resueltos, 164 acuerdos; abstenciones conservadas.')
    return result

def audit_predictions(private,base,out):
    from src.evaluador_comun_v2 import metrics,load_predictions
    gold=rows(private/'gold/labels_gold_referencia.csv');uids=[r['StudyInstanceUID'] for r in gold]
    y=np.asarray([[float(r[c]) for c in LABELS] for r in gold]);mask=np.ones_like(y,dtype=bool)
    expected=read(private/'gold/metricas_gold_comunes_v2.json')['modelos'];got={}
    for name in KEYS:
        pred=load_predictions(private/'gold'/f'predictions_gold_{name}.csv',uids,LABELS,'gold')
        if isinstance(pred,tuple):pred=pred[0]
        got[name]=metrics(y,pred,mask,LABELS)
        assert got[name]==expected[name],name
    save(out/'metricas_desde_csv.json',got)
    return uids,y,mask,expected

def ensure_images(config,manifests):
    """Extrae solo las series del manifiesto; rechaza rutas inseguras y CRC erróneos."""
    cache=resolve(config,'cache_root');cache.mkdir(parents=True,exist_ok=True)
    root=Path(config['data_root']).resolve() if config.get('data_root') else cache/'dataset'
    needed={f"train_series/{r['StudyInstanceUID']}/{r['SeriesInstanceUID']}":int(r['instances']) for r in manifests}
    incomplete={p:n for p,n in needed.items() if len(list((root/p).glob('*.dcm')))!=n}
    if incomplete:
        archive=resolve(config,'dataset_zip')
        if not archive.is_file():raise FileNotFoundError('Falta ZIP autorizado o data_root con las series completas.')
        emit(f'Extracción selectiva: {len(incomplete)} series; no se extrae el dataset completo.')
        with zipfile.ZipFile(archive) as z:
            by_series={k:[] for k in incomplete}
            for info in z.infolist():
                if info.filename.endswith('.dcm'):
                    parent=info.filename.rsplit('/',1)[0]
                    if parent in by_series:by_series[parent].append(info)
            for i,(folder,items) in enumerate(by_series.items(),1):
                if len(items)!=incomplete[folder]:raise ValueError('Conteo de serie distinto: '+folder)
                for info in items:
                    target=(root/info.filename).resolve()
                    if not target.is_relative_to(root.resolve()):raise ValueError('Ruta ZIP fuera de destino.')
                    target.parent.mkdir(parents=True,exist_ok=True)
                    # Releer y comprobar CRC incluso si el archivo ya existía.
                    raw=z.read(info)
                    if not target.is_file() or sha(target)!=hashlib.sha256(raw).hexdigest():target.write_bytes(raw)
                if i%20==0 or i==len(by_series):emit(f'Series listas: {i}/{len(by_series)}')
    return root

def inputs(root,manifest,references):
    from src.study_reader import read_study
    ref={r['StudyInstanceUID']:r for r in references};cache={};series={}
    for i,r in enumerate(manifest,1):
        uid=r['StudyInstanceUID'];sid=r['SeriesInstanceUID'];expected=ref[uid]
        for f in expected['archivos_seleccionados']:
            path=root/f['member'];assert path.stat().st_size==f['bytes'] and sha(path)==f['sha256']
        image,mask,meta=read_study(root/'train_series'/uid/sid,uid,sid)
        assert meta['indices_espaciales_1based']==expected['indices_espaciales_1based']
        cache[uid]=(image,mask);series[uid]=sid
        if i%20==0 or i==len(manifest):emit(f'Entradas DICOM verificadas: {i}/{len(manifest)}')
    return cache,series

def predict(model,uids,cache,device,policy):
    import torch
    from src.batchnorm_policy import evaluation_forward
    model.eval();result=[]
    with torch.inference_mode():
        for uid in uids:
            x,m=cache[uid]
            z=evaluation_forward(model,torch.from_numpy(x[None]).to(device),torch.from_numpy(m[None]).to(device),policy)
            result.append(torch.sigmoid(z)[0].cpu().numpy())
    return np.asarray(result,dtype=np.float64)

def model_for(name,imagenet=False):
    from src.study_cnn import StudyCNN
    from src.study_resnet import StudyResNet18
    return StudyCNN() if name=='CNN_BCE' else StudyResNet18(weights='imagenet' if imagenet else None)

def restore(private,name,out,device):
    import torch
    entry=next(r for r in read(private/'modelos.json') if r['model']==name)
    archive=private/entry['archive'];assert sha(archive)==entry['sha256']
    with zipfile.ZipFile(archive) as z:
        summary=read_zip_json(z,'corrida/run_summary.json')
        raw=z.read('corrida/best.pt');assert hashlib.sha256(raw).hexdigest()==summary['sha256']['best.pt']
        reference=z.read('corrida/predictions_dev.csv')
        assert hashlib.sha256(reference).hexdigest()==summary['sha256']['predictions_dev.csv']
        history=read_zip_json(z,'corrida/history.json')
        assert min(history,key=lambda r:r['dev']['masked_bce'])['epoch']==summary['selected_epoch']
    p=out/'checkpoints'/name/'best.pt';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    (p.parent/'reference_predictions_dev.csv').write_bytes(reference)
    # Solo checkpoints locales aprobados por SHA-256. Contienen estado RNG/optimizador.
    checkpoint=torch.load(p,map_location='cpu',weights_only=False)
    assert checkpoint['label_order']==LABELS and checkpoint['epoch']==summary['selected_epoch']
    model=model_for(name).to(device);model.load_state_dict(checkpoint['model'],strict=True);model.eval()
    return model,summary

def read_zip_json(z,n):return json.loads(z.read(n).decode('utf-8-sig'))

def inference(config,private,base,filtered,manifest,release,filtered_release,gold_uids,y,gmask,expected,out,train=False):
    import torch
    from src.batchnorm_policy import POLICY_ID
    from src.evaluador_comun_v2 import metrics,load_predictions
    torch.set_num_threads(int(config.get('cpu_threads',4)))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device(config.get('device','cpu'))
    goldmanifest=read(private/'gold/manifest_gold_v1_revision.json')
    dev_uids=base.training_view('dev')['StudyInstanceUID'];devset=set(dev_uids)
    selected=manifest if train else [r for r in manifest if r['StudyInstanceUID'] in devset]
    root=ensure_images(config,selected+goldmanifest)
    refs=read(private/'insumos_REV2/referencias_lectura300.json')+read(private/'gold/referencias_lectura_gold_v1.json')
    cache,series=inputs(root,selected+goldmanifest,refs)
    got={};checks={};trained={}
    for name in KEYS:
        emit('Modelo: '+name)
        policy=None if name=='CNN_BCE' else POLICY_ID
        if train:
            if device.type!='cuda' or not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(device):
                raise RuntimeError('Reentrenar con config.device=cuda:0 en Tesla T4; no sustituir la referencia silenciosamente.')
            from dataclasses import replace
            from functools import partial
            from src.training_engine import StudyDataset,fit,seed_everything
            from src.training_contract import TrainConfig
            from src.masked_losses import masked_focal
            bundle=filtered if name.endswith('FILTRO') else base;rel=filtered_release if name.endswith('FILTRO') else release
            datasets={s:StudyDataset(bundle,s,root,__import__('src.study_reader',fromlist=['read_study']).read_study,max_cache_studies=300) for s in ['train','dev']}
            for ds in datasets.values():
                for u in ds.view['StudyInstanceUID']:ds.cache[u]=cache[u]
            cfg=TrainConfig(seed=42,batch_size=1,learning_rate=1e-4,max_epochs=20,patience=3,loss_name='focal' if name.endswith('FOCAL') else 'bce')
            seed_everything(42);model=model_for(name,imagenet=True)
            criterion=partial(masked_focal,gamma=2.,alpha=None) if name.endswith('FOCAL') else None
            kwargs={'train_criterion':criterion} if criterion else {}
            trained[name]=fit(model,datasets['train'],datasets['dev'],out/'entrenamiento'/name,cfg,{'version':'REV2','supervision_sha256':bundle.provenance,'reproduccion':'corrida nueva; GOLD no usado para seleccionar'},device=str(device),release=rel,bundle=bundle,evaluation_policy=policy,**kwargs)
            model.eval()
        else:model,summary=restore(private,name,out,device)
        probs=predict(model,gold_uids,cache,device,policy)
        got[name]=metrics(y,probs,gmask,LABELS)
        table(out/f'predictions_gold_{name}.csv',gold_uids,probs,series,'gold')
        if not train:
            old=load_predictions(private/'gold'/f'predictions_gold_{name}.csv',gold_uids,LABELS,'gold')
            delta=float(np.max(np.abs(probs-old)))
            if delta>2e-4:raise ValueError(f'Diferencia de probabilidades mayor que tolerancia declarada: {name} {delta}')
            checks[name]={'max_abs_probability_delta_gold':delta,'selected_epoch':summary['selected_epoch']}
        devprob=predict(model,dev_uids,cache,device,policy)
        table(out/f'predictions_dev_{name}.csv',dev_uids,devprob,series,'dev')
        if not train:
            olddev=load_predictions(out/'checkpoints'/name/'reference_predictions_dev.csv',dev_uids,LABELS,'dev')
            devdelta=float(np.max(np.abs(devprob-olddev)))
            if devdelta>2e-4:raise ValueError(f'Diferencia DEV mayor que tolerancia declarada: {name} {devdelta}')
            checks[name]['max_abs_probability_delta_dev']=devdelta
            checks[name]['selected_epoch_matches_history']=True
        view=base.training_view('dev')
        save(out/f'metrics_dev_{name}.json',metrics(view['targets'],devprob,view['label_mask'],LABELS))
        if name=='ResNet18_FOCAL':
            from src.gradcam_study import gradcam
            from src.gradcam_figures import render_case,render_overview
            plan=read(private/'plan_gradcam.json');examples=[];camchecks=[]
            for case in plan['cases']:
                x,m=cache[case['StudyInstanceUID']]
                result=gradcam(model,model.backbone.layer4[-1],torch.from_numpy(x[None]).to(device),torch.from_numpy(m[None]).to(device),case['class_index'],POLICY_ID)
                render_case(out/'gradcam',case,x,result);examples.append((case,x,result,m))
                camchecks.append({'case_id':case['case_id'],'state_unchanged':True,'zero_positive_map':result['zero_positive_map']})
            render_overview(out/'gradcam/resumen.png',examples);save(out/'gradcam/verificacion.json',camchecks)
        del model
    save(out/'metrics_gold.json',got);save(out/'comparacion_recarga.json',checks)
    if train:save(out/'entrenamiento_resumen.json',trained)
    return checks

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=HERE/'config.local.json')
    parser.add_argument('--mode',choices=['audit','reproduce','train'],default='reproduce')
    args=parser.parse_args();config=read(args.config)
    root=resolve(config,'output_root');root.mkdir(parents=True,exist_ok=True)
    out=root/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir()
    emit('Salida privada: '+str(out))
    try:
        private,base,filtered,manifest,gold,release,fr,reports=verify(config)
        emit('Contratos REV2: 240 TRAIN, 60 DEV; 1458/1295 TRAIN y 364 DEV.')
        save(out/'matrices_verificadas.json',{'train':240,'dev':60,'known_train':1458,'known_train_filtered':1295,'known_dev':364,'excluded_train':163,'gold_excluded':58})
        textual(private,reports,out)
        gu,y,gm,expected=audit_predictions(private,base,out)
        checks=inference(config,private,base,filtered,manifest,release,fr,gu,y,gm,expected,out,train=args.mode=='train') if args.mode!='audit' else None
        versions={'Python':platform.python_version(),'numpy':np.__version__}
        if args.mode!='audit':
            import torch,torchvision,pydicom,PIL
            versions.update(torch=str(torch.__version__),torchvision=str(torchvision.__version__),pydicom=pydicom.__version__,Pillow=PIL.__version__,CUDA=torch.version.cuda)
        save(out/'ejecucion.json',{'estado':'OK','mode':args.mode,'versiones':versions,'inferencia_DICOM':args.mode!='audit','nuevo_entrenamiento':args.mode=='train','gold':'exploratorio; sin ajuste de reglas, umbral o checkpoint con GOLD','salida':str(out),'checks':checks})
        emit('OK: '+args.mode)
    except Exception as e:
        save(out/'ejecucion.json',{'estado':'ERROR','mode':args.mode,'tipo':type(e).__name__,'error':str(e)});raise
if __name__=='__main__':main()
