"""Enlace REV2 y mascara efectiva. Sin PyTorch; no modifica matrices fuente."""
from pathlib import Path
from dataclasses import replace
from copy import deepcopy
import json,hashlib
import numpy as np
from .label_adapter import load_label_bundle,read_table
from .data_checks import LABELS
from .training_contract import require_release
from .run_budget import require_run_budget

VERSION='REV2_consenso168_localizado_18Oct2026_v1'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def json_bytes(d):return (json.dumps(d,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def review_bundles(folder,filtered_folder,*,excluded_uid):
    folder=Path(folder);filtered_folder=Path(filtered_folder)
    source=read_json(folder/'release.json')
    if source['labels_version']!=VERSION:raise ValueError('Se requiere supervision REV2.')
    manifest=read_json(folder/'manifest.json');groups=read_json(folder/'groups.json')
    gold=set(read_json(folder/'ids_gold_excluidos.json'))
    if len(gold)!=58:raise ValueError('Referencia gold distinta.')
    base=load_label_bundle(folder/'labels.csv',folder/'masks.csv',folder/'annotations.csv',manifest,
        gold_uids=gold,excluded_uids={excluded_uid},partitions_path=folder/'partitions.json',groups_by_study=groups)
    if source['supervision_sha256']!=base.provenance:raise ValueError('Release fuente no coincide con matrices.')
    if base.available_mask.sum()!=1822 or base.approved_mask.sum()!=1822:raise ValueError('Conteo base REV2 distinto.')
    train,dev=base.training_view('train'),base.training_view('dev')
    if len(train['indices'])!=240 or len(dev['indices'])!=60 or train['label_mask'].sum()!=1458 or dev['label_mask'].sum()!=364:raise ValueError('Particiones/cobertura REV2 distintas.')
    confirmation=read_json(folder/'confirmacion_Nicolas_REV2_y_filtro.json')
    if digest(filtered_folder/'politica_filtro_REV2.json')!=confirmation['filtro_politica_sha256'] or digest(filtered_folder/'masks_filtradas.csv')!=confirmation['filtro_sha256']:
        raise ValueError('Politica o mascara distintas de las contrastadas por Nicolas.')
    policy=read_json(filtered_folder/'politica_filtro_REV2.json')
    if len(policy['nuevos_indirectos_consenso'])!=10 or len({(x['StudyInstanceUID'],x['condicion']) for x in policy['nuevos_indirectos_consenso']})!=10:
        raise ValueError('Lista de nuevos indirectos incompleta o duplicada.')
    entries=read_table(filtered_folder/'masks_filtradas.csv',['StudyInstanceUID','SeriesInstanceUID',*LABELS])
    mask_by_uid={r['StudyInstanceUID']:r for r in entries}
    if len(entries)!=300 or len(mask_by_uid)!=300 or set(mask_by_uid)!=set(base.study_uids):raise ValueError('Mascara efectiva no cubre los mismos 300 estudios.')
    effective=np.zeros_like(base.approved_mask)
    for i,u in enumerate(base.study_uids):
        row=mask_by_uid[u]
        if row['SeriesInstanceUID']!=base.series_uids[i]:raise ValueError('Serie distinta en mascara efectiva.')
        for j,c in enumerate(LABELS):
            if row[c] not in ('0','1'):raise ValueError('Mascara efectiva no binaria.')
            effective[i,j]=row[c]=='1'
    if (effective & ~base.approved_mask).any():raise ValueError('Filtro agrega etiquetas no aprobadas.')
    actual={(u,c) for i,u in enumerate(base.study_uids) for j,c in enumerate(LABELS) if base.approved_mask[i,j] and not effective[i,j]}
    notes=read_table(folder/'annotations.csv',['StudyInstanceUID','condicion','origen_etiqueta','origen_etapa1'])
    expected={(a['StudyInstanceUID'],a['condicion']) for a in notes if
        (a['origen_etiqueta']=='negativo_indirecto_OA' or a['origen_etapa1']=='negativo_indirecto_OA') and base.splits[a['StudyInstanceUID']]=='train'}
    if len(expected)!=155:raise ValueError('Los 155 indirectos historicos no coinciden.')
    additions={(x['StudyInstanceUID'],x['condicion']) for x in policy['nuevos_indirectos_consenso'] if base.splits[x['StudyInstanceUID']]=='train'}
    if len(additions)!=8 or expected & additions:raise ValueError('Los 8 nuevos indirectos no coinciden.')
    if actual!=expected|additions or len(actual)!=163:raise ValueError('Exclusiones diferentes de la politica REV2.')
    lookup={(u,c):base.targets[i,j] for i,u in enumerate(base.study_uids) for j,c in enumerate(LABELS)}
    if any(base.splits[u]!='train' or c not in ('Medial OA','Lateral OA','PF OA') or lookup[u,c]!=0 for u,c in actual):raise ValueError('El filtro debe retirar exclusivamente negativos OA de train.')
    provenance=dict(base.provenance,mascaras=digest(filtered_folder/'masks_filtradas.csv'))
    # available_mask conserva lo conocido en la fuente; approved_mask define lo usado en esta variante.
    filtered=replace(base,approved_mask=effective.copy(),provenance=provenance)
    ft,fd=filtered.training_view('train'),filtered.training_view('dev')
    if ft['StudyInstanceUID']!=train['StudyInstanceUID'] or ft['label_mask'].sum()!=1295:raise ValueError('TRAIN filtrado distinto.')
    if fd['StudyInstanceUID']!=dev['StudyInstanceUID'] or not np.array_equal(fd['targets'],dev['targets']) or not np.array_equal(fd['label_mask'],dev['label_mask']):raise ValueError('DEV fue modificado.')
    return base,filtered,manifest,gold,source,policy

def release_proposal(base,filtered,source,folder,filtered_folder):
    folder=Path(folder);filtered_folder=Path(filtered_folder)
    common=deepcopy(source)
    common.update(protocol_accepted=True,labels_release_accepted=True,training_enabled=True,
        estado='cierre_comun_aplicable_solo_tras_confirmacion_de_notebooks',pendientes=[],
        supervision_sha256=base.provenance,
        parent_release_sha256=digest(folder/'release.json'),
        codigo_integracion_sha256=digest(Path(__file__)),
        condicion_activacion='confirmacion_explicita_version_comun_en_celda4; no_autoaprobacion',
        filtro_REV2={'estado':'criterio_y_matrices_contrastados_por_Nicolas','exclusiones_train':163,
           'politica_sha256':digest(filtered_folder/'politica_filtro_REV2.json'),
           'mascara_sha256':digest(filtered_folder/'masks_filtradas.csv'),'dev':'364 conocidas; comun'},
        evidencia_contraste_REV2_y_filtro_sha256=digest(folder/'confirmacion_Nicolas_REV2_y_filtro.json'))
    filtered_release=deepcopy(common)
    filtered_release.update(labels_version=VERSION+'_FILTRO_TRAIN163',supervision_sha256=filtered.provenance,
        parent_release_sha256=hashlib.sha256(json_bytes(common)).hexdigest(),
        known_train_effective=1295,known_dev_effective=364,
        alcance_mascara='approved_mask efectiva; targets y available_mask conservan fuente REV2',
        training_mask_sha256=digest(filtered_folder/'masks_filtradas.csv'))
    return common,filtered_release

def activate(folder,filtered_folder,base,filtered,source,*,common_confirmed):
    if common_confirmed is not True:raise ValueError('Falta confirmacion de comprobacion de notebooks y cierre comun; detener antes de entrenar.')
    common,fr=release_proposal(base,filtered,source,folder,filtered_folder)
    require_release(base,common);require_release(filtered,fr)
    return common,fr

def budget_for_release(original,release):
    out=deepcopy(original)
    out.update(version='presupuesto_REV2_mismos_parametros_18Oct2026',
        parent_budget_sha256=hashlib.sha256(json_bytes(original)).hexdigest(),
        release_sha256=hashlib.sha256(json_bytes(release)).hexdigest(),
        alcance_acuerdo='Parametros originales conservados; referencias REV2 sujetas al cierre comun de notebooks.',
        evaluacion_gold='Evaluacion posterior exploratoria; gold ya consultado.',
        filtro_confianza='Solo train de ResNet18_BCE_FILTRO: 163 exclusiones; dev comun 364.')
    config=require_run_budget(out,effective_release_sha256=out['release_sha256'],model='CNN_BCE')
    if config.validate()!={'seed':42,'batch_size':1,'learning_rate':0.0001,'max_epochs':20,'patience':3,'loss_name':'bce'}:raise ValueError('Cambio no acordado del presupuesto.')
    return out,config
