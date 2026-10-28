"""Comparación descriptiva de anotaciones textuales congeladas con referencia.
Sin entrenamiento, ajuste de reglas ni conversión de abstenciones a negativos.
"""
import re
from collections import Counter, defaultdict

CONDICIONES = ('ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA',
 'Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture')
ESTADOS = ('positivo','negativo','incierto','no_mencionado')

def evaluar_texto(gold, anotaciones, congelacion, idiomas=('en','es')):
    """gold: registros UID, condicion, gold (0/1), idioma confirmado.
    anotaciones: registros del adaptador NB28 para TODOS los pares del alcance.
    congelacion: declaración de versión fijada y SHA256 de los artefactos.
    El llamador verifica la correspondencia de esos hashes con archivos reales.
    """
    if congelacion.get('reglas_congeladas') is not True:
        raise ValueError('Falta congelar las reglas antes de evaluar.')
    if congelacion.get('gold_consultado_previamente') is not True:
        raise ValueError('Debe declararse la consulta previa de GOLD.')
    if not congelacion.get('fecha_congelacion_utc'):
        raise ValueError('Falta fecha de congelación.')
    hashes=congelacion.get('artefactos_sha256',{})
    if not hashes or any(not isinstance(v,str) or not re.fullmatch('[a-f0-9]{64}',v) for v in hashes.values()):
        raise ValueError('Faltan hashes válidos de los artefactos congelados.')
    if not idiomas or len(set(idiomas))!=len(idiomas):
        raise ValueError('Alcance de idiomas vacío o duplicado.')
    referencia={}; por_estudio=defaultdict(set); lengua={}
    for r in gold:
        uid,c=r['StudyInstanceUID'],r['condicion']; par=(uid,c)
        if not isinstance(uid,str) or not uid.strip() or c not in CONDICIONES:
            raise ValueError('Identificador o condición inválidos.')
        if par in referencia or r['gold'] not in (0,1) or not r.get('idioma'):
            raise ValueError('Referencia duplicada, inválida o sin idioma confirmado.')
        if uid in lengua and lengua[uid]!=r['idioma']:
            raise ValueError('Idioma inconsistente en un estudio.')
        referencia[par]=dict(r);por_estudio[uid].add(c);lengua[uid]=r['idioma']
    if not referencia or any(len(v)!=12 for v in por_estudio.values()):
        raise ValueError('Se requieren las doce referencias por estudio.')
    objetivo={p:r for p,r in referencia.items() if r['idioma'] in idiomas}
    if not objetivo:
        raise ValueError('No hay referencias en los idiomas elegidos.')
    pred={}; versiones=defaultdict(set)
    for r in anotaciones:
        par=(r['StudyInstanceUID'],r['condicion'])
        if par in pred or par not in objetivo:
            raise ValueError('Anotación duplicada o fuera del alcance.')
        if r.get('idioma')!=objetivo[par]['idioma'] or r.get('estado') not in ESTADOS:
            raise ValueError('Idioma o estado de anotación inválidos.')
        if not r.get('version_extractor') or not r.get('regla_id'):
            raise ValueError('Falta versión o regla del extractor.')
        esperado={'positivo':1,'negativo':0}.get(r['estado'])
        if r.get('valor_candidato')!=esperado:
            raise ValueError('Valor incompatible con el estado; no completar desconocidos.')
        if r.get('habilitada_entrenamiento') is not False:
            raise ValueError('La evaluación aislada no habilita entrenamiento.')
        pred[par]=dict(r);versiones[r['idioma']].add(r['version_extractor'])
    if set(pred)!=set(objetivo):
        raise ValueError('Faltan pares del alcance. Deben existir como abstenciones explícitas.')
    if any(len(v)!=1 for v in versiones.values()):
        raise ValueError('Se mezclaron versiones de extractor dentro de un idioma.')
    filas=[]
    def resumir(pares, idioma, condicion):
        estados=Counter(pred[p]['estado'] for p in pares)
        resueltos=[p for p in pares if pred[p]['estado'] in ('positivo','negativo')]
        tp=sum(objetivo[p]['gold']==1 and pred[p]['valor_candidato']==1 for p in resueltos)
        tn=sum(objetivo[p]['gold']==0 and pred[p]['valor_candidato']==0 for p in resueltos)
        fp=sum(objetivo[p]['gold']==0 and pred[p]['valor_candidato']==1 for p in resueltos)
        fn=sum(objetivo[p]['gold']==1 and pred[p]['valor_candidato']==0 for p in resueltos)
        n=len(pares);k=len(resueltos)
        return dict(idioma=idioma,condicion=condicion,estudios=n if condicion!='TODAS' else n//12,
            pares=n,gold_positivos=sum(objetivo[p]['gold']==1 for p in pares),
            gold_negativos=sum(objetivo[p]['gold']==0 for p in pares),
            resueltos=k,positivos_extraidos=estados['positivo'],negativos_extraidos=estados['negativo'],
            inciertos=estados['incierto'],no_mencionados=estados['no_mencionado'],
            abstenciones=n-k,cobertura=k/n if n else None,acuerdos=tp+tn,desacuerdos=fp+fn,
            tasa_acuerdo_resueltos=(tp+tn)/k if k else None,
            tasa_desacuerdo_resueltos=(fp+fn)/k if k else None,TP=tp,TN=tn,FP=fp,FN=fn)
    for idioma in idiomas:
        for condicion in CONDICIONES:
            pares=[p for p,r in objetivo.items() if r['idioma']==idioma and p[1]==condicion]
            filas.append(resumir(pares,idioma,condicion))
    totales=[resumir(list(objetivo),'ALCANCE','TODAS')]
    for idioma in idiomas:
        pares=[p for p,r in objetivo.items() if r['idioma']==idioma]
        totales.append(resumir(pares,idioma,'TODAS'))
    return dict(estado='evaluacion_textual_descriptiva',por_idioma_condicion=filas,totales=totales,
        estudios_gold_fuente=len(por_estudio),estudios_en_alcance=len({p[0] for p in objetivo}),
        estudios_fuera_alcance=dict(Counter(v for v in lengua.values() if v not in idiomas)),
        versiones={k:list(v)[0] for k,v in versiones.items()},congelacion=dict(congelacion),
        nota='Acuerdo/desacuerdo solo sobre pares resueltos; cobertura sobre todos los pares del alcance. Consulta previa de GOLD: no es validación independiente ni validación clínica.')
