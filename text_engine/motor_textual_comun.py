"""Versión textual candidata separada; no aplica revisiones manuales REV2 a nuevos casos."""
import re,json,types
from pathlib import Path
from copy import deepcopy
from correcciones_contexto import ampliar_eventos_ingles
from oa_compartida import normalidad_compartida
VERSION='EN_ES_textual_v1_2_OA_compartida_candidata'
CONDICIONES=('ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture')
OA=CONDICIONES[4:7]

def cargar_base(carpeta):
    carpeta=Path(carpeta)
    es=types.ModuleType('base_es_nb31');exec(compile((carpeta/'base_es_nb31.py').read_text(encoding='utf-8'),'base_es_nb31.py','exec'),es.__dict__)
    fuente=(carpeta/'historico_EN_v2.py').read_text(encoding='utf-8')
    origen="ev={'estado':estado,'regla':regla,'cita':cita}"
    assert fuente.count(origen)==1,'Interfaz histórica inesperada'
    # Instrumentación de una copia en memoria para conservar offsets exactos.
    fuente=fuente.replace(origen,"ev={'estado':estado,'regla':regla,'cita':cita,'offsets_originales':[offsets[a][0],offsets[b-1][1]]}")
    en=types.ModuleType('en_v2_instrumentado');exec(compile(fuente,'en_v2_instrumentado','exec'),en.__dict__)
    return es,en

def secciones(texto,idioma):
    if idioma=='es':
        cab=r'\b(antecedentes clinicos|antecedentes|motivo de consulta|indicacion|tecnica|hallazgos|resultados|impresion|conclusiones?|diagnostico)\s*:'
        no_diag={'antecedentes clinicos','antecedentes','motivo de consulta','indicacion','tecnica'}
    elif idioma=='en':
        cab=r'\b(clinical history|history|indication|technique|findings|results|impression|conclusions?|coclusion|diagnosis|procedure|comparison)\s*:'
        no_diag={'clinical history','history','indication','technique','procedure','comparison'}
    else:raise ValueError('Idioma fuera de alcance')
    norm=''.join(__import__('unicodedata').normalize('NFD',c.lower())[0] for c in texto)
    # Los encabezados usan letras latinas ordinarias y acentuadas de un carácter.
    matches=list(re.finditer(cab,norm));tramos=[]
    if not matches:return [('sin_encabezado',0,len(texto),True)]
    if matches[0].start():tramos.append(('sin_encabezado',0,matches[0].start(),True))
    for i,m in enumerate(matches):
        tramos.append((m.group(1),m.end(),matches[i+1].start() if i+1<len(matches) else len(texto),m.group(1) not in no_diag))
    if idioma=='en':
        ajustados=[]
        for sec,a,b,diag in tramos:
            # Plantilla literal observada: técnica breve seguida inmediatamente por hallazgos.
            m=re.match(r'\s*MRI of (?:the )?knee\.\s*(?=(?:ACL|MCL|medial meniscus|lateral meniscus|cartilages?\b|no (?:effusion|baker))\b)',texto[a:b],re.I) if sec=='technique' else None
            if m:
                ajustados.extend([(sec,a,a+m.end(),False),('hallazgos_plantilla_no_rotulada',a+m.end(),b,True)])
            else:ajustados.append(('conclusion' if sec=='coclusion' else sec,a,b,diag))
        tramos=ajustados
    return tramos

def evento(texto,a,b,estado,regla,seccion,origen='explicito'):
    while a<b and texto[a].isspace():a+=1
    while b>a and texto[b-1].isspace():b-=1
    return dict(estado=estado,regla_id=regla,seccion=seccion,fragmento_literal=texto[a:b],offsets_originales=[a,b],origen=origen)

def unidades(texto,tramos):
    for sec,a,b,diag in tramos:
        for m in re.finditer(r'[^.!?\n]+(?:[.!?]|$)',texto[a:b]):
            if m.group().strip():yield sec,a+m.start(),a+m.end(),diag

def cerrar(filas):
    for r in filas:
        todos=r['eventos']
        fallback={'ES_MENCION_SIN_DECISION','EN_V2_MENCION_SIN_REGLA'}
        binario=any(e['estado'] in ('positivo','negativo') for e in todos)
        for e in todos:
            e['participa_en_decision']=not (binario and e['regla_id'] in fallback)
        ev=[e for e in todos if e['participa_en_decision']]
        estados={e['estado'] for e in ev}
        if 'incierto' in estados or {'positivo','negativo'}<=estados:estado='incierto'
        elif estados:estado=next(iter(estados))
        else:estado='no_mencionado'
        principal=next((e for e in ev if e['estado']=='incierto'),next(iter(ev or r['menciones_no_diagnosticas']),None))
        r.update(estado=estado,valor={'positivo':1,'negativo':0}.get(estado),mascara=int(estado in ('positivo','negativo')),
                 regla_id='TEXT_CONTRADICCION' if {'positivo','negativo'}<=estados else (principal['regla_id'] if ev else 'TEXT_NO_MENCION_DIAGNOSTICA'),
                 evidencia=principal['fragmento_literal'] if principal else '',seccion=principal['seccion'] if principal else '',
                 offsets_originales=principal['offsets_originales'] if principal else None,version_reglas=VERSION,
                 origen_decision='indirecto_OA' if ev and all(e.get('origen')=='indirecto_OA' for e in ev) else ('sin_decision' if not ev else 'texto'))
    return filas

class MotorTextualComun:
    def __init__(self,carpeta):self.es,self.en=cargar_base(carpeta)
    def extraer_en(self,texto):
        if not isinstance(texto,str) or not texto.strip():raise ValueError('Reporte vacío')
        filas={c:dict(condicion=c,eventos=[],menciones_no_diagnosticas=[],bloqueos_indirectos=[]) for c in CONDICIONES}
        for sec,a,b,diag in secciones(texto,'en'):
            trozo=texto[a:b]
            if not trozo.strip():continue
            registros=self.en.extraer_reporte_v2(trozo)
            for r in registros:
                c=r['condicion'];evs=json.loads(r['evidencias_json'])
                for e in evs:
                    x,y=e['offsets_originales'];estado={'presente':'positivo','ausente':'negativo','incierto':'incierto'}[e['estado']]
                    regla='EN_V2_'+e['regla']
                    # Amplía el contexto local sin traspasar final de oración.
                    izq=max(trozo.rfind(k,0,x) for k in '.!?\n')+1
                    finales=[p for k in '.!?\n' if (p:=trozo.find(k,y))>=0]
                    der=min(finales,default=len(trozo));ctx=trozo[izq:der]
                    if c=='Fracture' and (re.search(r'\b(?:no|without)\s+(?:evidence of\s+)?(?:an?\s+)?acute\s+fracture\b',ctx,re.I) or re.search(r'\bcompatible with\s+(?:a\s+)?(?:\w+[ -]){0,4}fracture\b',ctx,re.I)):
                        estado='incierto';regla='TEXT_FRACTURA_CALIFICADA'
                        x,y=izq,der
                    ev=evento(texto,a+x,a+y,estado,regla,sec)
                    if 'contexto_literal' in e:ev['contexto_literal_historico']=e['contexto_literal']
                    # Antecedente sin encabezado formal: solo la mención histórica local.
                    historia=bool(re.search(r'\b(?:history of|status post)\s+(?:a\s+)?(?:\w+[ -]){0,3}fracture\b',ctx,re.I)) if c=='Fracture' else False
                    filas[c]['eventos' if diag and not historia else 'menciones_no_diagnosticas'].append(ev)
                if r['estado']=='incierto' and not evs:
                    for m in re.finditer(r'\b(?:'+self.en.TARGETS[c]+r')\b',trozo,re.I):
                        filas[c]['eventos' if diag else 'menciones_no_diagnosticas'].append(evento(texto,a+m.start(),a+m.end(),'incierto','EN_V2_MENCION_SIN_REGLA',sec))
        # Precauciones v3 reexpresadas como reglas puras, sin leer CSV ni propagar ajustes manuales.
        guardas={
          'Fracture':[r"\b(?:suspect(?:ed)?|possible|possibly|suspicious for|cannot exclude|can't exclude|R/O)\s+(?:[\w'-]+[\s-]+){0,7}fracture\b",r'\bfracture\b[^.\n]{0,100}\bmenisc(?:us|al|i)\b'],
          'PF OA':[r'\bsuggest(?:ing|ive of)\s+(?:evolving\s+)?patellofemoral\s+osteoarthritis\b'],
          'Synovitis':[r'\bno\s+(?:evidence of\s+)?synovitis\b[^.\n]{0,80}\bIT\s*band\b']}
        for sec,a,b,diag in secciones(texto,'en'):
            if not diag:continue
            for c,pats in guardas.items():
                if not any(e['estado'] in ('positivo','negativo') for e in filas[c]['eventos']):continue
                for pat in pats:
                    for m in re.finditer(pat,texto[a:b],re.I):filas[c]['eventos'].append(evento(texto,a+m.start(),a+m.end(),'incierto','EN_GUARDA_V3',sec))
        integradas=ampliar_eventos_ingles(texto,list(filas.values()),secciones(texto,'en'),evento,self.en.TARGETS)
        return self.armonizar(texto,'en',integradas)
    def extraer_es(self,texto):
        filas=deepcopy(self.es.extraer_es(texto))
        for r in filas:
            r['bloqueos_indirectos']=[]
            for e in r['eventos']+r['menciones_no_diagnosticas']:e.setdefault('origen','explicito')
        return self.armonizar(texto,'es',filas)
    def armonizar(self,texto,idioma,filas):
        por={r['condicion']:r for r in filas}
        tramos=secciones(texto,idioma)
        normalizar=lambda s:__import__('unicodedata').normalize('NFD',s.lower()).encode('ascii','ignore').decode()
        if idioma=='es':
            scopes={'Medial OA':r'\b(?:compartimento (?:femorotibial )?(?:medial|interno)|femorotibial medial)\b','Lateral OA':r'\b(?:compartimento (?:femorotibial )?(?:lateral|externo)|femorotibial lateral)\b','PF OA':r'\b(?:patelofemoral|femoropatelar|femororrotulian[oa]|rotulian[oa])\b'}
            cartilage=r'\bcartilago\b';normal=r'\b(?:normal|conservad[oa]|preservad[oa])\b'
            chondro=r'\bcondromalacia\b'
            patologia=r'\b(?:artrosis|osteoartrosis|osteoartritis|gonartrosis|condromalacia|condropatia|osteofitos?|defectos?|adelgazamiento|perdida|fisuras?|irregularidad)\b'
            calificador=r'\b(?:posible|probable|sospecha|excepto|salvo|pero|aunque|resto|parcialmente|groseramente|significativ[oa]|antecedentes|no|sin|compatible|reconstruccion|postoperatorio)\b'
        else:
            scopes={'Medial OA':r'\bmedial (?:compartment|femorotibial|tibiofemoral)\b','Lateral OA':r'\blateral (?:compartment|femorotibial|tibiofemoral)\b','PF OA':r'\b(?:patellofemoral|femoropatellar|patellar)\b'}
            cartilage=r'\bcartilage\b';normal=r'\b(?:normal|unremarkable|(?:well[ -])?preserved|well[ -]maintained)\b'
            chondro=r'\bchondromalacia\b'
            patologia=r'\b(?:osteoarthritis|oa|chondromalacia|chondrosis|osteophytes?|defects?|thinning|loss|fissuring|irregularity)\b'
            calificador=r'\b(?:possible|probable|suspect\w*|except|otherwise|but|although|partially|grossly|significant|definite|history|no|not|without|compatible|postoperative|reconstruction)\b'
        units=[]
        for sec,a,b,diag in unidades(texto,tramos):
            s=normalizar(texto[a:b]);ubic={c for c,p in scopes.items() if re.search(p,s)}
            # Solo la forma conjunta explícita permite expandir a los tres compartimentos.
            if normalidad_compartida(s,idioma):ubic=set(OA)
            units.append((sec,a,b,diag,s,ubic))
        for sec,a,b,diag,s,ubic in units:
            if not diag:continue
            if re.search(chondro,s) and len(ubic)==1:
                c=next(iter(ubic))
                # Una negación de condromalacia no demuestra ni sospecha artrosis.
                if not re.search(r'\b(?:no|without|sin|ausencia)\b',s):
                    ev=por[c]['eventos']
                    # Sustituye solo las menciones OA sin regla; conserva diagnósticos explícitos.
                    por[c]['eventos']=[e for e in ev if e['regla_id']!='EN_V2_MENCION_SIN_REGLA']
                    if not any(e['estado'] in ('positivo','negativo') for e in por[c]['eventos']):
                        por[c]['eventos'].append(evento(texto,a,b,'incierto','TEXT_CONDROMALACIA_NO_EQUIVALE_OA',sec))
            conjunta=normalidad_compartida(s,idioma)
            if not (conjunta or (re.search(cartilage,s) and re.search(normal,s) and len(ubic)==1)):continue
            for c in OA if conjunta else sorted(ubic):
                r=por[c];bloqueos=[]
                if re.search(calificador,s):bloqueos.append(evento(texto,a,b,'incierto','TEXT_CALIFICADOR_BLOQUEA_INDIRECTO',sec))
                for sec2,x,y,diag2,s2,ubic2 in units:
                    if diag2 and re.search(patologia,s2) and (not ubic2 or c in ubic2):
                        bloqueos.append(evento(texto,x,y,'incierto','TEXT_PATOLOGIA_BLOQUEA_INDIRECTO',sec2))
                if any(e['estado'] in ('positivo','incierto') and e['regla_id']!='EN_V2_MENCION_SIN_REGLA' for e in r['eventos']):
                    bloqueos.append(evento(texto,a,b,'incierto','TEXT_EVIDENCIA_PREVIA_BLOQUEA_INDIRECTO',sec))
                r['bloqueos_indirectos'].extend(bloqueos)
                if bloqueos:continue
                # Eliminar incertidumbre por simple mención anatómica, sin descartar evidencias contradictorias.
                r['eventos']=[e for e in r['eventos'] if e['regla_id']!='EN_V2_MENCION_SIN_REGLA']
                r['eventos'].append(evento(texto,a,b,'negativo','TEXT_OA_CARTILAGO_NORMAL_CONJUNTO' if conjunta else 'TEXT_OA_CARTILAGO_NORMAL_COMPARTIMENTO',sec,origen='indirecto_OA'))
        return cerrar(filas)
