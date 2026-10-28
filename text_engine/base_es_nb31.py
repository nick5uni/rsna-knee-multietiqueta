"""Extractor español candidato. Sin reglas ajustadas con GOLD; no congelado."""
import re, unicodedata
from hashlib import sha256
from correcciones_contexto import saltos_continuacion

CONDICIONES=('ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA',
 'Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture')
VERSION='ES_motor_candidato_v0_2_reportes_completos'
PATRONES={
 'ACL':r'\b(?:ligamento cruzado anterior|lca)\b',
 'MCL':r'\b(?:ligamento (?:colateral (?:medial|interno)|lateral interno)|lcm)\b',
 'Medial Meniscus':r'\bmenisco (?:medial|interno)\b',
 'Lateral Meniscus':r'\bmenisco (?:lateral|externo)\b',
 'Medial OA':r'\b(?:artrosis|osteoartrosis|osteoartritis|gonartrosis|oa)\b[^.;\n]{0,65}?\b(?:medial|intern[oa])\b',
 'Lateral OA':r'\b(?:artrosis|osteoartrosis|osteoartritis|gonartrosis|oa)\b[^.;\n]{0,65}?\b(?:lateral|extern[oa])\b',
 'PF OA':r'\b(?:artrosis|osteoartrosis|osteoartritis|gonartrosis|oa)\b[^.;\n]{0,65}?\b(?:patelofemoral|femoropatelar|femororrotulian[oa]|pf)\b',
 'Effusion':r'\b(?:derrame|hidrartrosis)\b',
 'Synovitis':r'\bsinovitis\b',
 "Baker's":r'\b(?:quistes?\s+(?:de\s+)?baker|quistes?\s+popliteos?)\b',
 'Contusion':r'\bcontusion(?:es)?\s+oseas?\b',
 'Fracture':r'\b(?:fracturas?|microfracturas?)\b',
}
for _c in ('Medial OA','Lateral OA','PF OA'):
    PATRONES[_c] = '(?:' + PATRONES[_c] + r'|\b(?:artrosis|osteoartrosis|osteoartritis|oa)\s+tricompartimental\b)'

LESION=r'\b(?:(?:rotura|ruptura|desgarro|esguince)s?|lesion(?:es)?)\b'
DUDA=r'\b(?:posible|probable|descartar|sospecha de|sugestiv[oa] de|sugiere|sugiriendo|no se descarta|no puede descartarse|no se puede descartar|a descartar|compatible con|en favor de|podria|sin criterios categoricos|sin signos categoricos|sin evidente|sin evidencia concluyente)\b'
NEG=r'\b(?:no hay|sin evidencia de|sin signos de|sin|ausencia de|no se observa[n]?|no se identifica[n]?|se descarta|no presenta|no)\b'
NORMAL=r'\b(?:normal(?:es)?|integr[oa]s?|conservad[oa]s?|sin alteraciones)\b'
TEMPORAL=r'\b(?:antigu[oa]s?|consolidad[oa]s?|reconstruid[oa]s?|reconstruccion|injerto|postoperatorio|postquirurgico|antecedente de|historia de|cicatricial(?:es)?)\b'
HEADER=r'\b(antecedentes clinicos|antecedentes|motivo de consulta|indicacion|tecnica|hallazgos|resultados|impresion|conclusion(?:es)?|diagnostico)\s*:'
NO_DIAG={'antecedentes clinicos','antecedentes','motivo de consulta','indicacion','tecnica'}

def normalizar_con_offsets(texto):
    letras=[];indices=[]
    for i,c in enumerate(texto):
        for v in unicodedata.normalize('NFD',c.lower()):
            if not unicodedata.combining(v):letras.append(v);indices.append(i)
    return ''.join(letras),indices

def extraer_es(texto):
    if not isinstance(texto,str) or not texto.strip():raise ValueError('Reporte vacío.')
    norm,indices=normalizar_con_offsets(texto)
    uniones=saltos_continuacion(texto)
    norm=''.join(' ' if c=='\n' and indices[i] in uniones else c for i,c in enumerate(norm))
    cab=list(re.finditer(HEADER,norm));secciones=[]
    if not cab:secciones=[('sin_encabezado',0,len(norm))]
    else:
        if cab[0].start():secciones.append(('sin_encabezado',0,cab[0].start()))
        for i,m in enumerate(cab):secciones.append((m.group(1),m.end(),cab[i+1].start() if i+1<len(cab) else len(norm)))
    eventos={c:[] for c in CONDICIONES};ignorados={c:[] for c in CONDICIONES}
    def emitir(c,estado,regla,a,b,seccion):
        # Trim whitespace in normalized coordinates; map back to original bytes as text offsets.
        while a<b and norm[a].isspace():a+=1
        while b>a and norm[b-1].isspace():b-=1
        if a>=b:return
        inicio,fin=indices[a],indices[b-1]+1
        evento=dict(estado=estado,regla_id=regla,seccion=seccion,
                    fragmento_literal=texto[inicio:fin],offsets_originales=[inicio,fin])
        destino=ignorados if seccion in NO_DIAG else eventos
        if evento not in destino[c]:destino[c].append(evento)
    for seccion,ini,fin in secciones:
        # Clause boundaries end negation scope; 'sin' remains within clause for postposed negation.
        cortes=list(re.finditer(r'[.;!?\n]+|\\n|\b(?:pero|aunque|sin embargo|asociad[oa] a)\b',norm[ini:fin]))
        limites=[ini];tramos=[]
        for m in cortes:
            tramos.append((limites[-1],ini+m.start()));limites.append(ini+m.end())
        tramos.append((limites[-1],fin))
        for a,b in tramos:
            unidad=norm[a:b]
            if not unidad.strip():continue
            menciones=[]
            for c,pat in PATRONES.items():
                for m in re.finditer(pat,unidad):menciones.append((m.start(),m.end(),c))
            # Generic anatomical names delimit scope but never create labels themselves.
            anclas=[(m.start(),m.end()) for m in re.finditer(r'\b(?:lcp|lcl|ligamento cruzado posterior|ligamento colateral lateral|tendon\w*|menisco (?:medial|interno|lateral|externo))\b',unidad)]
            anclas+= [(x,y) for x,y,_ in menciones]
            for x,y,c in menciones:
                anteriores=[e for s,e in anclas if e<=x]
                siguientes=[s for s,e in anclas if s>=y]
                izq=max(anteriores,default=0);der=min(siguientes,default=len(unidad))
                pref=unidad[izq:x];suf=unidad[y:der]
                local=unidad[izq:der]
                # Coordinated 'ni' carries an immediately preceding negative list.
                if anteriores and re.search(r'\bni\b', pref):
                    # La lista con ni comparte el operador previo, sin cruzar cláusulas.
                    izq=0
                    pref=unidad[:x]
                    local=unidad[:der]
                elif re.match(r'^\s*(?:,\s*)?y\s*$',pref) and anteriores:
                    izq=0
                    pref=unidad[:x]
                    local=unidad[:der]
                duda=bool(re.search(DUDA,pref+' '+suf))
                neg_pre=bool(re.search(NEG,pref))
                neg_post=bool(re.search(NEG,suf))
                normal=bool(re.search(NORMAL,local))
                if c in CONDICIONES[:4]:
                    lesion_pre=bool(re.search(LESION,pref))
                    lesion_post=bool(re.search(LESION,suf))
                    if duda:estado,regla='incierto','ES_DUDA_ESTRUCTURA'
                    elif re.search(TEMPORAL,local):estado,regla='incierto','ES_TEMPORAL_PENDIENTE'
                    elif (neg_pre and lesion_pre) or (neg_post and lesion_post):estado,regla='negativo','ES_NEGACION_LESION'
                    elif normal:
                        if re.search(r'\bno\s*(?:es\s*)?(?:normal|integr[oa]|conservad[oa])',local):estado,regla='incierto','ES_NORMALIDAD_NEGADA'
                        else:estado,regla='negativo','ES_NORMALIDAD_ESTRUCTURA'
                    elif lesion_pre or lesion_post:estado,regla='positivo','ES_LESION_EXPLICITA'
                    else:estado,regla='incierto','ES_MENCION_SIN_DECISION'
                else:
                    if duda:estado,regla='incierto','ES_DUDA_HALLAZGO'
                    elif re.search(TEMPORAL,local) or re.search(r'\b(?:no|sin)\b[^.;]{0,30}\b(?:agud[oa]|significativ[oa]|definitiv[oa])',local):estado,regla='incierto','ES_CALIFICACION_PENDIENTE'
                    elif c=='Fracture' and re.search(r'\bmenisc\w*\b',local):estado,regla='incierto','ES_ANATOMIA_FRACTURA_PENDIENTE'
                    elif neg_pre or re.match(r'\s*(?:ausente|no presente|no se observa|no se identifica)',suf):estado,regla='negativo','ES_NEGACION_HALLAZGO'
                    else:estado,regla='positivo','ES_HALLAZGO_EXPLICITO'
                emitir(c,estado,regla,a+izq,a+der,seccion)
            # Shared normality supports only the named ligament/meniscus findings.
            if re.search(NORMAL,unidad):
                comunes=[]
                if re.search(r'\bligamentos cruzados\b',unidad):comunes.append('ACL')
                if re.search(r'\bligamentos (?:cruzados y )?colaterales\b',unidad):comunes.append('MCL')
                if re.search(r'\b(?:ambos meniscos|meniscos)\b',unidad):comunes+=['Medial Meniscus','Lateral Meniscus']
                for c in comunes:emitir(c,'incierto' if re.search(DUDA,unidad) or re.search(r'\bno\s+(?:son\s+)?normales',unidad) else 'negativo','ES_NORMALIDAD_COMPARTIDA',a,b,seccion)
    salida=[]
    for c in CONDICIONES:
        ev=eventos[c];estados={e['estado'] for e in ev}
        if 'incierto' in estados or {'positivo','negativo'}<=estados:estado='incierto'
        elif estados:estado=next(iter(estados))
        else:estado='no_mencionado'
        evidencia=ev or ignorados[c]
        # NB28 accepts one literal segment; all events and contradictory spans remain in extra fields.
        principal=next((e for e in evidencia if e['estado']=='incierto'),evidencia[0] if evidencia else None)
        regla='ES_CONTRADICCION' if {'positivo','negativo'}<=estados else (principal['regla_id'] if ev else 'ES_NO_MENCION_DIAGNOSTICA')
        salida.append(dict(condicion=c,estado=estado,regla_id=regla,
            evidencia=principal['fragmento_literal'] if principal else '',
            seccion=principal['seccion'] if principal else '',
            offsets_originales=principal['offsets_originales'] if principal else None,
            eventos=ev,menciones_no_diagnosticas=ignorados[c],version_reglas=VERSION,
            valor={'positivo':1,'negativo':0}.get(estado),mascara=int(estado in ('positivo','negativo'))))
    return salida
