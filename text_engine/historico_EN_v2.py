# DEFINIR REGLAS V2 Y CONSERVAR LAS CITAS ORIGINALES.
import re, json

CONDICIONES = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
TARGETS = {
 'ACL':r'(?:ACL|anterior cruciate ligaments?)',
 'MCL':r'(?:MCL|medial collateral ligament(?:ous)?(?: complexes?|s)?)',
 'Medial Meniscus':r'medial menisc(?:us|i)',
 'Lateral Meniscus':r'lateral menisc(?:us|i)',
 'Effusion':r'(?:(?:knee|joint)\s+)?effusion',
 'Synovitis':r'synovitis',
 "Baker's":r'(?:Baker[\x27’]?s?|popliteal)\s+cyst',
 'Contusion':r'(?:bone\s+(?:contusion|bruise)|osseous\s+contusion)',
 'Fracture':r'fracture',
 'Medial OA':r'medial\s+(?:compartment|femorotibial|tibiofemoral)',
 'Lateral OA':r'lateral\s+(?:compartment|femorotibial|tibiofemoral)',
 'PF OA':r'(?:patellofemoral|femoropatellar)'
}
NORMAL=r'(?:normal|intact|unremarkable)'
ATTR=r'(?:tear(?:ing)?|torn|ruptur(?:e|ed)|sprain|injury)'
MOD=r'(?:(?:low|high)[ -]grade\s+|(?:mild|moderate|severe|partial|complete|complex|degenerative|acute|almost|unstable|small|radial|free[ -]edge|horizontal|longitudinal)\s+)*'
LINK=r'\s*(?:\([A-Z]{2,4}\)\s*)?(?:[:,;]\s*)?(?:(?:is|are|appears?|remains?|grossly)\s+)*'
STRUCT=r'(?:ACL|PCL|MCL|LCL|anterior cruciate ligament|posterior cruciate ligament|medial collateral ligament(?: complexes?)?|lateral collateral ligament(?: complexes?)?|medial meniscus|lateral meniscus|articular cartilage)'
LIST=rf'{STRUCT}(?:\s*(?:,|and|&)\s*{STRUCT})*'
UNC=r'\b(?:possible|possibly|suspected|suspicious|questionable|equivocal|cannot exclude|could represent|may represent|rule out|R/O)\b'

def _normalizar(texto):
    # Track offsets so quotes always come from the unchanged source.
    caracteres=[]; limites=[]
    for m in re.finditer(r'\s+|\S',texto):
        caracteres.append(' ' if m.group(0).isspace() else m.group(0))
        limites.append((m.start(),m.end()))
    return ''.join(caracteres), limites

def extraer_reporte_v2(texto):
    norm, offsets=_normalizar(texto)
    eventos={c:[] for c in CONDICIONES}

    def emitir(c, estado, regla, m, contexto=None):
        a,b=m.span()
        previo=norm[max(0,a-90):a]
        previo=re.split(r'[.;:]',previo)[-1]
        if estado=='ausente' and 'normal' in regla and re.search(r'\b(?:not|no|without)\s*$',previo,re.I):
            estado='incierto'; regla+='__normalidad_negada'
        # These cues suppress an automatic binary, but do not imply a diagnosis.
        if re.search(UNC,previo+' '+m.group(0),re.I):
            estado='incierto'; regla+='__incertidumbre'
        if re.search(r'\b(?:history of|status post|reconstruction|graft|healed|old fracture|indication|clinical history)\b',previo+' '+m.group(0),re.I):
            estado='incierto'; regla+='__contexto_por_revisar'
        cita=texto[offsets[a][0]:offsets[b-1][1]]
        assert cita in texto
        ev={'estado':estado,'regla':regla,'cita':cita}
        if contexto is not None:
            x,y=contexto.span()
            ev['contexto_literal']=texto[offsets[x][0]:offsets[y-1][1]]
        if ev not in eventos[c]: eventos[c].append(ev)

    def buscar(c,estado,regla,patron):
        for m in re.finditer(patron,norm,re.I): emitir(c,estado,regla,m)

    # Explicit names with explicit status; negation is linked to the finding.
    for c in ['ACL','MCL','Medial Meniscus','Lateral Meniscus']:
        t=rf'\b(?:{TARGETS[c]})\b'
        buscar(c,'ausente','normalidad_directa',rf'\b{NORMAL}\s+{t}|{t}{LINK}{NORMAL}\b')
        buscar(c,'ausente','no_rotura_directa',rf'{t}{LINK}(?:no\s+(?:evidence of\s+)?{ATTR}|(?:is\s+)?not\s+torn)\b')
        buscar(c,'ausente','sin_rotura_directa',rf'\b(?:no|without)\s+(?:evidence of\s+)?{ATTR}\s+of\s+(?:the\s+)?{t}')
        for m in re.finditer(rf'{t}{LINK}{MOD}{ATTR}\b|\b{MOD}{ATTR}\s+(?:of|involving|at)\s+(?:the\s+)?(?:anterior horn|posterior horn|body|root)?(?:\s+(?:and|to)\s+(?:body|root|posterior root))?(?:\s+of\s+(?:the\s+)?)?{t}',norm,re.I):
            left=re.split(r'[.;:]',norm[max(0,m.start()-65):m.start()])[-1]
            if re.search(r'\b(?:no|without|not)\b',left,re.I):
                emitir(c,'incierto','lesion_negacion_proxima',m)
            else: emitir(c,'presente','lesion_directa',m)

    # Shared lists and paired anatomy, requiring an explicit shared normal status.
    for m in re.finditer(rf'\b{NORMAL}\s+{LIST}|\b{LIST}{LINK}{NORMAL}\b',norm,re.I):
        for c in ['ACL','MCL','Medial Meniscus','Lateral Meniscus']:
            if re.search(rf'\b{TARGETS[c]}\b',m.group(0),re.I): emitir(c,'ausente','lista_normal',m)
    for m in re.finditer(r'\b(?:anterior\s+and\s+posterior|cruciate)\s+ligaments?\s*(?:are\s+)?(?:normal|intact)\b|\b(?:normal|intact)\s+(?:anterior\s+and\s+posterior|cruciate)\s+ligaments?\b',norm,re.I):
        emitir('ACL','ausente','cruzados_normales',m)
    for m in re.finditer(r'\b(?:medial\s+and\s+lateral|both)\s+menisci\s*(?:are\s+)?(?:normal|intact|unremarkable)\b|\b(?:normal|intact)\s+(?:medial\s+and\s+lateral|both)\s+menisci\b',norm,re.I):
        for c in ['Medial Meniscus','Lateral Meniscus']: emitir(c,'ausente','ambos_meniscos_normales',m)

    # Exact structured templates. Normal compartment does not automatically label OA.
    encabezados={
      'ACL':r'Intercondylar compartment\s*\(anterior and posterior cruciate ligaments\)',
      'Medial Meniscus':r'Medial compartment\s*\(meniscus, collateral ligament, cartilage\)',
      'MCL':r'Medial compartment\s*\(meniscus, collateral ligament, cartilage\)',
      'Lateral Meniscus':r'Lateral compartment\s*\(meniscus, collateral ligament complex, cartilage\)'
    }
    for c,h in encabezados.items(): buscar(c,'ausente','seccion_normal_explicita',rf'\b{h}\s*:\s*Normal\b')

    # Implicit meniscus only inside an explicit compartment block.
    cab=r'\b(?:In\s+the\s+|The\s+)?(?P<lado>medial|lateral)\s+compartment\s*[,:(]'
    stop=r'\b(?:In\s+the\s+|The\s+)?(?:medial compartment|lateral compartment|patellofemoral joint|intercondylar notch)\b'
    for h in re.finditer(cab,norm,re.I):
        rest=norm[h.end():]; fin=re.search(stop,rest,re.I)
        end=h.end()+(fin.start() if fin else min(len(rest),650))
        block=norm[h.end():end]
        c='Medial Meniscus' if h.group('lado').lower()=='medial' else 'Lateral Meniscus'
        for local in re.finditer(r'\bthe meniscus\s+(?:is\s+)?not torn\b|\bthe meniscus\s*;\s*(?:complex\s+)?tear\b',block,re.I):
            class _Match:
                def span(self): return (h.end()+local.start(),h.end()+local.end())
                def group(self,n=0): return local.group(n)
            emitir(c,'ausente' if 'not torn' in local.group(0).lower() else 'presente','menisco_contexto_compartimento',_Match(),h)

    # Findings with explicit absence, uncertainty, or affirmative mention.
    for c in ['Effusion','Synovitis',"Baker's",'Contusion','Fracture']:
        t=rf'\b{TARGETS[c]}\b'
        for m in re.finditer(t,norm,re.I):
            left=re.split(r'[.;:]',norm[max(0,m.start()-90):m.start()])[-1]
            right=norm[m.end():m.end()+45]
            # Build a larger literal quote containing the modifier/negation.
            prefix=re.search(r'\b(?:no|without|absence of)\s+(?:evidence of\s+)?(?:an?\s+)?(?:acute\s+)?$',left,re.I)
            qualified=re.search(r'\bno\s+(?:significant|definite)\s*$',left,re.I)
            negative_tail=re.match(r'\s*(?::\s*)?(?:is\s+)?(?:absent|not present|none)\b',right,re.I)
            a=m.start()-(len(left)-prefix.start() if prefix else 0)
            b=m.end()+(negative_tail.end() if negative_tail else 0)
            class _Finding:
                def span(self): return a,b
                def group(self,n=0): return norm[a:b]
            if qualified: emitir(c,'incierto','ausencia_calificada',_Finding())
            elif prefix or negative_tail: emitir(c,'ausente','ausencia_directa',_Finding())
            elif re.search(r'\b(?:no|without|not)\b',left,re.I): emitir(c,'incierto','negacion_alcance_no_resuelto',_Finding())
            else: emitir(c,'presente','mencion_afirmativa',_Finding())

    # OA only when explicitly named and anatomically linked. No cartilage proxies.
    oa=r'(?:osteoarthritis|OA)'
    for c in ['Medial OA','Lateral OA','PF OA']:
        t=TARGETS[c]
        # Do not match affirmative OA inside an explicitly negated phrase.
        for m in re.finditer(rf'\b{t}\s+{oa}\b|\b{oa}\s+(?:changes\s+)?(?:of|in|at)\s+(?:the\s+)?{t}\b',norm,re.I):
            prev=re.split(r'[.;:]',norm[max(0,m.start()-50):m.start()])[-1]
            if re.search(r'\b(?:no|without)\s+(?:evidence of\s+)?$',prev,re.I):
                emitir(c,'ausente','sin_oa_directa',m)
            else: emitir(c,'presente','oa_compartimento_directo',m)
        buscar(c,'ausente','sin_oa_compartimento',rf'\b(?:no|without)\s+(?:evidence of\s+)?{t}\s+{oa}\b')
        buscar(c,'presente','oa_tricompartimental',rf'\btricompartmental\s+{oa}\b')

    out=[]
    for c in CONDICIONES:
        ev=eventos[c]; states={e['estado'] for e in ev}
        if 'presente' in states and 'ausente' in states: estado='incierto'; motivo='contradiccion'
        elif 'incierto' in states: estado='incierto'; motivo='incertidumbre_o_alcance'
        elif states: estado=next(iter(states)); motivo='reglas_explicitas'
        elif re.search(rf'\b{TARGETS[c]}\b',norm,re.I): estado='incierto'; motivo='mencion_sin_regla'
        else: estado='no_mencionado'; motivo='sin_mencion_reconocida'
        value={'presente':'1','ausente':'0'}.get(estado,'')
        out.append({'condicion':c,'estado':estado,'valor_propuesto':value,'mascara_propuesta':int(value!=''),'motivo':motivo,'evidencias_json':json.dumps(ev,ensure_ascii=False),'fuente_evidencia':'texto_para_extraccion','version_extractor':'reglas_en_v2','validacion':'pendiente'})
    return out
