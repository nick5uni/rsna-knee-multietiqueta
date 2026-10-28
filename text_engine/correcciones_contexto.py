"""Correcciones de lectura; conserva offsets originales y no usa GOLD."""
import re

def saltos_continuacion(texto):
    posiciones=set()
    for m in re.finditer(r'\n(?=[ \t]*[a-záéíóúñü])',texto):
        previo=texto[:m.start()].rsplit('\n',1)[-1].rstrip()
        if previo and previo[-1] not in '.;:!?':posiciones.add(m.start())
    return posiciones

def vista_continua(texto):
    posiciones=saltos_continuacion(texto)
    return ''.join(' ' if i in posiciones else c for i,c in enumerate(texto))

def ampliar_eventos_ingles(texto,filas,tramos,crear_evento,targets):
    por={r['condicion']:r for r in filas}
    def agregar(c,sec,a,b,pat,regla,estado_base):
        trozo=vista_continua(texto[a:b])
        for m in re.finditer(pat,trozo,re.I):
            izquierda=re.split(r'[.;!?\n]',trozo[:m.start()])[-1]
            contexto=izquierda+' '+m.group()
            duda=re.search(r'\b(?:possible|possibly|suspected|suspicious|questionable|equivocal|cannot exclude|R/O|old|healed|history of|status post|chronic)\b',contexto,re.I)
            neg=re.search(r'\b(?:no|not|without)\s+(?:evidence of\s+)?$',izquierda,re.I)
            if duda:estado='incierto'
            elif neg:estado='negativo' if estado_base=='positivo' else 'incierto'
            else:estado=estado_base
            # Guardar también el prefijo que determina la polaridad.
            x=m.start()-len(izquierda) if duda or neg else m.start()
            e=crear_evento(texto,a+x,a+m.end(),estado,regla,sec)
            if e not in por[c]['eventos']:por[c]['eventos'].append(e)
    for sec,a,b,diag in tramos:
        if not diag:continue
        for c in ('ACL','MCL','Medial Meniscus','Lateral Meniscus'):
            t=r'\b(?:'+targets[c]+r')\b'
            agregar(c,sec,a,b,t+r'\s+(?:appears? to be\s+|is\s+|are\s+)?within normal(?: limits)?\b','EN_NORMALIDAD_WITHIN','negativo')
            pref=r'\b(?:tear|rupture|disruption)(?:\s+(?:of|the|posterior|anterior|horn|body|root|and|posteromedial|corner)){0,14}\s+'
            agregar(c,sec,a,b,pref+t,'EN_LESION_ESTRUCTURA_EXTENDIDA','positivo')
        par=r'\bno\s+(?:evidence of\s+)?tear\s+(?:is noted\s+)?in either medial or lateral meniscus\b'
        for c in ('Medial Meniscus','Lateral Meniscus'):
            agregar(c,sec,a,b,par,'EN_NEGACION_AMBOS_MENISCOS','negativo')
        for c in ('Medial Meniscus','Lateral Meniscus'):
            agregar(c,sec,a,b,r'\bno convincing medial or lateral meniscal tears\b','EN_AUSENCIA_CALIFICADA_MENISCOS','incierto')
        for c in ('Medial OA','Lateral OA','PF OA'):
            agregar(c,sec,a,b,r'\b(?:OA|osteoarthritis) changes of all three compartments\b','EN_OA_TRES_COMPARTIMENTOS_EXPLICITA','positivo')
        agregar('MCL',sec,a,b,r'\bmedial collateral ligament and posterior oblique ligament\s+(?:are\s+)?intact\b','EN_NORMALIDAD_LISTA_EXTENDIDA','negativo')
        agregar('MCL',sec,a,b,r'\bboth medial\s*(?:&|and)\s*lateral collateral ligaments\s+(?:appear to be\s+|are\s+)?within normal(?: limits)?\b','EN_NORMALIDAD_COLATERALES_COMPARTIDA','negativo')
    return filas
