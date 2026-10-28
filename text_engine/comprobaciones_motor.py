"""Pruebas adicionales propuestas por Víctor, sin reportes GOLD."""
CASOS_ADICIONALES=[
 ('Artrosis tricompartimental.','Medial OA','positivo'),
 ('Artrosis tricompartimental.','Lateral OA','positivo'),
 ('Artrosis tricompartimental.','PF OA','positivo'),
 ('Rotura del LCA sin traslación tibial anterior.','ACL','positivo'),
 ('Contusiones óseas.','Contusion','positivo'),
 ('No sinovitis.','Synovitis','negativo'),
 ('Compatible con fractura tibial.','Fracture','incierto'),
 ('En favor de fractura tibial.','Fracture','incierto'),
 ('Menisco medial degenerado.','Medial Meniscus','incierto'),
 ('Edema óseo.','Contusion','no_mencionado'),
 ('Condromalacia rotuliana.','PF OA','no_mencionado'),
 ('Cartílago del compartimento medial normal.','Medial OA','no_mencionado'),
 ('Sin fractura aguda.','Fracture','incierto'),
 ('Fractura antigua consolidada.','Fracture','incierto'),
 ('Rotura del ligamento colateral lateral.','MCL','no_mencionado'),
 ('Quiste subcondral.','Baker\'s','no_mencionado'),
 ('Artrosis femorotibial medial.','Lateral OA','no_mencionado'),
 ('No hay derrame. Sinovitis.','Synovitis','positivo'),
 ('Antecedentes: sinovitis. Hallazgos: sin derrame.','Synovitis','no_mencionado'),
 ('Hallazgos: Fractura. Impresión: sin fractura.','Fracture','incierto'),
 ('Técnica: sospecha de rotura meniscal. Hallazgos: sinovitis.','Synovitis','positivo'),
]

def comprobar_motor(extraer, adaptador, casos_nicolas):
    resultados=[]
    for caso in casos_nicolas:
        filas=extraer(caso['texto']);estados={r['condicion']:r['estado'] for r in filas}
        fallos={c:{'esperado':esp,'obtenido':estados[c]} for c,esp in caso['esperado'].items() if estados[c]!=esp}
        # Check values, masks and every literal evidence offset, not just the target state.
        assert len(filas)==12 and len(estados)==12
        for r in filas:
            assert r['valor']=={'positivo':1,'negativo':0}.get(r['estado'])
            assert r['mascara']==int(r['estado'] in ('positivo','negativo'))
            for evento in r['eventos']+r['menciones_no_diagnosticas']:
                a,b=evento['offsets_originales']
                assert caso['texto'][a:b]==evento['fragmento_literal']
        salida=adaptador.procesar(caso['id'],caso['texto'],'es')
        assert len(salida)==12 and not any(r['habilitada_entrenamiento'] for r in salida)
        resultados.append(dict(id=caso['id'],correcto=not fallos,fallos=fallos))
    for i,(texto,c,esperado) in enumerate(CASOS_ADICIONALES,1):
        filas=extraer(texto);r=next(r for r in filas if r['condicion']==c)
        fallos={} if r['estado']==esperado else {c:{'esperado':esperado,'obtenido':r['estado']}}
        resultados.append(dict(id=f'VIC{i:03}',correcto=not fallos,fallos=fallos))
    # Offsets remain literal with accented text, line breaks, and a synthetic ligature.
    for texto in ['Hallazgos:\nContusión ósea.','Hallazgos:\\nContusión ósea.','Hallazgos: posible lesión del LCA.']:
        for r in extraer(texto):
            for e in r['eventos']:
                a,b=e['offsets_originales'];assert texto[a:b]==e['fragmento_literal']
    return resultados
