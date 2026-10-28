"""Pares artificiales; no son juicios clínicos ni reportes GOLD."""
PARES=[
 ('Cartílago del compartimento medial normal.','Medial compartment cartilage is normal.','Medial OA','negativo','indirecto_OA'),
 ('Cartílago del compartimento lateral conservado.','Lateral compartment cartilage is preserved.','Lateral OA','negativo','indirecto_OA'),
 ('Cartílago femoropatelar normal.','Patellofemoral cartilage is normal.','PF OA','negativo','indirecto_OA'),
 ('Condromalacia rotuliana.','Patellar chondromalacia.','PF OA','incierto',None),
 ('Condromalacia del compartimento medial.','Medial compartment chondromalacia.','Medial OA','incierto',None),
 ('Condromalacia.','Chondromalacia.','PF OA','no_mencionado',None),
 ('Sin fractura aguda.','No acute fracture.','Fracture','incierto',None),
 ('Compatible con fractura tibial.','Compatible with tibial fracture.','Fracture','incierto',None),
 ('Antecedentes: fractura tibial.','Clinical history: tibial fracture.','Fracture','no_mencionado',None),
 ('Antecedentes: fractura. Hallazgos: fractura tibial.','History: fracture. Findings: tibial fracture.','Fracture','positivo',None),
 ('Hallazgos: sin fractura.','Findings: no fracture.','Fracture','negativo',None),
 ('Hallazgos: fractura tibial. Conclusión: sin fractura.','Findings: tibial fracture. Impression: no fracture.','Fracture','incierto',None),
 ('Antecedentes: fractura. Hallazgos: sin fractura.','History: fracture. Findings: no fracture.','Fracture','negativo',None),
 ('Cartílago del compartimento medial normal excepto un defecto.','Medial compartment cartilage is normal except for a defect.','Medial OA','incierto_o_no_mencionado',None),
 ('Cartílago del compartimento medial normal. Artrosis del compartimento medial.','Medial compartment cartilage is normal. Medial compartment osteoarthritis.','Medial OA','positivo',None),
 ('Cartílago del compartimento medial normal. Condromalacia del compartimento medial.','Medial compartment cartilage is normal. Medial compartment chondromalacia.','Medial OA','incierto',None),
 ('Cartílago normal.','Normal cartilage.','Medial OA','no_mencionado',None),
 ('Cartílago del compartimento medial no normal.','Medial compartment cartilage is not normal.','Medial OA','incierto_o_no_mencionado',None),
 ('Técnica: cartílago del compartimento medial normal.','Technique: medial compartment cartilage is normal.','Medial OA','no_mencionado',None),
 ('No hay derrame.','No effusion.','Effusion','negativo',None),
 ('Sinovitis.','Synovitis.','Synovitis','positivo',None),
 ('Antecedentes: sinovitis. Hallazgos: sin derrame.','History: synovitis. Findings: no effusion.','Synovitis','no_mencionado',None),
 ('Cartílago del compartimento medial normal. Cartílago del compartimento lateral con pérdida.','Medial compartment cartilage is normal. Lateral compartment cartilage loss.','Medial OA','negativo','indirecto_OA'),
 ('Cartílago del compartimento medial normal. Pérdida de cartílago.','Medial compartment cartilage is normal. Cartilage loss.','Medial OA','incierto_o_no_mencionado',None),
]
def comprobar(motor,adaptador):
    resultados=[]
    for i,(es,en,c,esp,origen) in enumerate(PARES,1):
        for idioma,t in [('es',es),('en',en)]:
            filas=adaptador.procesar('ART'+str(i),t,idioma)
            por={r['condicion']:r for r in filas};r=por[c]
            opciones=('incierto','no_mencionado') if esp=='incierto_o_no_mencionado' else (esp,)
            assert r['estado'] in opciones,(i,idioma,t,esp,r['estado'])
            if origen:assert r['origen_decision']==origen,(i,idioma,r)
            assert len(filas)==12
            for x in filas:
                assert not x['habilitada_entrenamiento']
                assert x['valor_candidato']=={'positivo':1,'negativo':0}.get(x['estado'])
                for e in x['eventos']+x['menciones_no_diagnosticas']+x['bloqueos_indirectos']:
                    a,b=e['offsets_originales'];assert t[a:b]==e['fragmento_literal']
            if i==12:assert {e['estado'] for e in r['eventos']}=={'positivo','negativo'}
            if i==9:assert r['menciones_no_diagnosticas'] and not r['eventos']
            resultados.append(dict(par=i,idioma=idioma,condicion=c,estado=r['estado'],esperado=esp,correcto=True))
    return resultados
