"""Regresiones artificiales derivadas de fallos de lectura, sin GOLD."""
CASOS=[
 ('es','El menisco lateral\nes de morfología y señal conservada, sin signos de rotura.','Lateral Meniscus','negativo'),
 ('es','Rotura compleja que compromete el cuerpo y cuerno posterior del\nmenisco medial.','Medial Meniscus','positivo'),
 ('es','Ligamentos cruzados y colaterales dentro de límites normales.','MCL','negativo'),
 ('es','Desgarro parcial del LCA y LCM.','MCL','positivo'),
 ('es','Menisco medial degenerado. Rotura del menisco medial.','Medial Meniscus','positivo'),
 ('es','Posible rotura del menisco medial. Rotura del menisco medial.','Medial Meniscus','incierto'),
 ('es','Menisco lateral normal. Rotura del menisco lateral.','Lateral Meniscus','incierto'),
 ('es','No hay derrame.\nsinovitis.','Synovitis','positivo'),
 ('es','No hay derrame\nSinovitis.','Synovitis','positivo'),
 ('es','Antecedentes: rotura del LCA. Hallazgos: LCA normal.','ACL','negativo'),
 ('es','Signos cicatriciales de un esguince del ligamento colateral medial.','MCL','incierto'),
 ('en','Technique: MRI of the knee. ACL normal. MCL normal. Medial meniscus tear. Coclusion: Medial meniscus tear.','Medial Meniscus','positivo'),
 ('en','Technique: MRI of the knee. ACL normal. MCL normal. Conclusion: No effusion.','MCL','negativo'),
 ('en','Technique: MRI of the knee. Evaluate possible ACL tear.','ACL','no_mencionado'),
 ('en','Clinical history: Medial meniscus tear. Findings: medial meniscus normal.','Medial Meniscus','negativo'),
 ('en','No evidence of tear is noted in either medial or lateral meniscus.','Medial Meniscus','negativo'),
 ('en','No evidence of tear is noted in either medial or lateral meniscus.','Lateral Meniscus','negativo'),
 ('en','Tear posterior horn medial meniscus extending into the body.','Medial Meniscus','positivo'),
 ('en','Complex tear of the posterior horn and posteromedial corner of the medial meniscus.','Medial Meniscus','positivo'),
 ('en','Possible tear posterior horn medial meniscus.','Medial Meniscus','incierto'),
 ('en','The medial collateral ligament and posterior oblique ligament are intact.','MCL','negativo'),
 ('en','Both medial & lateral collateral ligaments appear to be within normal.','MCL','negativo'),
 ('en','The contour of ACL appears to be within normal, no abnormal signal intensity is noted.','ACL','negativo'),
 ('en','ACL normal. Tear of the ACL.','ACL','incierto'),
 ('en','Compatible with fracture.','Fracture','incierto'),
 ('en','No acute fracture.','Fracture','incierto'),
]
CASOS += [
 ('es','Meniscos de morfología y señal conservada, sin criterios categóricos de rotura.','Medial Meniscus','incierto'),
 ('en','No convincing medial or lateral meniscal tears.','Medial Meniscus','incierto'),
 ('en','No convincing medial or lateral meniscal tears.','Lateral Meniscus','incierto'),
 *[('en','OA changes of all three compartments.',c,'positivo') for c in ('Medial OA','Lateral OA','PF OA')],
]
def comprobar(motor,clase,version):
    ad=clase();ad.registrar('es',version,motor.extraer_es);ad.registrar('en',version,motor.extraer_en)
    resultados=[]
    for i,(idioma,texto,c,esperado) in enumerate(CASOS,1):
        filas=ad.procesar('CTX'+str(i),texto,idioma);r=next(r for r in filas if r['condicion']==c)
        assert r['estado']==esperado,(i,texto,c,r['estado'],esperado)
        if i==4:assert 'Desgarro parcial' in r['evidencia'] and 'LCM' in r['evidencia']
        for r in filas:
            for ev in r['eventos']+r['menciones_no_diagnosticas']+r['bloqueos_indirectos']:
                a,b=ev['offsets_originales'];assert texto[a:b]==ev['fragmento_literal']
        resultados.append(dict(id=f'CTX{i:02}',idioma=idioma,condicion=c,esperado=esperado,correcto=True))
    return resultados
