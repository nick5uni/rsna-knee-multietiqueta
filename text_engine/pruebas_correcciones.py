import copy
CASOS=[
 ('No hay rotura del LCA ni lesión del LCM.',{'ACL':'negativo','MCL':'negativo'}),
 ('Ligamento cruzado anterior con rotura completa.',{'ACL':'positivo'}),
 ('Menisco medial con rotura horizontal.',{'Medial Meniscus':'positivo'}),
 ('Sin rotura del menisco medial ni del menisco lateral.',{'Medial Meniscus':'negativo','Lateral Meniscus':'negativo'}),
 ('No hay derrame articular ni sinovitis.',{'Effusion':'negativo','Synovitis':'negativo'}),
 ('No hay derrame articular. Sinovitis.',{'Effusion':'negativo','Synovitis':'positivo'}),
 ('No hay derrame articular pero hay sinovitis.',{'Effusion':'negativo','Synovitis':'positivo'}),
 ('No hay rotura del LCA. Lesión del LCM.',{'ACL':'negativo','MCL':'positivo'}),
 ('No hay rotura del LCA pero hay lesión del LCM.',{'ACL':'negativo','MCL':'positivo'}),
 ('Compatible con fractura tibial.',{'Fracture':'incierto'}),
 ('Hallazgos: fractura tibial. Conclusión: sin fractura.',{'Fracture':'incierto'}),
 ('Antecedentes: fractura tibial. Hallazgos: sin derrame.',{'Fracture':'no_mencionado'}),
]
def comprobar(extraer, clase, version):
    ad=clase();ad.registrar('es',version,extraer)
    resultados=[]
    for i,(texto,esperado) in enumerate(CASOS,1):
        original=extraer(texto);salida=ad.procesar(str(i),texto,'es')
        por={r['condicion']:r for r in salida}
        assert all(por[c]['estado']==e for c,e in esperado.items()), (texto,esperado,por)
        for r,s in zip(original,salida):
            for campo in ('seccion','offsets_originales','eventos','menciones_no_diagnosticas','version_reglas'):
                assert r[campo]==s[campo],campo
            assert not s['habilitada_entrenamiento']
        resultados.append({'id':f'CORR{i:02}','correcto':True})
    texto=CASOS[10][0]
    capturada=extraer(texto);antes=copy.deepcopy(capturada)
    ad2=clase();ad2.registrar('es',version,lambda t:capturada)
    filas=ad2.procesar('contradiccion',texto,'es')
    f=next(r for r in filas if r['condicion']=='Fracture')
    assert {e['estado'] for e in f['eventos']}=={'positivo','negativo'}
    f['eventos'][0]['estado']='alterado_solo_en_copia'
    assert capturada==antes
    # Los offsets alterados deben bloquear la salida.
    corrupta=copy.deepcopy(antes)
    next(r for r in corrupta if r['condicion']=='Fracture')['eventos'][0]['offsets_originales']=[0,1]
    mal=clase();mal.registrar('es',version,lambda t:corrupta)
    try:mal.procesar('offset_invalido',texto,'es')
    except ValueError:pass
    else:raise AssertionError('Se aceptaron offsets corruptos')
    return resultados
