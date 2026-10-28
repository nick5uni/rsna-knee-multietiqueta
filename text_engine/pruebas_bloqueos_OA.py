"""Controles artificiales de alcance y bloqueos para normalidad conjunta."""
CASOS=[
 ('en','Normal cartilage.',set()),
 ('es','Cartílago normal.',set()),
 ('en','History: The tibiofemoral and patellofemoral articular cartilage is normal.',set()),
 ('es','Antecedentes: El cartílago tibiofemoral y patelofemoral es normal.',set()),
 ('en','Technique: The tibiofemoral and patellofemoral articular cartilage is normal.',set()),
 ('es','Técnica: El cartílago tibiofemoral y patelofemoral es normal.',set()),
 ('en','Possible normal tibiofemoral and patellofemoral articular cartilage.',set()),
 ('es','Posible cartílago tibiofemoral y patelofemoral normal.',set()),
 ('en','The tibiofemoral and patellofemoral articular cartilage is normal except for a defect.',set()),
 ('es','El cartílago tibiofemoral y patelofemoral es normal excepto un defecto.',set()),
 ('en','The tibiofemoral and patellofemoral articular cartilage is not normal.',set()),
 ('es','El cartílago tibiofemoral y patelofemoral no es normal.',set()),
 ('en','The tibiofemoral and patellofemoral articular cartilage is normal. Medial compartment osteoarthritis.',{'Lateral OA','PF OA'}),
 ('es','El cartílago tibiofemoral y patelofemoral es normal. Artrosis del compartimento medial.',{'Lateral OA','PF OA'}),
 ('en','The tibiofemoral and patellofemoral articular cartilage is normal. Cartilage loss.',set()),
 ('es','El cartílago tibiofemoral y patelofemoral es normal. Pérdida de cartílago.',set()),
 ('en','The tibiofemoral and patellofemoral articular cartilage is normal. Patellar chondromalacia.',{'Medial OA','Lateral OA'}),
 ('es','El cartílago tibiofemoral y patelofemoral es normal. Condromalacia rotuliana.',{'Medial OA','Lateral OA'}),
 ('en','Normal tibiofemoral and patellofemoral articular cartilage.',{'Medial OA','Lateral OA','PF OA'}),
 ('es','Cartílago tibiofemoral y patelofemoral normal.',{'Medial OA','Lateral OA','PF OA'}),
]
def comprobar(motor,clase,version):
    ad=clase();ad.registrar('en',version,motor.extraer_en);ad.registrar('es',version,motor.extraer_es)
    resultados=[]
    for i,(idioma,texto,esperados) in enumerate(CASOS,1):
        filas=ad.procesar('BLOQUEO_ARTIFICIAL_'+str(i),texto,idioma)
        neg={r['condicion'] for r in filas if r['condicion'] in ('Medial OA','Lateral OA','PF OA') and r['estado']=='negativo' and r['origen_decision']=='indirecto_OA'}
        assert neg==esperados,(i,texto,esperados,neg)
        for r in filas:
            assert not r['habilitada_entrenamiento']
            for e in r['eventos']+r['menciones_no_diagnosticas']+r['bloqueos_indirectos']:
                a,b=e['offsets_originales'];assert texto[a:b]==e['fragmento_literal']
        resultados.append(dict(id=f'OA_BLOQ_{i:02}',idioma=idioma,negativos_indirectos=sorted(neg),correcto=True))
    return resultados
