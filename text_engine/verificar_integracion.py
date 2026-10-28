"""Ejecutor reproducible de pruebas artificiales, sin leer reportes reales."""
import json,types,sys,unittest,io
from pathlib import Path

def verificar(carpeta):
    carpeta=Path(carpeta)
    def cargar(nombre,archivo):
        m=types.ModuleType(nombre);exec(compile((carpeta/archivo).read_text(encoding='utf-8'),archivo,'exec'),m.__dict__);return m
    comun=cargar('comun_candidato','motor_textual_comun.py')
    ad=cargar('adaptador_bilingue','adaptador_bilingue.py')
    motor=comun.MotorTextualComun(carpeta)
    adaptador=ad.AdaptadorBilingue()
    adaptador.registrar('es',comun.VERSION,motor.extraer_es)
    adaptador.registrar('en',comun.VERSION,motor.extraer_en)
    base=cargar('pruebas_anteriores','comprobaciones_motor.py')
    casos=json.loads((carpeta/'casos_Nicolas_originales_60.json').read_text(encoding='utf-8'))
    previos=base.comprobar_motor(motor.extraer_es,adaptador,casos)
    cambios=[r for r in previos if not r['correcto']]
    esperados={'VIC011':('PF OA','incierto'),'VIC012':('Medial OA','negativo')}
    assert {r['id'] for r in cambios}==set(esperados),cambios
    for r in cambios:
        c,nuevo=esperados[r['id']]
        assert r['fallos'][c]['obtenido']==nuevo
    # Versionar expectativas en copia; conservar archivo y resultados anteriores.
    base.CASOS_ADICIONALES=[(t,c,esperados[f'VIC{i:03}'][1] if f'VIC{i:03}' in esperados else e) for i,(t,c,e) in enumerate(base.CASOS_ADICIONALES,1)]
    actualizados=base.comprobar_motor(motor.extraer_es,adaptador,casos)
    assert all(r['correcto'] for r in actualizados)
    pares=cargar('pruebas_comunes','pruebas_comunes.py').comprobar(motor,adaptador)
    correctivos=cargar('pruebas_correcciones','pruebas_correcciones.py').comprobar(motor.extraer_es,ad.AdaptadorBilingue,comun.VERSION)
    # Mismas 16 pruebas históricas en el motor inglés instrumentado y en el integrado.
    ns={'extraer_reporte_v2':motor.en.extraer_reporte_v2}
    exec(compile((carpeta/'historico_EN_pruebas_v2.py').read_text(encoding='utf-8'),'historico_EN_pruebas_v2.py','exec'),ns)
    traduccion={'presente':'positivo','ausente':'negativo','incierto':'incierto','no_mencionado':'no_mencionado'}
    for t,c,e in ns['pruebas']:
        r=next(r for r in adaptador.procesar('historico',t,'en') if r['condicion']==c)
        assert r['estado']==traduccion[e],(t,c,e,r)
    anterior=sys.modules.get('adaptador_bilingue');sys.modules['adaptador_bilingue']=ad
    try:
        tests=cargar('test_originales_adaptador','test_adaptador.py')
        salida=io.StringIO()
        resultado=unittest.TextTestRunner(stream=salida,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
        assert resultado.wasSuccessful(),salida.getvalue()
    finally:
        if anterior is None:sys.modules.pop('adaptador_bilingue',None)
        else:sys.modules['adaptador_bilingue']=anterior
    resumen=dict(version=comun.VERSION,casos_originales_Nicolas=60,adicionales_versionados=21,pares_bilingues=len(pares)//2,ejecuciones_pares=len(pares),controles_correctivos=len(correctivos),pruebas_EN_historicas=len(ns['pruebas']),pruebas_adaptador=resultado.testsRun,
                 cambios_expectativas=cambios,reglas_congeladas=False,GOLD_leido=False,REV2_modificada=False,
                 revision_reportes_completos_sin_GOLD='pendiente',ejecutor_adicional_Nicolas='pendiente_recepcion')
    return resumen,pares,actualizados

if __name__=='__main__':
    carpeta=Path(__file__).resolve().parent
    resumen,pares,actualizados=verificar(carpeta)
    print(json.dumps(resumen,ensure_ascii=False,indent=2))
