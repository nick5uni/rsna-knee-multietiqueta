"""Ejecutar: python comprobar_regresiones_Nicolas.py CARPETA_DEL_MOTOR

Usa seis textos artificiales y comprueba que el adaptador conserve la trazabilidad.
No carga reportes reales, matrices ni GOLD y no modifica el motor recibido.
"""
from pathlib import Path
import importlib.util,json,sys

def cargar(name,p):
    s=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
    folder=Path(sys.argv[1]).resolve()
    es=cargar('es_revisado',folder/'extractor_es_candidato.py')
    ad=cargar('adaptador_revisado',folder/'adaptador_bilingue.py')
    cases=json.loads((Path(__file__).parent/'regresiones_adicionales_Nicolas.json').read_text(encoding='utf-8'))
    failed=[]
    for case in cases:
        states={r['condicion']:r['estado'] for r in es.extraer_es(case['texto_artificial'])}
        differences={c:{'esperado':v,'obtenido':states[c]} for c,v in case['esperado'].items() if states[c]!=v}
        print(case['id'],'OK' if not differences else json.dumps(differences,ensure_ascii=False))
        if differences:failed.append(case['id'])
    wrapper=ad.AdaptadorBilingue();wrapper.registrar('es',es.VERSION,es.extraer_es)
    text='Hallazgos: Fractura. Impresión: sin fractura.'
    direct=next(r for r in es.extraer_es(text) if r['condicion']=='Fracture')
    wrapped=next(r for r in wrapper.procesar('SOFTWARE_FIXTURE',text,'es') if r['condicion']=='Fracture')
    keys=['seccion','offsets_originales','eventos','menciones_no_diagnosticas']
    ok=all(wrapped.get(k)==direct.get(k) for k in keys)
    print('TRAZA_ADAPTADOR','OK' if ok else 'Debe conservar seccion, offsets y ambas evidencias de contradiccion.')
    if not ok:failed.append('TRAZA_ADAPTADOR')
    print('Fallos:',failed)
    return bool(failed)

if __name__=='__main__':sys.exit(main())
