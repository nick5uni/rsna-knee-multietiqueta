"""python comprobar_OA_compartida_Nicolas.py CARPETA_NB34

Control artificial del alcance ampliado de negativos indirectos OA acordado.
No consulta GOLD ni equivale a afirmar ausencia clínica de artrosis.
"""
from pathlib import Path
import importlib.util,json,sys

def main():
    folder=Path(sys.argv[1]).resolve();sys.path.insert(0,str(folder))
    spec=importlib.util.spec_from_file_location('motor_revision_compartida',folder/'motor_textual_comun.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    motor=module.MotorTextualComun(folder)
    cases=json.loads((Path(__file__).parent/'controles_OA_normalidad_compartida.json').read_text(encoding='utf-8'))
    failures=[]
    for i,case in enumerate(cases,1):
        rows=(motor.extraer_en if case['idioma']=='en' else motor.extraer_es)(case['texto_artificial'])
        rows=[r for r in rows if r['condicion'] in ('Medial OA','Lateral OA','PF OA')]
        bad={r['condicion']:r['estado'] for r in rows if r['estado']!='negativo' or r['origen_decision']!='indirecto_OA'}
        print('OA_COMPARTIDA_'+str(i),'OK' if not bad else json.dumps(bad,ensure_ascii=False))
        if bad:failures.append(i)
    print('Alcances pendientes:',failures)
    return bool(failures)

if __name__=='__main__':sys.exit(main())
