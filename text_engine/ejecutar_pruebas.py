from pathlib import Path
import json
from verificar_integracion import verificar
from pruebas_contexto import comprobar
from motor_textual_comun import MotorTextualComun, VERSION
from adaptador_bilingue import AdaptadorBilingue
from pruebas_bloqueos_OA import comprobar as comprobar_OA
from verificar_nicolas import verificar as verificar_Nicolas
carpeta=Path(__file__).resolve().parent
resumen,pares,pruebas=verificar(carpeta)
motor=MotorTextualComun(carpeta)
contexto=comprobar(motor,AdaptadorBilingue,VERSION)
bloqueos=comprobar_OA(motor,AdaptadorBilingue,VERSION)
nicolas=verificar_Nicolas(carpeta)
registro=dict(resumen_regresion=resumen,contexto_32=len(contexto),controles_OA_20=bloqueos,ejecutores_Nicolas=nicolas,todos_correctos=True)
(carpeta/'resultados_pruebas_v1_2.json').write_text(json.dumps(registro,ensure_ascii=False,indent=2),encoding='utf-8')
print('Regresión previa: todos los grupos correctos.')
print('Contexto NB34:',len(contexto),'/ 32')
print('Controles de alcance y bloqueos OA:',len(bloqueos),'/ 20')
print(nicolas['regresiones_originales_6_y_traza'].strip())
print(nicolas['OA_compartida_4'].strip())
print('Versión:',VERSION)
