# Comprobación de reproducción

Comprobación local del 27 de octubre de 2026 (Lima). Se utilizó un entorno virtual nuevo en Windows, Python 3.13.2, torch 2.11.0+cpu, torchvision 0.26.0+cpu, numpy 2.1.3, pydicom 3.0.2 y Pillow 12.3.0. El archivo requirements-cpu-lock.txt registra las dependencias instaladas.

Comando ejecutado desde el repositorio:

```powershell
python pipeline.py --config config.local.json --mode reproduce
```

Resultado: OK. Se verificaron matrices REV2, filtro, hashes, 118 entradas DICOM (60 DEV y 58 GOLD), inferencia de cuatro checkpoints, extracción textual EN/ES y seis explicaciones Grad-CAM de ocho cortes. La época seleccionada de cada modelo coincide con el mínimo de su historial DEV.

| Configuración | Época | Diferencia máxima DEV | Diferencia máxima GOLD |
|---|---:|---:|---:|
| CNN_BCE | 17 | 1.1920929e-07 | 1.1920929e-07 |
| ResNet18_BCE | 8 | 1.87754631e-06 | 2.56299973e-06 |
| ResNet18_FOCAL | 19 | 1.90734863e-06 | 1.54972076e-06 |
| ResNet18_BCE_FILTRO | 1 | 4.17232513e-07 | 2.98023224e-07 |

Las diferencias numéricas se comparan con las probabilidades originales de T4; todas están por debajo de 0,0002. No se cambiaron las métricas académicas del informe ni se realizaron nuevos entrenamientos. GOLD conserva carácter exploratorio.

También pasaron 28 pruebas de modelos y contratos (11 BN, 7 focal y 10 GOLD), las comprobaciones del motor textual congelado y 9 pruebas de su evaluador. Las pruebas con tensores o textos artificiales se distinguen de la inferencia sobre los estudios reales.

El modo train está implementado para repetir las cuatro corridas en T4 desde las inicializaciones acordadas. Esta comprobación no acredita una nueva ejecución completa de ese modo; las corridas originales se conservan en sus ZIP y notebooks ejecutados.

La reproducción requiere el paquete privado y los datos autorizados de Kaggle. El repositorio por sí solo no contiene estos insumos.
