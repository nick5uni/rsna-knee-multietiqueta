# Cierre del comando único REV2 del 28 de octubre de 2026

## Ejecución acreditada

El procedimiento corregido se ejecutó completamente en Kaggle Tesla T4. El ZIP de resultados contiene los cuatro entrenamientos nuevos, best.pt y last.pt para cada configuración, predicciones TRAIN/DEV/GOLD, evaluaciones textuales y de imágenes, 48 mapas Grad-CAM y registros. Los hashes de sus 64 archivos se verificaron localmente contra su manifiesto. La captura y el notebook ejecutado muestran salida 0 y COMPROBACION COMPLETA: OK. Se conservan localmente los archivos completos con imágenes y resultados por estudio; aquí se publica la evidencia agregada sin datos clínicos.

La evidencia original se conserva en el archivo conjunto del proyecto: `Entrega_Victor_Comando_Unico_REV2_Cierre_28Oct2026.zip`, `Resultados_Comando_Unico_REV2_train_20261028T202738980849Z.zip` y `Screenshot-de-Evidencia.png`.

## Alcance exacto de la celda consolidada

La ejecución GPU se realizó con la preparación y corrección en varias celdas. La copia final autónoma superó cinco pruebas funcionales que simulan descarga y lanzamiento, sin ejecutar entrenamiento. No se declara una segunda ejecución T4 de esa copia exacta. No se requiere repetir los cuatro entrenamientos para publicar esta corrección con ese alcance.

Las pruebas originales están en `tests/preparacion_v2/`. En Linux/Kaggle: `python tests/preparacion_v2/test_preparacion_celda_unica.py`. Para repetir su preparación en Windows con la semántica LF de Kaggle: `python tests/preparacion_v2/ejecutar_pruebas_windows.py`. La diferencia Windows de escritura CRLF no modifica el pipeline entregado ni el notebook de Kaggle.

## Corrección y versiones

El único cambio del pipeline es `model.zero_grad(set_to_none=True)` antes de Grad-CAM. SHA-256 del pipeline corregido, igual al ejecutado en T4: `ab93fb0781403a9fe7d1da1911693dd9d5d42e96ea368bc13ec48ce5c46de3d1`. No se cambiaron módulos congelados, inicializaciones, pérdidas, datos ni parámetros. La Release `rev2-reproduccion-v1` y su paquete original quedan intactos. La entrada autónoma comprueba ese paquete y aplica de forma trazable la corrección.

Las métricas y figuras del informe se conservan; la nueva corrida acredita reproducción y no constituye una nueva validación independiente de GOLD. La evaluación GOLD sigue siendo exploratoria.

## Entorno

Python 3.13.15, torch 2.11.0+cu128, torchvision 0.26.0+cu128, CUDA 12.8, numpy 2.1.3, pydicom 3.0.2 y Pillow 12.3.0; GPU Tesla T4. Dependencias en `requirements.txt`; el lanzador verifica el entorno antes de entrenar.
