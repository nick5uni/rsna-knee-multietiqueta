# Ejecución del comando único REV2

Archivos de esta carpeta:
- 36_RSNA_Comando_Unico_REV2.ipynb: importar en un notebook nuevo y privado de Kaggle.
- Paquete_Privado_RSNA_REV2_ComandoUnico.zip: subir como dataset PRIVADO y adjuntarlo al notebook. Incluye código, etiquetas consensuadas, máscaras, cierres, pruebas, referencias y los cuatro ZIP originales. No contiene el dataset DICOM; se adjunta la competencia por separado.
- SHA256SUMS.json: comprobación del ZIP y del notebook.

Preparación: adjuntar la competencia y el dataset privado, activar T4 e Internet y ejecutar la única celda del notebook. No se requiere conectar GitHub ni introducir tokens. No importar sobre los notebooks de las corridas anteriores. La copia del código incluida en el paquete es la que se ejecuta y queda identificada mediante hashes.

Para un evaluador que ya tiene código, insumos privados y competencia montados en Kaggle, el comando es:

    python ejecutar_kaggle.py --mode train

El lanzador genera config.kaggle.local.json sin editar rutas. Mantiene el entrenamiento acordado y crea resultados nuevos. Las cuatro configuraciones se ejecutan secuencialmente en la T4; no se deben lanzar en paralelo dentro de este notebook.

Pasos finales: descargar Resultados_Comando_Unico_REV2_train_*.zip, descargar el notebook con sus salidas y detener la GPU. La prueba completa se acredita solo después de terminar las cuatro configuraciones y pasar las comprobaciones finales. No se promete igualdad bit a bit con las corridas anteriores ni se sustituyen las cifras del informe automáticamente.

Limitación: no hay reanudación automática del lanzador si Kaggle interrumpe la sesión. Conservar las salidas parciales antes de reiniciar. Si el runtime no coincide con la referencia, se detiene con un mensaje, en lugar de entrenar con otra configuración.
