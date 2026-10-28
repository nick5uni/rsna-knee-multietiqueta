# Ejecución del comando único REV2 en Kaggle

Usar `notebooks/36_RSNA_Comando_Unico_REV2_CORREGIDO_v2.ipynb`, importar en un notebook nuevo privado, adjuntar la competencia, activar Tesla T4 e Internet y ejecutar la única celda. La celda y `bootstrap_kaggle_v2.py` contienen el mismo código.

La entrada descarga el ZIP original de la Release base y verifica su SHA-256; también admite el ZIP íntegro adjunto como Input o en caché. No depender del paquete descomprimido automáticamente por Kaggle: sus cuatro ZIP internos pueden faltar. No cambiar la Release ni sus hashes. El notebook corregido tiene hashes propios en `SHA256SUMS_cierre_20261028.json`.

Se extrae y verifica el manifiesto original, se comprueban los módulos congelados y se aplica la limpieza de gradientes del pipeline antes de Grad-CAM. Se ejecuta una sola vez `python ejecutar_kaggle.py --mode train`. Las cuatro configuraciones se entrenan secuencialmente desde las inicializaciones acordadas; no lanzar en paralelo.

Descargar el ZIP final y el notebook con salidas y detener la GPU. No hay reanudación automática. Si se interrumpe, conservar las salidas antes de volver a lanzar.

El lanzador corregido terminó en T4 el 28/10/2026. La preparación se ejecutó en varias celdas. La copia consolidada final pasó cinco pruebas funcionales con lanzamiento simulado, sin otra ejecución GPU. Véase `CIERRE_COMANDO_UNICO_20261028.md`.
