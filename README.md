# Clasificación multietiqueta de anomalías de rodilla

Proyecto de examen parcial del curso MIA-07, UNI, 2026-II. Se comparan una CNN propia y ResNet18 ajustada desde ImageNet para doce condiciones por estudio. Las etiquetas de entrenamiento son débiles y parcialmente revisadas. Los reportes supervisan el entrenamiento; no son entradas del clasificador de imágenes.

## Ejecución completa en Kaggle

Usar el [notebook corregido de una celda](notebooks/36_RSNA_Comando_Unico_REV2_CORREGIDO_v2.ipynb). El notebook anterior permanece como historial y no debe utilizarse para una nueva ejecución.

1. Crear un notebook privado en Kaggle e importar el notebook corregido.
2. Adjuntar la competencia RSNA Knee Abnormality Detection y aceptar sus condiciones con la cuenta correspondiente.
3. Activar GPU Tesla T4 e Internet y ejecutar la única celda. La celda descarga el ZIP original desde la [Release base](https://github.com/nick5uni/rsna-knee-multietiqueta/releases/tag/rev2-reproduccion-v1), comprueba su SHA-256, prepara el entorno y lanza las cuatro configuraciones. Si ya hay un ZIP íntegro en Input o caché, lo reutiliza. No necesita los cuatro ZIP internos montados por separado.

El paquete base se conserva sin cambios: [Paquete_Privado_RSNA_REV2_ComandoUnico.zip](https://github.com/nick5uni/rsna-knee-multietiqueta/releases/download/rev2-reproduccion-v1/Paquete_Privado_RSNA_REV2_ComandoUnico.zip), SHA-256 `b4897e57edf9b426ce6040ab3791f2ac0e8a63831467881b773aed172825160e`. El hash del notebook antiguo incluido en esa Release no corresponde al notebook corregido: utilizar [los hashes del cierre](docs/SHA256SUMS_cierre_20261028.json).

No basta con el paquete descomprimido automáticamente por Kaggle: puede perder los ZIP internos. Se recomienda la descarga directa que realiza la celda, o adjuntar el ZIP original íntegro. Si el repositorio cambia a privado, esa descarga sin autenticación dejará de funcionar; en tal caso debe proporcionarse el ZIP íntegro por Input. No introducir credenciales en el código.

El comando de entrenamiento que ejecuta la preparación es:

```bash
python ejecutar_kaggle.py --mode train
```

También se ofrece `bootstrap_kaggle_v2.py`, idéntico a la celda, para ejecutar `python bootstrap_kaggle_v2.py` dentro de Kaggle. El dataset DICOM se obtiene por separado desde su fuente autorizada.

**Estado verificado el 28/10/2026:** el lanzador corregido completó cuatro entrenamientos nuevos, evaluaciones y Grad-CAM en Tesla T4, con salida 0 y `COMPROBACION COMPLETA: OK | nuevo entrenamiento: True`. Se contrastaron los hashes de los 64 archivos del ZIP de resultados. La preparación se ejecutó en varias celdas. La copia consolidada de una celda superó cinco pruebas funcionales con lanzamiento simulado; no se ha vuelto a ejecutar íntegramente en una sesión T4 limpia. Véanse [el registro de cierre](docs/CIERRE_COMANDO_UNICO_20261028.md) y [la evidencia agregada](docs/verificacion_T4_20261028.json).

## Alcance y versiones

- 300 estudios: 240 TRAIN y 60 DEV. REV2 contiene 1.458 etiquetas TRAIN conocidas y 364 DEV. La variante filtrada excluye 163 negativos indirectos OA solo en TRAIN y utiliza 1.295 etiquetas.
- CNN+BCE, ResNet18+BCE, ResNet18+Focal (gamma=2, sin alfa) y ResNet18+BCE con filtro. Adam, lr=0,0001, lote de un estudio, semilla 42, hasta 20 épocas y paciencia 3. Se selecciona por mínima BCE DEV. Todas las configuraciones reciben las mismas series y cortes.
- Evaluación GOLD de 58 estudios y doce condiciones. La evaluación REV2 es exploratoria: GOLD ya se había consultado. No es una nueva validación independiente ni una prueba de utilidad clínica.
- Extractor EN/ES v1.2 congelado, separado de las matrices REV2: 38 reportes en alcance, 456 pares, 197 resueltos, 164 coincidencias y 259 abstenciones. El acuerdo de 83,2 % se limita a resueltos; la cobertura es 43,2 %. Los otros 20 GOLD tienen idioma indeterminado y quedan fuera del alcance.
- Grad-CAM: seis casos DEV preseleccionados, ocho mapas por caso y escala compartida entre cortes. No son segmentaciones expertas ni validación anatómica.

## Instalación

Se requiere Python 3.13. En Windows:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

En Linux se utilizan `python3.13 -m venv .venv` y `.venv/bin/python`. Para repetir entrenamientos se necesita PyTorch con CUDA y una GPU Tesla T4, referencia de las corridas académicas. Usar el índice `https://download.pytorch.org/whl/cu128` en lugar del índice CPU. CPU y CUDA son entornos distintos; no se promete igualdad bit a bit entre dispositivos.

## Datos y artefactos privados

1. Obtener los datos de la competición desde la fuente autorizada, aceptando sus reglas con la cuenta correspondiente: <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection>.
2. Disponer de `train.csv`, `train_series.csv` y del ZIP original o las series DICOM extraídas. No es necesario extraer todo el ZIP de 265 GB: el programa extrae únicamente las series de los manifiestos.
3. Descargar el paquete de insumos congelados desde la Release indicada arriba. Al extraerlo, utilizar la carpeta `rsna-knee-multietiqueta/private/` junto al código de esa versión. Contiene matrices, manifiestos, decisiones de idioma, referencias y los cuatro ZIP completos de corridas con best.pt y last.pt. `private/SHA256SUMS.json` y los manifiestos verifican su integridad. No se regeneran decisiones de consenso a partir de reglas bilingües posteriores.
4. Copiar `config.example.json` a `config.local.json` y configurar las rutas. Las rutas relativas se resuelven desde esta carpeta, no desde el directorio de invocación. `data_root`, si se proporciona, debe contener `train_series/UID/SID/*.dcm`.

Los datos, reportes clínicos, manifiestos con identificadores, checkpoints, resultados por estudio, claves y archivos de configuración local se excluyen de Git. No se reemplazan insumos privados por ejemplos inventados cuando faltan archivos. Los ejemplos artificiales de las pruebas no son resultados académicos.

## Un comando de reproducción desde texto e imágenes

Después de instalar el entorno y configurar las rutas:

```powershell
python pipeline.py --config config.local.json --mode reproduce
```

Este comando verifica los hashes y contratos, ejecuta el extractor congelado sobre los reportes GOLD EN/ES, reproduce las decisiones y métricas textuales, valida las matrices y el filtro, prepara los DICOM de DEV y GOLD, recarga los cuatro checkpoints seleccionados, vuelve a inferir DEV y GOLD, calcula métricas oficiales y genera los 48 mapas Grad-CAM. Los resultados se escriben en una carpeta nueva bajo `resultados/`. No se alteran los resultados históricos, reglas, etiquetas ni checkpoints. No se selecciona época ni umbral con GOLD.

El resultado `ejecucion.json` distingue expresamente inferencia de nuevo entrenamiento. Se compara la probabilidad GOLD con el CSV original con tolerancia absoluta máxima de 0,0002 para diferencias numéricas entre T4 y CPU; la diferencia observada queda registrada por modelo. Esta tolerancia no cambia etiquetas, umbral ni selección.

### Reentrenamiento completo desde las inicializaciones acordadas

En el entorno CUDA con `config.device` igual a `cuda:0`:

```powershell
python pipeline.py --config config.local.json --mode train
```

Este modo prepara también los 240 TRAIN y repite las cuatro configuraciones desde sus inicializaciones acordadas, conservando el motor, las pérdidas, el lector, la política BN por estudio y los parámetros. ResNet18 usa ImageNet IMAGENET1K_V1; la CNN propia parte de inicialización aleatoria. Requiere acceso a los pesos oficiales ImageNet o su caché local, y puede consumir horas de GPU. Se guardan best.pt y last.pt y se continúa con la evaluación y los mapas. No se usa un checkpoint entrenado como inicialización.

Las cuatro corridas académicas originales se conservan. El 28/10/2026 se completó además la ejecución del lanzador corregido en T4, con cuatro entrenamientos nuevos; esta comprobación no sustituye los resultados del informe.

### Auditoría ligera

```powershell
python pipeline.py --config config.local.json --mode audit
```

Verifica contratos y reproduce extracción textual y métricas desde CSV, sin inferencia DICOM. **Este modo no constituye por sí solo reproducción completa del clasificador.**

## Ejecución en Kaggle sin editar rutas

Seguir la preparación inicial y usar `notebooks/36_RSNA_Comando_Unico_REV2_CORREGIDO_v2.ipynb`, o `python bootstrap_kaggle_v2.py` dentro de Kaggle. Véase [EJECUCION_KAGGLE.md](docs/EJECUCION_KAGGLE.md). Se comprueba el paquete original antes de aplicar únicamente la limpieza de gradientes del orquestador anterior a Grad-CAM. Se mantienen los módulos congelados, datos, parámetros y política BatchNorm.

Las cuatro configuraciones se ejecutan secuencialmente. Cada ejecución crea una carpeta nueva y un ZIP con checkpoints, predicciones, métricas, mapas y hashes. No hay reanudación automática: repetir el lanzamiento vuelve a entrenar. Los resultados históricos no se sustituyen automáticamente.

## Pruebas

```powershell
python -m unittest discover -s tests -v
python text_engine/ejecutar_pruebas.py
```

Para el evaluador textual: ejecutar `python -m unittest test_evaluador_textual -v` desde `text_engine/`. Las pruebas de modelos utilizan tensores artificiales; no sustituyen la ejecución con datos RSNA. Si una restricción del sistema impide a PyTorch escribir en la carpeta temporal, configurar TEMP/TMP a una carpeta local escribible; no modificar el código congelado para eludir controles.

## Estructura

```text
pipeline.py                    Entrada única
src/                           Lector, modelos, pérdidas, entrenamiento, BN y evaluador
text_engine/                   Extractor EN/ES y pruebas
tests/                         Pruebas sintéticas y contratos
protocolo_bn_v1.json            Referencia de política y hashes
codigo_congelado_sha256.json    Integridad del código recibido
requirements.txt               Dependencias directas fijadas
config.example.json            Configuración sin rutas personales
private/                       Insumos privados, excluidos de Git
cache/                         DICOM extraídos selectivamente, excluidos de Git
resultados/                    Salidas nuevas, excluidas de Git
```

## Interpretación de resultados

AUROC/AP son macro por condición; precisión, recall y F1 se calculan al umbral 0,5 fijado antes de GOLD. La BCE se agrega sobre etiquetas conocidas. Se conserva desconocido como máscara cero, nunca como negativo. La revisión por consenso es textual y no reemplaza adjudicación radiológica. Una sola semilla, el muestreo de ocho cortes, el tamaño pequeño y la selección dirigida limitan generalización. No se atribuye causalmente la mejora REV2 a una sola intervención.

## Referencias

Bien et al., MRNet, PLOS Medicine (2018), DOI 10.1371/journal.pmed.1002699. He et al., Deep Residual Learning, CVPR (2016), DOI 10.1109/CVPR.2016.90. Ratner et al., Snorkel, PVLDB (2017), DOI 10.14778/3157794.3157797. Lin et al., Focal Loss, ICCV (2017), DOI 10.1109/ICCV.2017.324. Selvaraju et al., Grad-CAM, ICCV (2017), DOI 10.1109/ICCV.2017.74. Datos: RSNA Knee Abnormality Detection, Kaggle.
