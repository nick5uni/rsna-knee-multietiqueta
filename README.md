# Clasificación multietiqueta de anomalías de rodilla

Proyecto de examen parcial del curso MIA-07, UNI, 2026-II. Se comparan una CNN propia y ResNet18 ajustada desde ImageNet para doce condiciones por estudio. Las etiquetas de entrenamiento son débiles y parcialmente revisadas. Los reportes supervisan el entrenamiento; no son entradas del clasificador de imágenes.

## Descarga del paquete de reproducción

La [Release `rev2-reproduccion-v1`](https://github.com/nick5uni/rsna-knee-multietiqueta/releases/tag/rev2-reproduccion-v1) reúne los insumos necesarios para el equipo y el evaluador:

- [Paquete privado completo, aproximadamente 709 MB](https://github.com/nick5uni/rsna-knee-multietiqueta/releases/download/rev2-reproduccion-v1/Paquete_Privado_RSNA_REV2_ComandoUnico.zip).
- [SHA256SUMS.json](https://github.com/nick5uni/rsna-knee-multietiqueta/releases/download/rev2-reproduccion-v1/SHA256SUMS.json), con los hashes esperados del paquete y del notebook.
- [Notebook de una celda, correspondiente a esta Release](https://github.com/nick5uni/rsna-knee-multietiqueta/blob/rev2-reproduccion-v1/notebooks/36_RSNA_Comando_Unico_REV2.ipynb).

Descargar el paquete y el notebook con una cuenta que tenga acceso al repositorio privado. En un notebook nuevo y privado de Kaggle, adjuntar la competencia RSNA y el paquete como dataset privado; activar T4 e Internet y ejecutar la única celda. El paquete admite que Kaggle conserve el ZIP o lo descomprima al crear el dataset. No se necesitan credenciales GitHub dentro de Kaggle.

Con el paquete ya preparado y los datos montados, la celda lanza:

```bash
python ejecutar_kaggle.py --mode train
```

La preparación inicial es la misma para el equipo y el evaluador. El lanzador comprueba las rutas y los insumos, ejecuta las pruebas, encadena las cuatro configuraciones y guarda un ZIP con los resultados y sus hashes. El dataset DICOM de la competencia se obtiene por separado desde su fuente autorizada.

**Estado de verificación:** 38 pruebas locales correctas, extracción e integridad del paquete comprobadas, auditoría desde la copia extraída correcta y reproducción DICOM en CPU correcta. La ejecución completa de los cuatro entrenamientos mediante este lanzador en T4 todavía está pendiente; publicar la Release no acredita esa ejecución.

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

Las cuatro corridas académicas originales están acreditadas por sus ZIP y notebooks ejecutados. La nueva integración del modo train se ofrece para repetirlas; no se declara que se hayan vuelto a realizar cuatro entrenamientos completos durante el cierre de entrega en CPU.

### Auditoría ligera

```powershell
python pipeline.py --config config.local.json --mode audit
```

Verifica contratos y reproduce extracción textual y métricas desde CSV, sin inferencia DICOM. **Este modo no constituye por sí solo reproducción completa del clasificador.**

## Ejecución en Kaggle sin editar rutas

Se proporciona `notebooks/36_RSNA_Comando_Unico_REV2.ipynb` y el archivo `Paquete_Privado_RSNA_REV2_ComandoUnico.zip`, descargable desde la Release privada enlazada arriba. El mismo paquete se utiliza para el equipo y para el evaluador. Se adjuntan la competencia y el paquete privado, se activa T4 e Internet y se ejecuta una celda. El lanzador detecta rutas, comprueba el entorno, prepara dependencias ligeras, ejecuta pruebas y encadena las cuatro configuraciones con el pipeline existente:

```bash
python ejecutar_kaggle.py --mode train
```

Los resultados se escriben en una carpeta nueva y se empaquetan automáticamente con checkpoints, predicciones, métricas, Grad-CAM, log y hashes. No se necesitan credenciales GitHub dentro del notebook. Véase `docs/EJECUCION_KAGGLE.md`. El paquete también funciona si Kaggle descomprime el ZIP al crear el dataset privado.

La preparación local no acredita todavía la ejecución T4 completa. No hay reanudación automática del lanzador: una nueva ejecución crea otra carpeta y repite las cuatro configuraciones. Los resultados del informe no se reemplazan por las nuevas salidas sin revisión.

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
