# Correspondencia con los entregables de la pregunta 1

| Requisito | Implementación y evidencia |
|---|---|
| Extracción de doce etiquetas | text_engine, extractor EN/ES v1.2; cobertura y acuerdo conservan abstenciones; 38 GOLD en alcance |
| CNN y transferencia | src: CNN propia y ResNet18 ImageNet con doce salidas; cuatro corridas originales y sus checkpoints privados |
| Métricas GOLD | evaluador común: AUROC macro y por etiqueta, AP, precisión, recall y F1 a 0,5; evaluación REV2 exploratoria |
| Dos factores de ablación | BCE frente a focal; uso frente a exclusión de negativos indirectos OA en TRAIN; mismo DEV |
| Grad-CAM | seis casos DEV de ACL, menisco medial y derrame; 48 mapas; interpretación exploratoria sin validación de localización clínica |
| Desbalance | proporciones por condición en el informe y matrices; focal sin alfa como estrategia evaluada |
| Sesgo lingüístico/geográfico | discusión del informe; alcance EN/ES, abstenciones y 20 GOLD de idioma indeterminado fuera del alcance textual |
| Reproducción | pipeline.py, README, requirements y comprobación CPU documentada; entrenamiento T4 disponible y no repetido durante el cierre |

El informe y las imágenes derivadas se entregan por el canal privado del curso. Este repositorio conserva código y pruebas; no contiene reportes clínicos, imágenes DICOM ni checkpoints.
