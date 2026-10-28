"""Entrada por estudio, sin reportes ni etiquetas; devuelve numpy y una mascara."""
from pathlib import Path
import numpy as np
import pydicom
from PIL import Image
from .pilot_review import spatial_order


def select_indices(coords, max_slices=8):
    coords = np.asarray(coords, dtype=float)
    if max_slices < 1 or coords.ndim != 1 or not len(coords):
        raise ValueError('Cantidad de cortes o posiciones invalida.')
    if not np.isfinite(coords).all() or np.any(np.diff(coords) <= 0):
        raise ValueError('Las posiciones deben ser finitas y estrictamente crecientes.')
    if len(coords) <= max_slices:
        return np.arange(len(coords))
    targets = np.linspace(coords[0] + .1*(coords[-1]-coords[0]),
                          coords[0] + .9*(coords[-1]-coords[0]), max_slices)
    return np.unique([int(np.argmin(abs(coords-t))) for t in targets])


def prepare_pixels(arrays, max_slices=8, size=224, invert=False):
    """P1-P99 sobre cortes seleccionados; escala 0..1, conserva aspecto y centra."""
    if not 0 < len(arrays) <= max_slices or size < 1:
        raise ValueError('Tamano o numero de cortes invalido.')
    volume = np.stack(arrays).astype(np.float32)
    if volume.ndim != 3 or not np.isfinite(volume).all():
        raise ValueError('Se requieren cortes 2D finitos y de igual tamano.')
    low, high = np.percentile(volume, [1, 99])
    if high <= low:
        raise ValueError('Serie sin rango de intensidad utilizable.')
    volume = np.clip((volume-low)/(high-low), 0, 1)
    if invert:
        volume = 1-volume
    images = np.zeros((max_slices, 1, size, size), dtype=np.float32)
    mask = np.zeros(max_slices, dtype=bool)
    for i, arr in enumerate(volume):
        h, w = arr.shape
        scale = size/max(h, w)
        nh, nw = max(1, round(h*scale)), max(1, round(w*scale))
        resized = np.asarray(Image.fromarray(arr).resize((nw, nh), Image.Resampling.BILINEAR))
        top, left = (size-nh)//2, (size-nw)//2
        images[i, 0, top:top+nh, left:left+nw] = resized
        mask[i] = True
    return images, mask, [float(low), float(high)]


def read_study(folder, study_uid, series_uid, max_slices=8, size=224):
    files = sorted(Path(folder).glob('*.dcm'))
    if not files:
        raise FileNotFoundError(f'Sin DICOM en {folder}')
    headers = [pydicom.dcmread(p, stop_before_pixels=True) for p in files]
    if any(str(d.StudyInstanceUID) != study_uid or str(d.SeriesInstanceUID) != series_uid for d in headers):
        raise ValueError('La serie contiene identificadores diferentes del manifiesto.')
    if len({str(d.SOPInstanceUID) for d in headers}) != len(headers):
        raise ValueError('SOPInstanceUID duplicados.')
    order, coords, method, _, warnings = spatial_order(headers)
    if method != 'posicion_espacial_DICOM':
        raise ValueError('El lector de entrenamiento requiere orden espacial comprobado.')
    if len(coords) > 2 and not np.allclose(np.diff(coords), np.median(np.diff(coords)), atol=.1, rtol=.05):
        raise ValueError('Espaciado irregular: revisar la serie antes de entrenar.')
    chosen = select_indices(coords, max_slices)
    arrays, selected_files, photometry = [], [], set()
    for i in chosen:
        path = files[int(order[i])]
        ds = pydicom.dcmread(path)
        if str(ds.get('Modality', '')) != 'MR' or int(ds.get('SamplesPerPixel', 1)) != 1:
            raise ValueError('Se requieren imagenes MR monocromaticas.')
        photometry.add(str(ds.get('PhotometricInterpretation', '')))
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim != 2:
            raise ValueError('Se esperaba un corte 2D.')
        arr = arr*float(ds.get('RescaleSlope', 1)) + float(ds.get('RescaleIntercept', 0))
        arrays.append(arr)
        selected_files.append(path.name)
    if len(photometry) != 1 or not photometry <= {'MONOCHROME1', 'MONOCHROME2'}:
        raise ValueError('Fotometria ausente o inconsistente.')
    images, mask, window = prepare_pixels(arrays, max_slices, size, photometry == {'MONOCHROME1'})
    metadata = {'StudyInstanceUID': study_uid, 'SeriesInstanceUID': series_uid,
                'cortes_disponibles': len(files), 'cortes_validos': int(mask.sum()),
                'indices_espaciales_1based': (chosen+1).tolist(), 'archivos': selected_files,
                'orden': method, 'ventana_p1_p99': window, 'advertencias': warnings}
    return images, mask, metadata
