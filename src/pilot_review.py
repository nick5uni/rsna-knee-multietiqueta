import numpy as np

def spatial_order(datasets):
    """Orden fisico; InstanceNumber se admite solo como alternativa explicita."""
    warnings = []
    geometry = all('ImageOrientationPatient' in d and 'ImagePositionPatient' in d for d in datasets)
    normal = None
    if geometry:
        orientation = np.asarray(datasets[0].ImageOrientationPatient, dtype=float)
        geometry = orientation.shape == (6,) and np.isfinite(orientation).all()
        geometry = geometry and all(np.allclose(np.asarray(d.ImageOrientationPatient, dtype=float),
                                                orientation, atol=1e-4, rtol=0) for d in datasets)
    if geometry:
        normal = np.cross(orientation[:3], orientation[3:])
        if np.linalg.norm(normal) < 1e-6:
            raise ValueError('Orientacion DICOM degenerada.')
        normal /= np.linalg.norm(normal)
        positions = np.asarray([d.ImagePositionPatient for d in datasets], dtype=float)
        if positions.shape != (len(datasets), 3) or not np.isfinite(positions).all():
            raise ValueError('Posiciones DICOM invalidas.')
        coords = positions @ normal
        method = 'posicion_espacial_DICOM'
    else:
        if not all('InstanceNumber' in d for d in datasets):
            raise ValueError('Sin metadatos suficientes para ordenar.')
        coords = np.asarray([int(d.InstanceNumber) for d in datasets], dtype=float)
        method = 'InstanceNumber_alternativa'
        warnings.append('Orden espacial sin comprobar: geometria ausente o inconsistente.')
    order = np.argsort(coords, kind='stable')
    ordered = coords[order]
    if not np.isfinite(ordered).all() or (len(ordered) > 1 and np.any(np.diff(ordered) <= 1e-4)):
        raise ValueError('Posiciones o numeros de corte duplicados; revisar la serie.')
    return order, ordered, method, normal, warnings
