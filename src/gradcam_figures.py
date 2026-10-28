"""Figuras cientificas con escala compartida; sin elegir el corte por su mapa."""
from pathlib import Path
import numpy as np


def blend_rgb(image, cam, colors):
    """Compone pixeles RGB explicitamente; cam=0 conserva la entrada exacta."""
    image, cam, colors = (np.asarray(a, dtype=np.float32) for a in (image, cam, colors))
    if image.ndim != 2 or cam.shape != image.shape or colors.shape != (*image.shape, 3):
        raise ValueError('Formas invalidas para la superposicion.')
    if any(not np.isfinite(a).all() or (a < 0).any() or (a > 1).any() for a in (image, cam, colors)):
        raise ValueError('Entrada, mapa y colores deben estar en 0..1 y ser finitos.')
    gray = np.repeat(image[..., None], 3, axis=-1)
    alpha = .55 * cam[..., None]
    return np.clip((1-alpha)*gray + alpha*colors, 0, 1)


def render_case(folder, case, images, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = Path(folder); folder.mkdir(exist_ok=True, parents=True)
    image = np.asarray(images)[:, 0]
    cam = result['cam'].numpy()
    raw = result['cam_raw'].numpy()
    np.savez_compressed(folder / (case['case_id'] + '_mapas.npz'), images=image, cam=cam, cam_raw=raw)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(8):
        original = axes.flat[i]; overlay = axes.flat[i + 8]
        for ax in (original, overlay):
            ax.imshow(image[i], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax.axis('off')
        overlay.imshow(blend_rgb(image[i], cam[i], plt.get_cmap('magma')(cam[i])[..., :3]), interpolation='nearest')
        original.set_title(f'Entrada / corte {i+1}', fontsize=9)
        overlay.set_title(f'Grad-CAM / corte {i+1}', fontsize=9)
    p = float(result['probabilities'][0, case['class_index']])
    fig.suptitle(f"{case['case_id']} | {case['condition']} | {case['category_weak']} segun etiqueta debil\n"
                 f"Etiqueta={case['weak_label']} | p={p:.4f} | umbral=0.5 | BN por estudio", fontsize=12)
    note = 'Mapa positivo nulo; no se fabrica localizacion.' if result['zero_positive_map'] else 'Color relativo al maximo de los 8 cortes; no es probabilidad ni segmentacion.'
    fig.text(.5, .012, note + '\nEtiqueta a nivel de estudio; no se conoce la localizacion de la lesion.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .94))
    path = folder / (case['case_id'] + '_ocho_cortes.png')
    fig.savefig(path, dpi=170); plt.close(fig)
    return path.name


def render_overview(path, examples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(examples), 2, figsize=(7.5, 3 * len(examples)), squeeze=False)
    for row, (case, images, result, mask) in enumerate(examples):
        valid = np.flatnonzero(mask)
        index = int(valid[(len(valid)-1)//2])  # Fixed spatial middle, chosen without CAM.
        image = images[index, 0]; cam = result['cam'][index].numpy()
        for ax in axes[row]:
            ax.imshow(image, cmap='gray', vmin=0, vmax=1, interpolation='nearest'); ax.axis('off')
        axes[row, 1].imshow(blend_rgb(image, cam, plt.get_cmap('magma')(cam)[..., :3]), interpolation='nearest')
        axes[row, 0].set_title(f"{case['case_id']} {case['condition']} | corte central {index+1}", fontsize=9)
        p = float(result['probabilities'][0, case['class_index']])
        axes[row, 1].set_title(f"{case['category_weak']} debil | etiqueta={case['weak_label']} | p={p:.3f}", fontsize=9)
    fig.suptitle('Grad-CAM exploratorio en dev | ResNet18 focal BN v1', fontsize=12)
    fig.text(.5, .008, 'Corte central fijado antes de mapas. Revisar tambien los ocho cortes.\nNo constituye localizacion clinica validada ni evaluacion gold.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, .97)); fig.savefig(path, dpi=180); plt.close(fig)
