"""Medicion de recursos con supervision debil; no selecciona modelos academicos."""
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .study_reader import read_study
from .training_engine import (StudyDataset, seed_everything, epoch_pass, make_loader,
                              cpu_tree, save_checkpoint, load_checkpoint, sync)
from .training_contract import TrainConfig, require_release


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare_cache(bundle, release, manifest, data_root, references, out):
    require_release(bundle, release)
    by_uid = {r['StudyInstanceUID']: r for r in references}
    assert set(by_uid) == set(bundle.study_uids)
    rows = {r['StudyInstanceUID']: r for r in manifest}
    start = time.perf_counter()
    datasets = {split: StudyDataset(bundle, split, data_root, read_study, max_cache_studies=300)
                for split in ('train', 'dev')}
    details = []
    for split, dataset in datasets.items():
        for i, uid in enumerate(dataset.view['StudyInstanceUID']):
            sid = dataset.series[uid]
            folder = dataset.root / 'train_series' / uid / sid
            reference = by_uid[uid]
            assert reference['SeriesInstanceUID'] == sid
            assert len(list(folder.glob('*.dcm'))) == int(rows[uid]['instances'])
            for selected in reference['archivos_seleccionados']:
                path = dataset.root / selected['member']
                assert path.resolve().is_relative_to(dataset.root)
                assert path.stat().st_size == selected['bytes'] and digest(path) == selected['sha256']
            images, mask, meta = read_study(folder, uid, sid)
            assert meta['indices_espaciales_1based'] == reference['indices_espaciales_1based']
            assert meta['archivos'] == [Path(r['member']).name for r in reference['archivos_seleccionados']]
            assert images.shape == (8, 1, 224, 224) and mask.sum() == 8
            assert not meta['advertencias']
            dataset.cache[uid] = (images, mask)
            data_hash = hashlib.sha256(images.tobytes()).hexdigest()
            details.append({'StudyInstanceUID': uid, 'SeriesInstanceUID': sid, 'split': split,
                'cortes': int(mask.sum()), 'entrada_sha256': data_hash,
                'entrada_bit_a_bit_igual_local': data_hash == reference['entrada_sha256'],
                'archivos_seleccionados_sha256_coinciden': True})
            if (i + 1) % 20 == 0 or i + 1 == len(dataset):
                print(f'{split}: {i+1}/{len(dataset)} entradas verificadas y en RAM CPU.', flush=True)
    result = {'estudios': len(details), 'train': len(datasets['train']), 'dev': len(datasets['dev']),
        'cortes': sum(r['cortes'] for r in details), 'segundos': time.perf_counter() - start,
        'RAM_entradas_MiB': sum(a.nbytes + m.nbytes for ds in datasets.values() for a, m in ds.cache.values()) / 1024**2,
        'diferencias_numericas_respecto_lector_local': sum(not r['entrada_bit_a_bit_igual_local'] for r in details),
        'nota_diferencias': 'Se registran posibles diferencias numericas entre versiones; los DICOM seleccionados deben ser identicos por SHA-256.',
        'estudios_detalle': details}
    write_json(Path(out) / 'preparacion_imagenes.json', result)
    return datasets, result


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left.cpu(), right.cpu(), rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right): assert_state_equal(a, b)
    else:
        assert left == right


def run_model(name, factory, restore_factory, datasets, bundle, release, out, provenance):
    require_release(bundle, release)
    device = torch.device('cuda:0')
    if not torch.cuda.is_available(): raise RuntimeError('El benchmark requiere GPU, sin reemplazo por CPU.')
    config = TrainConfig(seed=42, batch_size=1, learning_rate=1e-4, max_epochs=2, patience=3)
    target = Path(out) / name
    if target.exists(): raise ValueError(f'{name} ya tiene resultados; reiniciar la sesion para otra ejecucion limpia.')
    target.mkdir()
    gc.collect(); torch.cuda.empty_cache(); seed_everything(config.seed)
    start = time.perf_counter()
    model = factory().to(device)
    sync(device)
    init_seconds = time.perf_counter() - start
    # El calentamiento modifica BatchNorm en ResNet. Restaurar TODO el state_dict,
    # incluidos buffers, antes de crear el optimizador y medir las dos epocas.
    initial = cpu_tree(model.state_dict())
    batch = next(iter(DataLoader(datasets['train'], batch_size=1, shuffle=False)))
    start = time.perf_counter()
    model.train()
    logits = model(batch['images'].to(device), batch['slice_mask'].to(device))
    assert logits.shape == (1, 12) and bool(torch.isfinite(logits).all())
    from .training_engine import masked_bce
    loss = masked_bce(logits, batch['targets'].to(device), batch['label_mask'].to(device))
    loss.backward(); sync(device)
    warm_seconds = time.perf_counter() - start
    model.load_state_dict(initial, strict=True); model.zero_grad(set_to_none=True)
    assert_state_equal(initial, model.state_dict())
    del initial, logits, loss
    seed_everything(config.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []
    dev_loader = DataLoader(datasets['dev'], batch_size=1, shuffle=False, num_workers=0)
    for epoch in range(1, 3):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        train = epoch_pass(model, make_loader(datasets['train'], 1, config.seed + epoch), device, optimizer=optimizer)
        dev = epoch_pass(model, dev_loader, device)
        row = {'epoca': epoch, 'train': train, 'dev': dev,
            'estudios_train_por_segundo': len(datasets['train']) / train['segundos'],
            'vram_asignada_maxima_mib': torch.cuda.max_memory_allocated(device) / 1024**2,
            'vram_reservada_maxima_mib': torch.cuda.max_memory_reserved(device) / 1024**2}
        history.append(row)
        print(name, json.dumps(row, ensure_ascii=False), flush=True)
        write_json(target / 'historial_tecnico.json', history)
    # Ultima epoca tecnica: no se selecciona una corrida por su rendimiento.
    checkpoint = target / 'checkpoint_SOLO_BENCHMARK.pt'
    full_provenance = dict(provenance, modelo=name, tipo='benchmark_recursos_supervision_debil')
    save_checkpoint(checkpoint, model, optimizer, 2, config, full_provenance, history[-1]['dev']['masked_bce'], 0)
    model.eval()
    with torch.inference_mode():
        reference = model(batch['images'].to(device), batch['slice_mask'].to(device)).cpu()
    expected_model = cpu_tree(model.state_dict())
    expected_optimizer = cpu_tree(optimizer.state_dict())
    parameters = sum(p.numel() for p in model.parameters())
    del model, optimizer
    gc.collect(); torch.cuda.empty_cache()
    restored = restore_factory().to(device)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=config.learning_rate)
    load_checkpoint(checkpoint, restored, restored_optimizer, config, full_provenance, device, restore_random=False)
    assert_state_equal(expected_model, restored.state_dict())
    assert_state_equal(expected_optimizer, restored_optimizer.state_dict())
    restored.eval()
    with torch.inference_mode():
        recovered = restored(batch['images'].to(device), batch['slice_mask'].to(device)).cpu()
    torch.testing.assert_close(reference, recovered, atol=1e-6, rtol=1e-5)
    result = {'modelo': name, 'tipo': 'benchmark_recursos_no_resultado_predictivo',
        'config': config.validate(), 'parametros': parameters,
        'inicializacion_segundos': init_seconds, 'calentamiento_segundos': warm_seconds,
        'calentamiento_descartado_y_buffers_restaurados': True, 'epocas': history,
        'checkpoint_modelo_optimizador_y_salidas_verificados': True,
        'checkpoint_bytes': checkpoint.stat().st_size, 'checkpoint_sha256': digest(checkpoint),
        'usar_pesos_para_corridas_academicas': False, 'metricas_academicas': False,
        'metodo_tiempos': 'RAM CPU; incluye transferencia, BCE, controles de finitud y sincronizacion. Lectura/calentar/inicializar aparte.',
        'procedencia': full_provenance}
    write_json(target / 'resultado_benchmark.json', result)
    del restored, restored_optimizer, expected_model, expected_optimizer, reference, recovered, batch
    gc.collect(); torch.cuda.empty_cache()
    return result


def resource_projection(results, prep_seconds):
    # Estimacion BCE como proxy para focal/filtro; no fija presupuesto academico.
    seconds = {name: max(r['train']['segundos'] + r['dev']['segundos'] for r in value['epocas'])
               for name, value in results.items()}
    return {'base': 'Epoca mas lenta medida, train 240 y dev 60. Focal y filtro aproximados con ResNet BCE.',
        'margen': 1.5, 'estimaciones_no_presupuesto_aprobado': [
            {'epocas_por_configuracion': n,
             'horas_cuatro_configuraciones_con_margen': 1.5 * (n * (seconds['CNN_BCE'] + 3 * seconds['ResNet18_BCE'])
                + 4 * prep_seconds + results['CNN_BCE']['inicializacion_segundos']
                + 3 * results['ResNet18_BCE']['inicializacion_segundos']) / 3600}
            for n in (5, 10, 20)],
        'limites': 'No incluye horas humanas, esperas de Kaggle, descargas/exportaciones, evaluacion gold ni sobrecoste focal verificado. El filtro puede cambiar el tamano de train.',
        'decision_ABC': 'pendiente de revisar medicion, cobertura y horas humanas restantes'}
