"""Entrenador por estudio. Las perdidas de Victor se reciben como funciones."""
import copy
import csv
import hashlib
import json
import math
import random
import sys
import time
import datetime
import importlib.metadata
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .batchnorm_policy import POLICY_ID, validate_policy, evaluation_forward
from .data_checks import LABELS
from .training_contract import TrainConfig, require_release


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class StudyDataset(Dataset):
    """Lee solo estudios habilitados del bundle y conserva los IDs en cada lote."""
    def __init__(self, bundle, split, data_root, reader, max_cache_studies=0):
        if type(max_cache_studies) is not int or max_cache_studies < 0:
            raise ValueError('Limite de cache invalido.')
        self.view = bundle.training_view(split)
        self.series = {u: s for u, s in zip(bundle.study_uids, bundle.series_uids)}
        self.split = split
        self.provenance = dict(bundle.provenance)
        self.root = Path(data_root).resolve()
        self.reader = reader
        self.limit = max_cache_studies
        self.cache = OrderedDict()

    def __len__(self): return len(self.view['indices'])

    def __getitem__(self, index):
        uid = self.view['StudyInstanceUID'][index]; sid = self.series[uid]
        if uid in self.cache:
            images, mask = self.cache.pop(uid)
            self.cache[uid] = (images, mask)
        else:
            folder = (self.root/'train_series'/uid/sid).resolve()
            if not folder.is_relative_to(self.root):
                raise ValueError('Ruta de serie fuera de la raiz de datos.')
            images, mask, _ = self.reader(folder, uid, sid)
            if images.shape != (8, 1, 224, 224) or mask.shape != (8,) or mask.dtype != bool:
                raise ValueError('El lector debe devolver [8,1,224,224] y mascara booleana [8].')
            if not np.isfinite(images).all() or not mask.any():
                raise ValueError(f'Entrada invalida para {uid}.')
            if self.limit:
                self.cache[uid] = (images, mask)
                while len(self.cache) > self.limit: self.cache.popitem(last=False)
        return {'images': torch.from_numpy(images), 'slice_mask': torch.from_numpy(mask),
                'targets': torch.from_numpy(self.view['targets'][index]),
                'label_mask': torch.from_numpy(self.view['label_mask'][index]),
                'StudyInstanceUID': uid, 'SeriesInstanceUID': sid, 'split': self.split}


def supervision(logits, targets, mask):
    if logits.shape != targets.shape or logits.shape != mask.shape or logits.ndim != 2 or logits.shape[1] != len(LABELS):
        raise ValueError('Se esperan logits/targets/mascara de forma [B,12].')
    if mask.dtype != torch.bool or not bool(mask.any()):
        raise ValueError('El lote no tiene etiquetas conocidas o mascara booleana.')
    if not bool(torch.isfinite(logits).all()): raise ValueError('Logits no finitos.')
    known = targets[mask]
    if not bool(torch.isfinite(known).all()) or not bool(((known == 0) | (known == 1)).all()):
        raise ValueError('Targets conocidos deben ser binarios y finitos.')
    # Protege incluso ante NaN de relleno en entradas externas. No crea negativos.
    return torch.where(mask, targets, torch.zeros_like(targets))


def masked_bce(logits, targets, mask):
    safe = supervision(logits, targets, mask)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], safe[mask])


def make_loader(dataset, batch_size, seed):
    # Orden generado de nuevo para cada epoca: estable incluso despues de reanudar.
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)


def sync(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)


def epoch_pass(model, loader, device, criterion=masked_bce, optimizer=None, *, evaluation_policy=None):
    validate_policy(evaluation_policy)
    training = optimizer is not None
    model.train(training)
    total_loss = 0.; total_bce = 0.; known_count = 0; skipped = 0; steps = 0
    sync(device); start = time.perf_counter()
    with torch.set_grad_enabled(training):
        for batch in loader:
            mask = batch['label_mask'].to(device)
            if mask.dtype != torch.bool: raise ValueError('Mascara de etiquetas no booleana.')
            count = int(mask.sum())
            if not count:
                skipped += len(batch['StudyInstanceUID'])
                continue
            images = batch['images'].to(device); slice_mask = batch['slice_mask'].to(device)
            targets = batch['targets'].to(device)
            if training: optimizer.zero_grad(set_to_none=True)
            logits = model(images, slice_mask) if training else evaluation_forward(model, images, slice_mask, evaluation_policy)
            safe = supervision(logits, targets, mask)
            loss = criterion(logits, safe, mask)
            bce = masked_bce(logits, safe, mask)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise ValueError('La perdida debe ser un escalar finito.')
            if training:
                loss.backward()
                for param in model.parameters():
                    if param.grad is not None and not bool(torch.isfinite(param.grad).all()):
                        raise ValueError('Gradientes no finitos.')
                optimizer.step()
                if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise ValueError('Parametros no finitos despues del paso.')
            total_loss += float(loss.detach())*count
            total_bce += float(bce.detach())*count
            known_count += count; steps += 1
    sync(device)
    if not known_count: raise ValueError('Ninguna etiqueta habilitada en esta epoca.')
    return {'loss': total_loss/known_count, 'masked_bce': total_bce/known_count,
            'etiquetas_conocidas': known_count, 'pasos': steps,
            'estudios_omitidos_lotes_sin_etiquetas': skipped,
            'segundos': time.perf_counter()-start}


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def rng_state():
    n = np.random.get_state()
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'python': random.getstate(), 'numpy': [n[0], n[1].tolist(), n[2], n[3], n[4]]}


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu()); random.setstate(state['python'])
    n = state['numpy']; np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def save_checkpoint(path, model, optimizer, epoch, config, provenance, best_bce, bad_epochs):
    state = {'schema_version': 1, 'model': cpu_tree(model.state_dict()),
             'optimizer': cpu_tree(optimizer.state_dict()), 'epoch': epoch,
             'config': config.validate(), 'label_order': list(LABELS), 'provenance': provenance,
             'best_dev_bce': best_bce, 'bad_epochs': bad_epochs, 'rng': cpu_tree(rng_state())}
    temporary = Path(str(path)+'.tmp')
    torch.save(state, temporary); temporary.replace(path)


def load_checkpoint(path, model, optimizer, config, provenance, device, restore_random=True):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['label_order'] != list(LABELS) or checkpoint['provenance'] != provenance:
        raise ValueError('Checkpoint con etiquetas o procedencia diferentes.')
    expected = config.validate(); previous = checkpoint['config']
    if any(previous[k] != v for k, v in expected.items() if k != 'max_epochs'):
        raise ValueError('Configuracion incompatible con el checkpoint.')
    if expected['max_epochs'] < checkpoint['epoch']:
        raise ValueError('El maximo de epocas es menor a las ya completadas.')
    model.load_state_dict(checkpoint['model'], strict=True); model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer'])
    # Optimizer.load_state_dict adapta los estados a sus parametros actuales.
    if restore_random: restore_rng(checkpoint['rng'])
    return checkpoint


def export_predictions(model, dataset, path, device, batch_size=1, *, evaluation_policy=None):
    validate_policy(evaluation_policy)
    if evaluation_policy == POLICY_ID and batch_size != 1:
        raise ValueError("BN por estudio requiere batch_size=1.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval(); seen = set(); count = 0
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f, torch.inference_mode():
        writer = csv.DictWriter(f, fieldnames=['StudyInstanceUID', 'SeriesInstanceUID', 'split', *LABELS])
        writer.writeheader()
        for batch in loader:
            logits = evaluation_forward(model, batch['images'].to(device), batch['slice_mask'].to(device), evaluation_policy)
            if logits.shape != (len(batch['StudyInstanceUID']), len(LABELS)) or not bool(torch.isfinite(logits).all()):
                raise ValueError('Salida invalida al exportar predicciones.')
            probabilities = torch.sigmoid(logits).cpu().tolist()
            for i, uid in enumerate(batch['StudyInstanceUID']):
                if uid in seen: raise ValueError('Prediccion duplicada para un estudio.')
                seen.add(uid)
                row = {'StudyInstanceUID': uid, 'SeriesInstanceUID': batch['SeriesInstanceUID'][i], 'split': batch['split'][i]}
                row.update(zip(LABELS, probabilities[i])); writer.writerow(row); count += 1
    return count


def fit(model, train_dataset, dev_dataset, out, config, provenance, *, device,
        train_criterion=masked_bce, resume=None, run_kind='academic', release=None, bundle=None, evaluation_policy=None):
    """Evalua y selecciona SIEMPRE por BCE dev, tambien al entrenar con focal."""
    config.validate()
    validate_policy(evaluation_policy)
    if evaluation_policy == POLICY_ID and config.batch_size != 1:
        raise ValueError("BN por estudio requiere batch_size=1.")
    if run_kind not in ('academic', 'technical_fixture'): raise ValueError('Tipo de ejecucion desconocido.')
    if config.loss_name != 'bce' and train_criterion is masked_bce:
        raise ValueError('Se declaro otra perdida pero no se proporciono su implementacion.')
    if run_kind == 'academic':
        if bundle is None or release is None: raise ValueError('Falta bundle/release del entrenamiento academico.')
        require_release(bundle, release)
        if not isinstance(train_dataset, StudyDataset) or not isinstance(dev_dataset, StudyDataset):
            raise ValueError('El flujo academico requiere StudyDataset trazable.')
        if train_dataset.split != 'train' or dev_dataset.split != 'dev':
            raise ValueError('Los datasets deben corresponder a train y dev.')
        if train_dataset.view['StudyInstanceUID'] != bundle.training_view('train')['StudyInstanceUID'] or dev_dataset.view['StudyInstanceUID'] != bundle.training_view('dev')['StudyInstanceUID']:
            raise ValueError('Los datasets no coinciden con el bundle autorizado.')
        for ds in (train_dataset, dev_dataset):
            view = bundle.training_view(ds.split)
            if ds.provenance != bundle.provenance or not np.array_equal(ds.view['targets'], view['targets']) or not np.array_equal(ds.view['label_mask'], view['label_mask']):
                raise ValueError('Targets, mascaras o procedencia diferentes del bundle autorizado.')
        if train_dataset.root != dev_dataset.root:
            raise ValueError('Train y dev requieren la misma raiz de imagenes.')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if resume is None and any(out.iterdir()):
        raise ValueError('Carpeta de corrida no vacia: usar otra o reanudar explicitamente.')
    device = torch.device(device)
    seed_everything(config.seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best = math.inf; bad = 0; start_epoch = 1; history = []
    full_provenance = dict(provenance, run_kind=run_kind)
    if evaluation_policy is not None:
        policy_source = Path(__file__).with_name('batchnorm_policy.py')
        config_path = policy_source.parent.parent/'protocolo_bn_v1.json'
        if not config_path.is_file():
            raise ValueError('Falta protocolo_bn_v1.json junto al directorio src.')
        policy_config = json.loads(config_path.read_text(encoding='utf-8'))
        if policy_config.get('policy_id') != evaluation_policy:
            raise ValueError('Configuracion BN incompatible.')
        for filename in ('batchnorm_policy.py', 'training_engine.py', 'study_resnet.py'):
            actual = hashlib.sha256(policy_source.with_name(filename).read_bytes()).hexdigest()
            if policy_config['codigo_sha256'].get(filename) != actual:
                raise ValueError('Codigo diferente del protocolo BN: '+filename)
        full_provenance['evaluation_policy'] = evaluation_policy
        full_provenance['BN_config_sha256'] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        full_provenance['BN_codigo_sha256'] = policy_config['codigo_sha256']
    if resume is not None:
        if Path(resume).resolve() != (out/'last.pt').resolve() or not (out/'best.pt').is_file():
            raise ValueError('Reanudar desde last.pt en la carpeta original, conservando best.pt.')
        checkpoint = load_checkpoint(resume, model, optimizer, config, full_provenance, device)
        start_epoch = checkpoint['epoch']+1; best = checkpoint['best_dev_bce']; bad = checkpoint['bad_epochs']
        history = json.loads((out/'history.json').read_text(encoding='utf-8'))
        if not history or history[-1]['epoch'] != checkpoint['epoch']:
            raise ValueError('Historial diferente del checkpoint; revisar corrida.')
    (out/'config.json').write_text(json.dumps(dict(config.validate(), run_kind=run_kind, provenance=full_provenance,
          label_order=list(LABELS)), ensure_ascii=False, indent=2), encoding='utf-8')
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    for epoch in range(start_epoch, config.max_epochs+1):
        if bad >= config.patience: break
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        train = epoch_pass(model, make_loader(train_dataset, config.batch_size, config.seed+epoch),
                           device, train_criterion, optimizer)
        dev = epoch_pass(model, dev_loader, device, masked_bce, evaluation_policy=evaluation_policy)
        improved = dev['masked_bce'] < best
        if improved: best = dev['masked_bce']; bad = 0
        else: bad += 1
        row = {'epoch': epoch, 'train': train, 'dev': dev, 'selected_best': improved,
               'vram_maxima_pytorch_mib': torch.cuda.max_memory_allocated(device)/1024**2 if device.type == 'cuda' else None}
        history.append(row)
        save_checkpoint(out/'last.pt', model, optimizer, epoch, config, full_provenance, best, bad)
        if improved: save_checkpoint(out/'best.pt', model, optimizer, epoch, config, full_provenance, best, bad)
        (out/'history.json').write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Epoca {epoch}: train={train["loss"]:.6f}, dev_BCE={dev["masked_bce"]:.6f}', flush=True)
    if not (out/'best.pt').exists(): raise ValueError('No existe checkpoint seleccionado.')
    checkpoint = load_checkpoint(out/'best.pt', model, optimizer, config, full_provenance, device, restore_random=False)
    counts = {split: export_predictions(model, ds, out/f'predictions_{split}.csv', device, config.batch_size, evaluation_policy=evaluation_policy)
              for split, ds in [('train', train_dataset), ('dev', dev_dataset)]}
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name != 'run_summary.json'}
    summary = {'run_kind': run_kind, 'selected_epoch': checkpoint['epoch'], 'selection': 'min_dev_masked_bce',
               'best_dev_bce': best, 'epochs_completed': len(history), 'predictions': counts,
               'sha256': hashes, 'academic_metrics_produced': False,
               'torch': str(torch.__version__), 'device': str(device),
               'fecha_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'python': sys.version.split()[0], 'numpy': np.__version__,
               'cuda_runtime': torch.version.cuda,
               'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
               'estudios_omitidos_sin_etiquetas_aprobadas': {
                   split: ds.view['estudios_omitidos_sin_etiquetas_aprobadas'] if isinstance(ds, StudyDataset) else []
                   for split, ds in [('train', train_dataset), ('dev', dev_dataset)]}}
    for name in ('pydicom', 'Pillow'):
        try: summary[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: summary[name] = None
    (out/'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary
