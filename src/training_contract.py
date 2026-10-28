"""Configuracion y comprobaciones del entrenador sin depender de PyTorch."""
from dataclasses import dataclass, asdict
import math

from .data_checks import LABELS


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    batch_size: int = 1
    learning_rate: float = 0.0001
    max_epochs: int = 10
    patience: int = 3
    loss_name: str = 'bce'

    def validate(self):
        for name in ('seed', 'batch_size', 'max_epochs', 'patience'):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == 'seed' else 1):
                raise ValueError(f'Parametro entero invalido: {name}.')
        if type(self.learning_rate) not in (int, float) or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate debe ser positivo y finito.')
        if self.seed >= 2**32:
            raise ValueError('seed debe ser menor a 2**32.')
        if not self.loss_name or not isinstance(self.loss_name, str):
            raise ValueError('Falta el nombre de la perdida.')
        return asdict(self)


def require_release(bundle, release):
    """La aceptacion de estrategia OA no habilita por si sola este contrato."""
    if release.get('protocol_accepted') is not True:
        raise ValueError('Falta aceptacion explicita del protocolo comun.')
    if release.get('labels_release_accepted') is not True or not release.get('labels_version'):
        raise ValueError('Falta una version de etiquetas habilitada bajo el protocolo.')
    if list(release.get('label_order', [])) != list(LABELS):
        raise ValueError('Orden de condiciones diferente de train.csv.')
    if release.get('supervision_sha256') != bundle.provenance:
        raise ValueError('La autorizacion no corresponde a estos archivos de supervision.')
    if not bundle.summary()['puede_iniciar_entrenamiento']:
        raise ValueError('Faltan etiquetas habilitadas, particiones o control de grupos.')
    train = bundle.training_view('train')
    dev = bundle.training_view('dev')
    if set(train['StudyInstanceUID']) & set(dev['StudyInstanceUID']):
        raise ValueError('Estudios compartidos por train y dev.')
    return train, dev
