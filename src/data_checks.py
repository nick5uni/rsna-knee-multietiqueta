"""Validaciones de supervision; una etiqueta vacia nunca equivale a cero."""
LABELS = ('ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
          'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture')

def parse_label(value):
    text = str(value).strip()
    if text == '':
        return None
    if text not in ('0', '1'):
        raise ValueError(f'Etiqueta no binaria: {text!r}')
    return int(text)

def supervision(row):
    values = tuple(parse_label(row[label]) for label in LABELS)
    known = sum(value is not None for value in values)
    return ('gold' if known == len(LABELS) else 'unlabeled' if known == 0 else 'partial'), values

def assert_disjoint(*groups):
    seen = set()
    for group in groups:
        current = set(group)
        if seen & current:
            raise ValueError('Hay estudios compartidos entre particiones.')
        seen.update(current)
