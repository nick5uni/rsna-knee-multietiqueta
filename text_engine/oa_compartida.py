"""Normalidad conjunta explícita del cartílago; mantiene el origen indirecto OA."""
import re

def normalidad_compartida(texto_normalizado,idioma):
    if idioma=='en':
        anatomia=r'(?:tibiofemoral|femorotibial)\s+and\s+(?:patellofemoral|femoropatellar)'
        tejido=r'(?:articular\s+)?cartilage'
        normal=r'(?:normal|unremarkable|(?:well[ -])?preserved|well[ -]maintained)'
        patrones=[rf'\b{anatomia}\s+{tejido}\s+(?:(?:is|appears)\s+)?{normal}\b',rf'\b{normal}\s+{anatomia}\s+{tejido}\b']
    elif idioma=='es':
        anatomia=r'(?:tibiofemoral|femorotibial)\s+y\s+(?:patelofemoral|femoropatelar|femororrotuliano)'
        tejido=r'cartilago(?:\s+articular)?'
        normal=r'(?:normal|conservado|preservado)'
        patrones=[rf'\b{tejido}\s+{anatomia}\s+(?:(?:es|esta)\s+)?{normal}\b',rf'\b{normal}\s+{tejido}\s+{anatomia}\b',rf'\b{tejido}\s+{normal}\s+{anatomia}\b']
    else:raise ValueError('Idioma fuera de alcance')
    return any(re.search(p,texto_normalizado) for p in patrones)
