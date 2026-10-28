"""Integración textual aislada. No contiene reglas clínicas ni modifica REV2."""
from hashlib import sha256
from copy import deepcopy

CONDICIONES = (
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture'
)
ESTADOS = ('positivo', 'negativo', 'incierto', 'no_mencionado')

class AdaptadorBilingue:
    """Cada función registrada devuelve 12 registros; el envoltorio se ajustará
    a la interfaz real del extractor recibido, sin cambiar sus decisiones.
    Campos: condicion, estado, regla_id, evidencia (subcadena literal del reporte).
    """
    def __init__(self):
        self._extractores = {}

    def registrar(self, idioma, version, funcion):
        if idioma not in ('en', 'es') or not version or not callable(funcion):
            raise ValueError('Se requiere idioma en/es, versión y función válida.')
        if idioma in self._extractores:
            raise ValueError('Idioma ya registrado; crear otra instancia para otra versión.')
        self._extractores[idioma] = (version, funcion)

    def procesar(self, study_uid, texto, idioma):
        if not isinstance(study_uid, str) or not study_uid.strip():
            raise ValueError('Identificador de estudio vacío o inválido.')
        if not isinstance(texto, str) or not texto.strip():
            raise ValueError('Reporte vacío: registrar incidencia, no asignar negativos.')
        if idioma not in self._extractores:
            raise ValueError('Extractor no integrado para este idioma. No sustituir por otro.')
        version, funcion = self._extractores[idioma]
        registros = funcion(texto)
        if not isinstance(registros, (list, tuple)) or len(registros) != 12:
            raise ValueError('El extractor debe entregar exactamente 12 condiciones.')
        por_condicion = {}
        for registro in registros:
            if not isinstance(registro, dict):
                raise ValueError('Anotación inválida.')
            campos = ('condicion', 'estado', 'regla_id', 'evidencia')
            if any(k not in registro for k in campos):
                raise ValueError('Faltan campos de trazabilidad.')
            condicion, estado = registro['condicion'], registro['estado']
            evidencia, regla = registro['evidencia'], registro['regla_id']
            if condicion not in CONDICIONES or condicion in por_condicion:
                raise ValueError('Condición desconocida o duplicada.')
            if estado not in ESTADOS:
                raise ValueError('Estado no reconocido; no convertirlo automáticamente.')
            if not isinstance(evidencia, str) or not isinstance(regla, str) or not regla.strip():
                raise ValueError('Evidencia y regla deben ser textos; regla no vacía.')
            if estado != 'no_mencionado' and not evidencia.strip():
                raise ValueError('Una decisión explícita requiere evidencia.')
            if evidencia and evidencia not in texto:
                raise ValueError('La evidencia no corresponde literalmente al reporte.')
            def validar_cita(cita, offsets):
                if not isinstance(cita,str) or not isinstance(offsets,(list,tuple)) or len(offsets)!=2:
                    raise ValueError('Cita u offsets inválidos.')
                a,b=offsets
                if type(a) is not int or type(b) is not int or not 0<=a<b<=len(texto) or texto[a:b]!=cita:
                    raise ValueError('Offsets no corresponden al texto original.')
            if registro.get('offsets_originales') is not None:
                validar_cita(evidencia,registro['offsets_originales'])
            for campo in ('eventos','menciones_no_diagnosticas','bloqueos_indirectos'):
                if campo in registro:
                    if not isinstance(registro[campo],list):
                        raise ValueError('Los eventos deben ser una lista.')
                    for evento in registro[campo]:
                        if not isinstance(evento,dict):raise ValueError('Evento inválido.')
                        validar_cita(evento.get('fragmento_literal'),evento.get('offsets_originales'))
            por_condicion[condicion] = deepcopy(registro)
        huella = sha256(texto.encode('utf-8')).hexdigest()
        salida = []
        for condicion in CONDICIONES:
            r = por_condicion[condicion]
            estado = r['estado']
            adicionales={k:deepcopy(r[k]) for k in ('seccion','offsets_originales','eventos','menciones_no_diagnosticas','version_reglas','origen_decision','bloqueos_indirectos') if k in r}
            salida.append({
                **adicionales,
                'StudyInstanceUID': study_uid, 'idioma': idioma,
                'version_extractor': version, 'reporte_sha256': huella,
                'condicion': condicion, 'estado': estado,
                'valor_candidato': {'positivo': 1, 'negativo': 0}.get(estado),
                'conocida_candidata': estado in ('positivo', 'negativo'),
                'regla_id': r['regla_id'], 'evidencia': r['evidencia'],
                'habilitada_entrenamiento': False,
            })
        return salida
