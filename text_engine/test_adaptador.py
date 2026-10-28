import unittest
from copy import deepcopy
from adaptador_bilingue import AdaptadorBilingue, CONDICIONES

TEXTO = 'Ejemplo artificial A. Ejemplo artificial B. Duda artificial C.'
def artificial(_):
    # Resultados simulados, sin interpretar enfermedades ni reportes reales.
    filas = [dict(condicion=c, estado='no_mencionado', regla_id='DEMO_AUSENCIA', evidencia='') for c in CONDICIONES]
    for i,(estado,evidencia) in enumerate([
        ('positivo','Ejemplo artificial A.'),
        ('negativo','Ejemplo artificial B.'),
        ('incierto','Duda artificial C.')]):
        filas[i].update(estado=estado, evidencia=evidencia, regla_id='DEMO_'+str(i))
    return filas

class Comprobaciones(unittest.TestCase):
    def adaptador(self, funcion=artificial):
        a=AdaptadorBilingue(); a.registrar('es','DEMO_NO_CLINICA',funcion); return a
    def test_orden_y_cuatro_estados(self):
        r=self.adaptador().procesar('DEMO',TEXTO,'es')
        self.assertEqual([x['condicion'] for x in r],list(CONDICIONES))
        self.assertEqual([x['estado'] for x in r[:4]],['positivo','negativo','incierto','no_mencionado'])
    def test_desconocidos_sin_valor_ni_habilitacion(self):
        r=self.adaptador().procesar('DEMO',TEXTO,'es')
        self.assertEqual([x['valor_candidato'] for x in r[:4]],[1,0,None,None])
        self.assertFalse(any(x['habilitada_entrenamiento'] for x in r))
    def test_version_y_texto_conservados(self):
        r=self.adaptador().procesar('DEMO',TEXTO,'es')
        self.assertEqual(r[0]['version_extractor'],'DEMO_NO_CLINICA')
        self.assertEqual(r[0]['evidencia'],'Ejemplo artificial A.')
        self.assertEqual(len(r[0]['reporte_sha256']),64)
    def test_idioma_sin_extractor_bloqueado(self):
        with self.assertRaises(ValueError): self.adaptador().procesar('DEMO',TEXTO,'en')
    def test_reporte_vacio_bloqueado(self):
        with self.assertRaises(ValueError): self.adaptador().procesar('DEMO',' ','es')
    def test_condicion_duplicada_bloqueada(self):
        filas=artificial(TEXTO);filas[1]['condicion']='ACL'
        with self.assertRaises(ValueError): self.adaptador(lambda _:filas).procesar('DEMO',TEXTO,'es')
    def test_evidencia_ajena_bloqueada(self):
        filas=artificial(TEXTO);filas[0]['evidencia']='Texto inventado fuera del reporte'
        with self.assertRaises(ValueError): self.adaptador(lambda _:filas).procesar('DEMO',TEXTO,'es')
    def test_doce_resultados_obligatorios(self):
        with self.assertRaises(ValueError): self.adaptador(lambda _:[]).procesar('DEMO',TEXTO,'es')
    def test_entrada_no_modificada_y_estados_invalidos(self):
        filas=artificial(TEXTO);antes=deepcopy(filas)
        self.adaptador(lambda _:filas).procesar('DEMO',TEXTO,'es')
        self.assertEqual(filas,antes)
        filas[0]['estado']='otro'
        with self.assertRaises(ValueError): self.adaptador(lambda _:filas).procesar('DEMO',TEXTO,'es')

if __name__=='__main__': unittest.main(verbosity=2)
