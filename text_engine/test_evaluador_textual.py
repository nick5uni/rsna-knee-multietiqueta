import unittest
from copy import deepcopy
from evaluador_textual import evaluar_texto, CONDICIONES

def ejemplo():
    gold=[];anot=[]
    for uid,idioma in [('ARTIFICIAL_EN','en'),('ARTIFICIAL_ES','es')]:
        for i,c in enumerate(CONDICIONES):
            gold.append(dict(StudyInstanceUID=uid,condicion=c,gold=int(i==0),idioma=idioma))
            estado='positivo' if idioma=='en' else ('incierto' if i==0 else 'no_mencionado')
            anot.append(dict(StudyInstanceUID=uid,condicion=c,idioma=idioma,estado=estado,
                valor_candidato=1 if idioma=='en' else None,version_extractor='DEMO',
                regla_id='DEMO',habilitada_entrenamiento=False))
    congelacion=dict(reglas_congeladas=True,gold_consultado_previamente=True,
        fecha_congelacion_utc='DEMO_ARTIFICIAL',artefactos_sha256={'demo':'0'*64})
    return gold,anot,congelacion

class PruebasEvaluador(unittest.TestCase):
    def test_calculo_y_denominadores(self):
        r=evaluar_texto(*ejemplo())['totales'][0]
        self.assertEqual((r['pares'],r['resueltos'],r['abstenciones']),(24,12,12))
        self.assertEqual((r['TP'],r['TN'],r['FP'],r['FN']),(1,0,11,0))
        self.assertEqual(r['cobertura'],.5)
        self.assertAlmostEqual(r['tasa_acuerdo_resueltos'],1/12)
    def test_abstenciones_separadas_y_acuerdo_no_definido(self):
        r=evaluar_texto(*ejemplo())['totales'][2]
        self.assertEqual((r['inciertos'],r['no_mencionados']),(1,11))
        self.assertIsNone(r['tasa_acuerdo_resueltos'])
    def test_orden_no_cambia_resultado(self):
        g,a,c=ejemplo();r=evaluar_texto(g,a,c)
        self.assertEqual(r,evaluar_texto(list(reversed(g)),list(reversed(a)),c))
    def test_inmutabilidad(self):
        datos=ejemplo();antes=deepcopy(datos);evaluar_texto(*datos)
        self.assertEqual(datos,antes)
    def test_falta_par_y_duplicado_bloqueados(self):
        g,a,c=ejemplo()
        with self.assertRaises(ValueError):evaluar_texto(g,a[:-1],c)
        with self.assertRaises(ValueError):evaluar_texto(g,a+[a[0]],c)
    def test_negativo_no_imputado(self):
        g,a,c=ejemplo();a[-1]['valor_candidato']=0
        with self.assertRaises(ValueError):evaluar_texto(g,a,c)
    def test_congelacion_y_declaracion_obligatorias(self):
        for campo in ['reglas_congeladas','gold_consultado_previamente']:
            g,a,c=ejemplo();c[campo]=False
            with self.assertRaises(ValueError):evaluar_texto(g,a,c)
    def test_idioma_fuera_alcance_y_sin_casos(self):
        g,a,c=ejemplo()
        for r in g:
            if r['idioma']=='es':r['idioma']='fr'
        r=evaluar_texto(g,a[:12],c)
        self.assertEqual(r['estudios_fuera_alcance'],{'fr':1})
        self.assertEqual(r['totales'][2]['pares'],0)
        self.assertIsNone(r['totales'][2]['cobertura'])
    def test_referencias_incompletas_e_idioma_erroneo(self):
        g,a,c=ejemplo()
        with self.assertRaises(ValueError):evaluar_texto(g[:-1],a,c)
        a[0]['idioma']='es'
        with self.assertRaises(ValueError):evaluar_texto(g,a,c)

if __name__=='__main__':unittest.main(verbosity=2)
