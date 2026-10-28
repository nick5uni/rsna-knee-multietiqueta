"""Prueba tecnica de orquestacion; no acredita entrenamiento T4 real."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pipeline

class Bundle:
    def __init__(self,name):self.name=name;self.provenance={'version':name}
    def training_view(self,split):
        return {'StudyInstanceUID':['TRAIN' if split=='train' else 'DEV'],
                'targets':np.zeros((1,12)), 'label_mask':np.ones((1,12),dtype=bool)}

class Dataset:
    def __init__(self,bundle,split,*args,**kwargs):self.view=bundle.training_view(split);self.cache={}

class Model:
    def eval(self):pass

class TrainRoutingChecks(unittest.TestCase):
    def test_routes_four_configurations_and_does_not_restore_trained_weights(self):
        base=Bundle('base');filtered=Bundle('filtered');seen=[];initializations=[]
        cache={u:(np.zeros((8,1,224,224),dtype=np.float32),np.ones(8,dtype=bool)) for u in ['TRAIN','DEV','GOLD']}
        def read(path):
            if Path(path).name=='plan_gradcam.json':return {'cases':[]}
            return []
        def model_for(name,imagenet=False):initializations.append((name,imagenet));return Model()
        def fit(model,train,dev,out,config,provenance,**kwargs):
            Path(out).mkdir(parents=True,exist_ok=True)
            config.validate();seen.append((Path(out).name,config,kwargs))
            return {'selected_epoch':1,'best_dev_bce':.5,'sha256':{'best.pt':'a'*64,'last.pt':'b'*64}}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(pipeline,'read',side_effect=read), \
             patch.object(pipeline,'ensure_images',return_value=Path(tmp)), \
             patch.object(pipeline,'inputs',return_value=(cache,{'DEV':'S1','GOLD':'S2'})), \
             patch.object(pipeline,'model_for',side_effect=model_for), \
             patch.object(pipeline,'restore',side_effect=AssertionError('No recargar pesos entrenados')), \
             patch.object(pipeline,'predict',return_value=np.full((1,12),.25)), \
             patch('torch.cuda.is_available',return_value=True), \
             patch('torch.cuda.get_device_name',return_value='Tesla T4'), \
             patch('src.training_engine.StudyDataset',Dataset), \
             patch('src.training_engine.fit',side_effect=fit), \
             patch('src.evaluador_comun_v2.metrics',return_value={'technical_fixture':True}), \
             patch('src.gradcam_figures.render_overview'):
            checks=pipeline.inference({'device':'cuda:0'},Path(tmp),base,filtered,[],{'id':'base'},{'id':'filtered'},
                                      ['GOLD'],np.zeros((1,12)),np.ones((1,12),bool),{},Path(tmp)/'out',train=True)
        self.assertEqual([name for name,_,_ in seen],pipeline.KEYS)
        self.assertTrue(all(imagenet for _,imagenet in initializations))
        for name,config,kwargs in seen:
            self.assertEqual((config.seed,config.batch_size,config.learning_rate,config.max_epochs,config.patience),(42,1,.0001,20,3))
            self.assertIs(kwargs['bundle'],filtered if name.endswith('FILTRO') else base)
            self.assertEqual(kwargs['release']['id'],'filtered' if name.endswith('FILTRO') else 'base')
            self.assertEqual(config.loss_name,'focal' if name.endswith('FOCAL') else 'bce')
            if name.endswith('FOCAL'):self.assertEqual(kwargs['train_criterion'].keywords,{'gamma':2.,'alpha':None})
        self.assertEqual(set(checks),set(pipeline.KEYS))
        self.assertTrue(all(v['nuevo_entrenamiento'] for v in checks.values()))

if __name__=='__main__':unittest.main()
