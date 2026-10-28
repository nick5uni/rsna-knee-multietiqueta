import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import ejecutar_kaggle as launcher

class KaggleLauncherChecks(unittest.TestCase):
    def test_search_does_not_walk_dicom_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'competition/train_series/deep').mkdir(parents=True)
            (root/'competition/train_series/deep/train.csv').write_text('hidden')
            (root/'competition/train.csv').write_text('visible')
            self.assertEqual(list(launcher.bounded_find(root,'train.csv')),[root/'competition/train.csv'])
    def test_discovery_requires_matching_csv_and_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'train_series').mkdir()
            (root/'train.csv').write_text('train');(root/'train_series.csv').write_text('series')
            with patch.object(launcher,'CSV_SHA',launcher.sha(root/'train.csv')),patch.object(launcher,'SERIES_SHA',launcher.sha(root/'train_series.csv')):
                self.assertEqual(launcher.discover_data(root),root)
            with self.assertRaises(RuntimeError):launcher.discover_data(root)
    def test_duplicate_sources_are_not_chosen_silently(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ['one','two']:
                d=root/name;d.mkdir();(d/'train_series').mkdir()
                (d/'train.csv').write_text('train');(d/'train_series.csv').write_text('series')
            with patch.object(launcher,'CSV_SHA',launcher.sha(root/'one/train.csv')),patch.object(launcher,'SERIES_SHA',launcher.sha(root/'one/train_series.csv')):
                with self.assertRaises(RuntimeError):launcher.discover_data(root)
    def test_config_uses_mounted_data_and_separate_outputs(self):
        config=launcher.make_config(Path('/input/competition'),Path('/working'))
        self.assertEqual(config['device'],'cuda:0')
        self.assertEqual(config['data_root'],str(Path('/input/competition')))
        self.assertEqual(config['output_root'],str(Path('/working/resultados')))
    def test_success_flag_alone_does_not_accredit_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'ejecucion.json').write_text(json.dumps({'estado':'OK','mode':'train','nuevo_entrenamiento':False}))
            with self.assertRaises(RuntimeError):launcher.check_complete(root,'train')
    def test_missing_checkpoint_rejects_complete_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'ejecucion.json').write_text(json.dumps({'estado':'OK','mode':'train','nuevo_entrenamiento':True}))
            (root/'entrenamiento_resumen.json').write_text(json.dumps({'CNN_BCE':{}}))
            with self.assertRaisesRegex(RuntimeError,'Falta archivo'):launcher.check_complete(root,'train')

if __name__=='__main__':unittest.main()
