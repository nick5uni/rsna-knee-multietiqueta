import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from src.batchnorm_policy import POLICY_ID, evaluation_forward, per_study_batchnorm
from src.training_engine import epoch_pass, fit, masked_bce
from src.training_contract import TrainConfig

class TinyStudy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, 1)
        self.bn = nn.BatchNorm2d(3)
        self.dropout = nn.Dropout(.8)
        self.classifier = nn.Linear(3, 12)
    def forward(self, images, mask):
        valid = images[mask]
        features = self.dropout(self.bn(self.conv(valid))).mean((2,3))
        pooled = features.new_zeros((*mask.shape,3))
        pooled[mask] = features
        return self.classifier(pooled.sum(1)/mask.sum(1,keepdim=True))

class Fixture(Dataset):
    def __init__(self, split='dev'):
        self.split = split
        self.images = torch.rand((4,8,1,6,6),generator=torch.Generator().manual_seed(12))
    def __len__(self): return 4
    def __getitem__(self, i):
        mask = torch.ones(12,dtype=torch.bool); mask[::3] = False
        targets = torch.tensor([(i+j)%2 for j in range(12)],dtype=torch.float32)
        targets[~mask] = float('nan')
        return dict(images=self.images[i],slice_mask=torch.tensor([True]*6+[False]*2),
                    targets=targets,label_mask=mask,StudyInstanceUID=str(i),SeriesInstanceUID='s'+str(i),split=self.split)

class BNChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42); torch.set_num_threads(1)
        self.model = TinyStudy().eval()
        self.images = Fixture()[0]['images'].unsqueeze(0)
        self.mask = Fixture()[0]['slice_mask'].unsqueeze(0)
    def assert_state_equal(self, before, model):
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
    def test_matches_diagnostic_semantics(self):
        ref = copy.deepcopy(self.model)
        for m in ref.modules():
            if isinstance(m, nn.BatchNorm2d): m.track_running_stats=False; m.train()
        with torch.no_grad():
            expected=ref(self.images,self.mask)
            actual=evaluation_forward(self.model,self.images,self.mask,POLICY_ID)
        self.assertTrue(torch.equal(expected,actual))
    def test_parameters_buffers_and_flags_unchanged(self):
        before=copy.deepcopy(self.model.state_dict())
        flags=[(m.training,getattr(m,'track_running_stats',None)) for m in self.model.modules()]
        with torch.inference_mode(): evaluation_forward(self.model,self.images,self.mask,POLICY_ID)
        self.assert_state_equal(before,self.model)
        self.assertEqual(flags,[(m.training,getattr(m,'track_running_stats',None)) for m in self.model.modules()])
    def test_order_and_repetition_independent(self):
        data=Fixture()
        def run(order):
            with torch.no_grad():
                return {i:evaluation_forward(self.model,data[i]['images'][None],data[i]['slice_mask'][None],POLICY_ID) for i in order}
        first=run(range(4))
        for order in [list(reversed(range(4))),[2,0,3,1],list(range(4))]:
            other=run(order);self.assertTrue(all(torch.equal(first[i],other[i]) for i in first))
    def test_padding_is_ignored(self):
        changed=self.images.clone();changed[:,6:]=float('nan')
        with torch.no_grad():
            a=evaluation_forward(self.model,self.images,self.mask,POLICY_ID)
            b=evaluation_forward(self.model,changed,self.mask,POLICY_ID)
        self.assertTrue(torch.equal(a,b))
    def test_restores_flags_on_failure(self):
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError,'fixture'):
                with per_study_batchnorm(self.model,self.images,self.mask): raise RuntimeError('fixture')
        self.assertFalse(self.model.bn.training);self.assertTrue(self.model.bn.track_running_stats)
    def test_rejects_mixed_studies_and_unknown_policy(self):
        with torch.no_grad():
            with self.assertRaises(ValueError): evaluation_forward(self.model,self.images.repeat(2,1,1,1,1),self.mask.repeat(2,1),POLICY_ID)
            with self.assertRaises(ValueError): evaluation_forward(self.model,self.images,self.mask,'other')
    def test_rejects_training_gradients_empty_mask_and_no_bn(self):
        with self.assertRaises(ValueError): evaluation_forward(self.model,self.images,self.mask,POLICY_ID)
        with torch.no_grad():
            self.model.train()
            with self.assertRaises(ValueError): evaluation_forward(self.model,self.images,self.mask,POLICY_ID)
            self.model.eval()
            with self.assertRaises(ValueError): evaluation_forward(self.model,self.images,torch.zeros_like(self.mask),POLICY_ID)
            model=nn.Identity().eval()
            with self.assertRaises(ValueError): evaluation_forward(model,self.images,self.mask,POLICY_ID)
    def test_base_evaluation_is_unchanged(self):
        with torch.no_grad():
            self.assertTrue(torch.equal(self.model(self.images,self.mask),evaluation_forward(self.model,self.images,self.mask,None)))
    def test_training_step_unchanged_and_unknown_gradients_zero(self):
        a=copy.deepcopy(self.model);b=copy.deepcopy(self.model)
        loader=DataLoader(Fixture(),batch_size=1)
        torch.manual_seed(9);epoch_pass(a,loader,torch.device('cpu'),optimizer=torch.optim.Adam(a.parameters(),lr=1e-4))
        torch.manual_seed(9);epoch_pass(b,loader,torch.device('cpu'),optimizer=torch.optim.Adam(b.parameters(),lr=1e-4),evaluation_policy=POLICY_ID)
        self.assert_state_equal(a.state_dict(),b)
        logits=torch.randn(1,12,requires_grad=True);row=Fixture()[0]
        masked_bce(logits,row['targets'][None],row['label_mask'][None]).backward()
        self.assertTrue(torch.equal(logits.grad[~row['label_mask'][None]],torch.zeros(4)))
    def test_actual_resnet_matches_diagnostic_without_state_changes(self):
        from src.study_resnet import StudyResNet18
        model=StudyResNet18(weights=None).eval()
        reference=copy.deepcopy(model)
        for layer in reference.modules():
            if isinstance(layer,nn.BatchNorm2d):
                layer.track_running_stats=False;layer.train()
        before=copy.deepcopy(model.state_dict())
        images=torch.rand(1,8,1,64,64)
        mask=torch.tensor([[True]*6+[False]*2])
        with torch.inference_mode():
            expected=reference(images,mask)
            actual=evaluation_forward(model,images,mask,POLICY_ID)
        self.assertTrue(torch.equal(expected,actual))
        self.assertEqual(actual.shape,(1,12))
        self.assert_state_equal(before,model)

    def test_fit_selects_and_exports_using_same_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'
            summary=fit(self.model,Fixture('train'),Fixture('dev'),out,
                        TrainConfig(max_epochs=3,patience=2),{},device='cpu',
                        run_kind='technical_fixture',evaluation_policy=POLICY_ID)
            history=json.loads((out/'history.json').read_text())
            best=min(history,key=lambda r:r['dev']['masked_bce'])
            self.assertEqual(summary['selected_epoch'],best['epoch'])
            saved=torch.load(out/'best.pt',weights_only=True)
            self.assertTrue(saved['optimizer']['state'])
            self.assertEqual(saved['provenance']['evaluation_policy'],POLICY_ID)
            clone=TinyStudy();clone.load_state_dict(saved['model'],strict=True)
            check=epoch_pass(clone,DataLoader(Fixture(),batch_size=1),torch.device('cpu'),evaluation_policy=POLICY_ID)
            self.assertAlmostEqual(check['masked_bce'],summary['best_dev_bce'],places=7)
            with (out/'predictions_dev.csv').open(encoding='utf-8-sig') as handle:
                records=list(csv.DictReader(handle))
            from src.data_checks import LABELS
            clone.eval()
            with torch.no_grad():
                for i,record in enumerate(records):
                    row=Fixture()[i]
                    expected=torch.sigmoid(evaluation_forward(clone,row['images'][None],row['slice_mask'][None],POLICY_ID))[0]
                    self.assertEqual([float(record[c]) for c in LABELS],expected.tolist())
            with self.assertRaises(ValueError):
                fit(TinyStudy(),Fixture('train'),Fixture('dev'),out,TrainConfig(max_epochs=4),{},device='cpu',
                    run_kind='technical_fixture',resume=out/'last.pt',evaluation_policy=None)

if __name__=='__main__': unittest.main(verbosity=2)
