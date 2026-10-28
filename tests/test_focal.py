import unittest
import tempfile
from pathlib import Path
import torch
from src.study_resnet import StudyResNet18
from src.study_cnn import StudyCNN
from src.masked_losses import masked_bce, masked_focal
from src.training_engine import masked_bce as reference_bce, epoch_pass, save_checkpoint, load_checkpoint
from src.training_contract import TrainConfig


class IntegrationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_bce_equals_common_engine(self):
        x = torch.randn(2,12)
        y = torch.randint(0,2,(2,12)).float()
        mask = torch.rand(2,12) > .4
        self.assertTrue(torch.equal(masked_bce(x,y,mask),reference_bce(x,y,mask)))

    def test_focal_gamma_zero_equals_bce(self):
        x = torch.randn(2,12)
        y = torch.randint(0,2,(2,12)).float()
        mask = torch.rand(2,12) > .4
        torch.testing.assert_close(masked_focal(x,y,mask,gamma=0,alpha=None),masked_bce(x,y,mask))

    def test_unknown_nan_targets_zero_gradient(self):
        mask = torch.tensor([[True,False]*6])
        y = torch.where(mask,torch.ones(1,12),torch.full((1,12),float("nan")))
        for criterion in (masked_bce,masked_focal):
            x = torch.zeros(1,12,requires_grad=True)
            loss = criterion(x,y,mask);loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertTrue((x.grad[~mask] == 0).all())
            self.assertTrue((x.grad[mask] != 0).all())

    def test_empty_mask_and_bad_parameters_rejected(self):
        x=torch.zeros(1,12);y=torch.zeros_like(x);mask=torch.zeros_like(x,dtype=torch.bool)
        for criterion in (masked_bce,masked_focal):
            with self.assertRaises(ValueError):criterion(x,y,mask)
        mask[:]=True
        for gamma in (-1,float("nan")):
            with self.assertRaises(ValueError):masked_focal(x,y,mask,gamma=gamma)
        with self.assertRaises(ValueError):masked_focal(x,y,mask,alpha=2)

    def test_resnet_ignores_padding_and_uses_mean_features(self):
        model=StudyResNet18(weights=None).eval()
        images=torch.rand(1,2,1,64,64);mask=torch.ones(1,2,dtype=torch.bool)
        with torch.no_grad():
            expected=model(images,mask)
            padded=torch.cat([images,torch.full((1,1,1,64,64),float("nan"))],dim=1)
            actual=model(padded,torch.tensor([[True,True,False]]))
            torch.testing.assert_close(actual,expected)
            rgb=images[mask].repeat(1,3,1,1)
            features=model.backbone((rgb-model.imagenet_mean)/model.imagenet_std)
            torch.testing.assert_close(expected,model.classifier(features.mean(0,keepdim=True)))
        self.assertEqual(tuple(expected.shape),(1,12))
        with self.assertRaises(ValueError):model(images,torch.zeros_like(mask))

    def test_focal_resnet_step_through_common_engine(self):
        model=StudyResNet18(weights=None)
        before=model.backbone.conv1.weight.detach().clone()
        batch={"images":torch.rand(1,2,1,64,64),"slice_mask":torch.ones(1,2,dtype=torch.bool),
               "targets":torch.tensor([[1.,0.]*6]),"label_mask":torch.ones(1,12,dtype=torch.bool),
               "StudyInstanceUID":["TEST"],"SeriesInstanceUID":["TEST_SERIES"],"split":["train"]}
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        result=epoch_pass(model,[batch],torch.device("cpu"),masked_focal,optimizer)
        self.assertEqual(result["pasos"],1)
        self.assertFalse(torch.equal(before,model.backbone.conv1.weight))
        after={k:v.detach().clone() for k,v in model.state_dict().items()}
        epoch_pass(model,[batch],torch.device("cpu"),masked_bce)
        for k,v in model.state_dict().items():torch.testing.assert_close(v,after[k],rtol=0,atol=0)
        config=TrainConfig(max_epochs=1,loss_name="focal")
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"checkpoint.pt"
            save_checkpoint(path,model,optimizer,1,config,{"fixture":"TEST"},.7,0)
            restored=StudyResNet18(weights=None)
            opt2=torch.optim.Adam(restored.parameters(),lr=1e-4)
            load_checkpoint(path,restored,opt2,config,{"fixture":"TEST"},torch.device("cpu"),restore_random=False)
            for k,v in model.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[k],rtol=0,atol=0)
            self.assertTrue(opt2.state)

    def test_cnn_accepts_external_bce(self):
        model=StudyCNN()
        batch={"images":torch.rand(1,2,1,64,64),"slice_mask":torch.ones(1,2,dtype=torch.bool),
               "targets":torch.tensor([[1.,0.]*6]),"label_mask":torch.ones(1,12,dtype=torch.bool),
               "StudyInstanceUID":["TEST"],"SeriesInstanceUID":["TEST_SERIES"],"split":["train"]}
        result=epoch_pass(model,[batch],torch.device("cpu"),masked_bce,torch.optim.Adam(model.parameters()))
        self.assertEqual(result["etiquetas_conocidas"],12)
