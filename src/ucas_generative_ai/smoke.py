"""Meaningful small CPU checks; no MNIST download or real training runs."""
import contextlib
import copy
import io
import itertools
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .config import experiment_config,seed_all
from .models import ChannelLayerNorm,MaskedConv2d,PixelCNN,GatedPixelCNN,RegularizedGatedPixelCNN
from .sampling import sample_images
from .training import TrainingSession,update_parameter_average,load_training_state


def assert_exact(first,second):
    if torch.is_tensor(first):
        assert torch.is_tensor(second) and torch.equal(first,second)
    elif isinstance(first,dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_exact(first[key],second[key])
    elif isinstance(first,(tuple,list)):
        assert len(first) == len(second)
        for left,right in zip(first,second):
            assert_exact(left,right)
    else:
        assert first == second,(first,second)


def stable_state(state):
    result = copy.deepcopy(state)
    result.pop("training_seconds",None)
    for row in result.get("history",[]):
        row.pop("seconds",None)
    return result


def run_smoke():
    torch.set_num_threads(1)
    seed_all(103)
    results = {}
    for kind in ("A","B"):
        layer = MaskedConv2d(kind,1,1,3,padding=1,bias=False).double()
        with torch.no_grad():layer.weight.fill_(1.)
        image = torch.arange(25,dtype=torch.float64).reshape(1,1,5,5)
        expected = 6+7+8+11+(12 if kind == "B" else 0)
        output = layer(image)[0,0,2,2]
        assert output.item() == expected
        output.backward()
        assert not layer.weight.grad[layer.mask == 0].count_nonzero()
    results["mask_forward_and_zero_excluded_gradient"] = "PASS"
    norm = ChannelLayerNorm(4).double()
    values = torch.randn(2,4,3,3,dtype=torch.float64)
    before = norm(values)
    altered = values.clone();altered[1] += torch.randn_like(altered[1])*10
    altered[0,:,2,2] += torch.tensor([1.,3.,8.,-2.],dtype=torch.float64)
    assert torch.equal(before[0,:,:2,:2],norm(altered)[0,:,:2,:2])
    reference = (values-values.mean(1,keepdim=True))/(values.var(1,unbiased=False,keepdim=True)+norm.norm.eps).sqrt()
    assert torch.allclose(before,reference,atol=1e-12,rtol=0)
    results["normalization_channels_only"] = "PASS"
    for constructor in (lambda:PixelCNN(4,2),lambda:GatedPixelCNN(4,2,[1,2]),
                        lambda:RegularizedGatedPixelCNN(4,2,[1,2],.2)):
        model = constructor().double()
        pixels = torch.randn(2,1,4,4,dtype=torch.float64)
        for training in (True,False):
            model.train(training)
            for index in (0,1,3,4,9,15):
                changed = pixels.clone();changed.flatten(1)[:,index:] += 3.125
                state = torch.get_rng_state()
                with torch.no_grad():before = model.logits(pixels).flatten(1)[:,:index+1]
                torch.set_rng_state(state)
                with torch.no_grad():after = model.logits(changed).flatten(1)[:,:index+1]
                assert torch.equal(before,after)
            variable = pixels[:1].clone().requires_grad_(True)
            gradient, = torch.autograd.grad(model.logits(variable).flatten()[9],variable)
            assert not gradient.flatten()[9:].count_nonzero()
        model.eval()
        patterns = torch.tensor(list(itertools.product([0.,1.],repeat=4)),dtype=torch.float64).reshape(16,1,2,2)
        with torch.no_grad():
            total = (-F.binary_cross_entropy_with_logits(model.logits(patterns),patterns,reduction="none").flatten(1).sum(1)).exp().sum().item()
        assert abs(total-1.) < 1e-12
    results["standard_gated_dropout_causality_and_joint_normalization"] = "PASS"

    class BoundaryModel(nn.Module):
        def __init__(self,p):super().__init__();self.p=p;self.calls=0
        def forward(self,x):
            if self.p == 1:
                flat=x.flatten(1)
                assert torch.equal(flat[:,:self.calls],torch.ones_like(flat[:,:self.calls]))
                assert not flat[:,self.calls:].count_nonzero()
            self.calls += 1
            return torch.full_like(x,self.p)
    for p in (0.,1.):
        boundary=BoundaryModel(p)
        images,frames,_=sample_images(boundary,2,seed=109,capture=True,device="cpu")
        assert boundary.calls == 784 and len(frames) == 5
        assert torch.equal(images,torch.full_like(images,p))
        if p == 1:
            assert [int(frame[0].sum()) for frame in frames] == [0,196,392,588,784]
    results["sampler_first_row_column_and_all_784_positions"] = "PASS"
    raw = RegularizedGatedPixelCNN(4,2,[1,2],.2).double()
    averaged = copy.deepcopy(raw)
    original = {name:value.clone() for name,value in averaged.named_parameters()}
    with torch.no_grad():
        for value in raw.parameters():value.add_(.03125)
    update_parameter_average(averaged,raw,.875)
    for name,value in averaged.named_parameters():
        assert torch.equal(value,original[name]*.875+dict(raw.named_parameters())[name]*.125)
    for name,value in averaged.named_buffers():assert torch.equal(value,dict(raw.named_buffers())[name])
    results["ema_parameter_formula_and_buffer_copy"] = "PASS"

    synthetic = torch.randint(0,2,(12,1,4,4),generator=torch.Generator().manual_seed(127)).float()
    training_data = TensorDataset(synthetic[:8],torch.zeros(8,dtype=torch.long))
    validation_data = TensorDataset(synthetic[8:],torch.zeros(4,dtype=torch.long))
    config = dict(experiment_config("s2"),channels=4,n_blocks=2,batch_size=4,epochs=4,
                  n_train=8,n_val=4,image_numel=16,seed=103,split_seed=103)
    def loaders():
        training = DataLoader(training_data,batch_size=4,shuffle=True,
                              generator=torch.Generator().manual_seed(104))
        validation = DataLoader(validation_data,batch_size=4,shuffle=False)
        return training,validation
    with tempfile.TemporaryDirectory(prefix="pixelcnn-smoke-") as temporary:
        root=Path(temporary)
        def standard_run(directory,epochs,resume=False):
            seed_all(config["seed"])
            train_loader,val_loader=loaders()
            model=PixelCNN(4,2)
            optimizer=torch.optim.Adam(model.parameters(),lr=config["lr"])
            session=TrainingSession(config,val_loader,"cpu",directory)
            session.train(train_loader,model,optimizer,epochs,output_dir=directory,resume=resume)
            return session,model
        # Keep the same full cosine horizon while stopping the interrupted run at 2.
        with contextlib.redirect_stdout(io.StringIO()):
            standard_run(root/"standard-full",4)
            standard_run(root/"standard-resume",2)
            standard_session,standard_model=standard_run(root/"standard-resume",4,True)
        full=load_training_state(root/"standard-full/training_state.pt")
        resumed=load_training_state(root/"standard-resume/training_state.pt")
        assert_exact(stable_state(full),stable_state(resumed))
        assert len(standard_session.history)==4
        evaluated,values=standard_session.evaluate(standard_model,loaders()[1])
        assert evaluated["n"] == 4 and torch.isfinite(values).all()
        assert abs(evaluated["nll_nats_per_image"]-16*evaluated["bce_nats_per_pixel"]) < 1e-12
        results["standard_complete_epoch_resume_all_states_exact"] = "PASS"
        results["evaluation_image_pixel_units"] = "PASS"
        seed_all(131)
        initialization=GatedPixelCNN(4,2,[1,2]).state_dict()
        fine_config = dict(config,architecture="dilated_gated",dilations=[1,2],dropout_p=.2,
                           ema_decay=.999,early_stop_patience=5,early_stop_min_delta=.02,
                           plateau_patience=2,lr=1e-4,initial_train_nll=16.)
        def fine_run(directory,pause=False,resume=False):
            options=dict(fine_config)
            if pause:options["pause_after_epoch"]=2
            seed_all(options["seed"])
            train_loader,val_loader=loaders()
            model=RegularizedGatedPixelCNN(4,2,[1,2],.2)
            optimizer=torch.optim.Adam(model.parameters(),lr=options["lr"])
            session=TrainingSession(options,val_loader,"cpu",directory)
            session.train_finetuning(train_loader,model,optimizer,4,directory,initialization,resume=resume)
            return session,model
        with contextlib.redirect_stdout(io.StringIO()):
            fine_run(root/"fine-full")
            fine_run(root/"fine-resume",pause=True)
            fine_session,fine_model=fine_run(root/"fine-resume",resume=True)
        full=load_training_state(root/"fine-full/training_state.pt")
        resumed=load_training_state(root/"fine-resume/training_state.pt")
        assert_exact(stable_state(full),stable_state(resumed))
        assert resumed["finished"] and resumed["successful_steps"] == 8
        assert (root/"fine-resume/pixelcnn_epoch_01.pt").exists()
        best_before={name:value.clone() for name,value in fine_model.state_dict().items()}
        with contextlib.redirect_stdout(io.StringIO()):
            restored_session,restored_model=fine_run(root/"fine-resume",resume=True)
        assert len(restored_session.history) == len(fine_session.history)
        assert_exact(best_before,restored_model.state_dict())
        results["dropout_ema_complete_epoch_resume_all_states_exact"] = "PASS"
        results["finished_fine_run_restores_without_extra_updates"] = "PASS"
    assert experiment_config("s0")["epochs"]==25 and experiment_config("s0")["min_lr"]==1e-4
    assert experiment_config("s1")["batch_size"]==256
    assert experiment_config("s2")["batch_size"]==128 and experiment_config("s3")["lr"]==5e-4
    assert experiment_config("f0")["dropout_p"]==0 and experiment_config("f02")["dropout_p"]==.2
    results["recorded_protocols"] = "PASS"
    results["overall"] = "PASS"
    return results
