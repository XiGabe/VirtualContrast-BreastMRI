import numpy as np
import pytest
import torch

from virtualcontrast.generator import Generator
from virtualcontrast.inference import sliding_window_predict
from virtualcontrast.losses import generator_loss
from virtualcontrast.preprocess import prepare_breast
from virtualcontrast.data import load_case


def test_identity_initialization_and_learning():
    torch.set_num_threads(2)
    model=Generator()
    pre=torch.rand(1,1,16,32,32)
    delta=model(pre)
    assert torch.count_nonzero(delta)==0
    loss=(pre+delta-(pre+.1)).square().mean()
    loss.backward()
    assert model.output.weight.grad.abs().sum()>0
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    optimizer.step();optimizer.zero_grad()
    model(pre).square().mean().backward()
    assert model.encoder[0][0].weight.grad.abs().sum()>0
    assert not any('aux' in name for name,_ in model.named_parameters())


def test_sliding_window_padding():
    class Constant(torch.nn.Module):
        def forward(self,x):return torch.ones_like(x)*.25
    pre=np.zeros((5,19,22),dtype=np.float32)
    valid=np.ones_like(pre,dtype=bool);valid[:,:2]=False
    result=sliding_window_predict(Constant(),pre,valid,torch.device('cpu'),(8,16,16),.5,2)
    np.testing.assert_allclose(result[valid],.25,atol=1e-6)
    assert not result[~valid].any()


def test_padding_has_no_loss_or_gradient():
    delta=torch.zeros(1,1,4,8,8,requires_grad=True)
    batch={key:torch.zeros_like(delta) for key in ['pre','post','breast','valid','guidance','boundary','target']}
    batch['post'].fill_(100)
    loss,_=generator_loss(delta,batch)
    assert loss.item()==0
    loss.backward()
    assert not delta.grad.any()


def test_preprocessing_and_nc_only(tmp_path):
    pre=np.random.default_rng(2).random((32,40,16),dtype=np.float32)
    mask=np.ones_like(pre,dtype=bool)
    arrays,meta=prepare_breast(pre,None,mask,np.eye(4),(.01,.99))
    assert 'ce' not in arrays
    assert arrays['nc'].shape[1:]==(280,336)
    (tmp_path/'nc').mkdir()
    np.save(tmp_path/'nc/B0000.npy',arrays['nc'])
    import json
    (tmp_path/'nc/B0000.json').write_text(json.dumps(meta))
    result=load_case(tmp_path,{'sample_id':'B0000'},paired=False)
    assert not result['pre'][~result['valid']].any()
    assert 'post' not in result


def test_generation_resume_and_identity(tmp_path):
    import json
    from virtualcontrast.inference import generate
    class Constant(torch.nn.Module):
        def forward(self,x):return torch.ones_like(x)*.1
    (tmp_path/'nc').mkdir()
    np.save(tmp_path/'nc/B0.npy',np.ones((8,16,16),dtype=np.float16)*.2)
    (tmp_path/'nc/B0.json').write_text(json.dumps(dict(pad_lower=[0,0],resampled_shape_lr_ap=[16,16])))
    rows=[{'sample_id':'B0'}]
    args=(Constant(),rows,tmp_path,torch.device('cpu'),(8,16,16),1)
    generate(*args,checkpoint_hash='test-generator')
    output=tmp_path/'vce/B0.npy'
    content=output.read_bytes()
    generate(*args,checkpoint_hash='test-generator')
    assert output.read_bytes()==content
    with pytest.raises(ValueError,match='another generator'):
        generate(*args,checkpoint_hash='different-generator')
    np.save(tmp_path/'nc/B0.npy',np.ones((8,16,16),dtype=np.float16)*.3)
    with pytest.raises(ValueError,match='Invalid existing'):
        generate(*args,checkpoint_hash='test-generator')
