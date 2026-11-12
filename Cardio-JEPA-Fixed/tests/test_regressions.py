import torch
from cardio_jepa.engine import InterleavedLoader, forward_batch
from cardio_jepa.data.longform import WindowSample, split_by_record
from cardio_jepa.eval.benchmarks import make_feature_fn
from cardio_jepa.eval.linear_probe import train_linear_probe
from cardio_jepa.config import smoke_config
from cardio_jepa.models import CardioJEPA

def test_empty_loader_weight_alignment():
    loader=InterleavedLoader([[],[1,2]],weights=[1,4])
    assert loader.weights==[1.] and list(loader)==[1,2]

def test_split_ignores_unused_records():
    samples=[WindowSample(1,0,0),WindowSample(4,0,1)]
    a,b=split_by_record(samples,20)
    assert len(a)==len(b)==1
    a,b=split_by_record(samples,20,held_out_records={4})
    assert a[0].record==1 and b[0].record==4

def test_mixed_missing_targets():
    class M:
        forward_predictor=True
        def __call__(self,x,future_window=None,l2_future=None,**kw):
            if future_window is not None:assert torch.all(future_window==7)
            if l2_future is not None:assert torch.all(l2_future==9)
            return {'loss':x.mean()}
    b={'ctx':torch.tensor([[1.],[3.],[5.]]),'has_future':torch.tensor([0,1,1]),'future':torch.tensor([[0.],[7.],[7.]]),'horizon_s':torch.ones(3),'has_l2':torch.tensor([0,0,1]),'l2_future':torch.tensor([[0.],[0.],[9.]]),'l2_horizon_s':torch.ones(3)}
    assert torch.isclose(forward_batch(M(),b)['loss'],torch.tensor(3.))

def test_context_all_windows():
    class M:
        def representation(self,x):return x.mean((1,2)).unsqueeze(1)
    assert make_feature_fn(M())(torch.tensor([[[[2.]],[[6.]]]])).item()==4

def test_singleton_probe():
    r=train_linear_probe(torch.randn(1,4),torch.ones(1,1),torch.randn(2,4),torch.tensor([[0.],[1.]]),1,epochs=1,device='cpu')
    assert r['y_score'].shape==(2,1)

def test_hierarchical_vq_and_inference():
    cfg=smoke_config();cfg.hierarchical.enabled=True;cfg.loss.anti_collapse='vq';cfg.loss.beta_reg=.1
    m=CardioJEPA(cfg.signal,cfg.model,cfg.loss,cfg.hierarchical,cfg.forward)
    c,t=cfg.signal.n_leads,cfg.signal.window_samples
    x=torch.randn(2,2,c,t);future=torch.randn(2,c,t)
    m(x,future_window=future,l2_future=future)['loss'].backward()
    assert m.anticipation_features(x,[10.]).shape[1]==2*(cfg.model.encoder_embed_dim+cfg.hierarchical.l2_embed_dim)

def test_runtime_estimator_preserves_weights_and_caps_steps():
    from cardio_jepa.engine import estimate_runtime
    cfg=smoke_config();cfg.optim.max_steps_per_epoch=1
    model=CardioJEPA(cfg.signal,cfg.model,cfg.loss,cfg.hierarchical,cfg.forward)
    before={k:v.clone() for k,v in model.state_dict().items()}
    c,t=cfg.signal.n_leads,cfg.signal.window_samples
    batch={'ctx':torch.randn(2,1,c,t),'has_future':torch.zeros(2,dtype=torch.bool)}
    result=estimate_runtime(model,[batch,batch],cfg,torch.device('cpu'),n_steps=1)
    assert result['steps_per_epoch']==1
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
