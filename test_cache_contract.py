import ast, pathlib, tempfile, hashlib, json, os, contextlib, types, time
import torch
p=pathlib.Path(__file__).parent
a=ast.parse((p/'baseline.py').read_text(encoding='utf-8'))
b=ast.parse((p/'1_pre_cache_qwen_image21.py').read_text(encoding='utf-8'))
results=[]
def check(n,v):
    results.append({'name':n,'pass':bool(v)})
    print(('PASS ' if v else 'FAIL ')+n,flush=True)
def extract(tree,name,ns):
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<extracted>','exec'),ns)
    return ns[name]
class Pipe:
    def encode_prompt(self,**kw):
        return torch.arange(12,dtype=torch.float32).reshape(1,3,4).to(torch.bfloat16),None,torch.tensor([[1,0,1]])
class FakeImage:
    def convert(self,mode):return self
class VAE:
    dtype=torch.bfloat16
    config=types.SimpleNamespace(z_dim=64,latents_mean=[0.0]*64,latents_std=[1.0]*64)
    def encode(self,x):
        return types.SimpleNamespace(latent_dist=types.SimpleNamespace(mode=lambda:torch.arange(64*4*6,dtype=torch.float32).reshape(1,64,1,4,6)/128))
with tempfile.TemporaryDirectory() as d:
    d=pathlib.Path(d)
    for label,tree in [('baseline',a),('candidate',b)]:
        c=d/label;c.mkdir()
        ns=dict(torch=torch,os=os,CACHE_DIR=str(c),t=lambda x,**kw:x.format(**kw),Image=types.SimpleNamespace(open=lambda x:FakeImage()),fit=lambda x,w,h:x,F_vision=types.SimpleNamespace(pil_to_tensor=lambda x:torch.zeros(4,64,96,dtype=torch.uint8)))
        text=extract(tree,'encode_and_save',ns)
        vae=extract(tree,'encode_latents',ns)
        for name,images in [('_neg',None),('normal',None),('edit',[object()]),('_custom',[object()]),('_custom2',None)]:
            check(label+' '+name+' token count',text(Pipe(),'fixture',name,images,device='cpu')==3)
        jobs=[('normal.png','normal_latent.pt',(96,64)),('edit.png','edit_latent.pt',(96,64)),('before.png','edit_ctrl.pt',(96,64)),('ref.png','_custom_ctrl.pt',(96,64))]
        vae(VAE(),jobs,'cpu')
        check(label+' no leftover temporary files',not list(c.glob('*.tmp')))
    ca=d/'candidate';ba=d/'baseline'
    names={f.name for f in ca.glob('*.pt')}
    expected={'_neg_embed.pt','_neg_mask.pt','normal_embed.pt','normal_mask.pt','edit_embed.pt','edit_mask.pt','edit_imgmask.pt','_custom_embed.pt','_custom_mask.pt','_custom_imgmask.pt','_custom2_embed.pt','_custom2_mask.pt','normal_latent.pt','edit_latent.pt','edit_ctrl.pt','_custom_ctrl.pt'}
    check('normal/edit/preview complete exact expected filenames',names==expected=={f.name for f in ba.glob('*.pt')})
    for n in sorted(expected):
        x=torch.load(ba/n,map_location='cpu',weights_only=True);y=torch.load(ca/n,map_location='cpu',weights_only=True)
        check(n+' exact tensor equality',x.dtype==y.dtype and x.shape==y.shape and torch.equal(x,y))
        if n.endswith('_embed.pt'): check(n+' BF16 embedding shape',y.dtype==torch.bfloat16 and tuple(y.shape)==(1,3,4))
        elif n.endswith('mask.pt'):check(n+' bool mask shape',y.dtype==torch.bool and tuple(y.shape)==(1,3))
        else:check(n+' BF16 latent shape',y.dtype==torch.bfloat16 and tuple(y.shape)==(1,64,4,6))
    # Execute actual extracted VAE scheduling code with synthetic cached tensors.
    jobs=[]
    for n in ['normal_latent.pt','edit_ctrl.pt','_custom_ctrl.pt']:
        src=d/(n+'.png');src.write_bytes(b'fixture');os.utime(src,(1,1));jobs.append((str(src),n,(96,64)))
    stale=ca/'_custom_ctrl_stale.pt';stale.write_bytes(b'fixture')
    calls=[]
    class Loader:
        @classmethod
        def from_pretrained(cls,*args,**kw):return cls()
        def to(self,device):return self
    ns=dict(torch=torch,os=os,CACHE_DIR=str(ca),MODEL_ID='fixture',t=lambda x,**kw:x.format(**kw),AutoencoderKLQwenImage21=Loader,encode_latents=lambda vae,pending,device:calls.extend(pending),free_vram=lambda:None)
    extract(b,'remove_stale_custom_ctrls',ns)
    phase=extract(b,'run_vae_phase',ns)
    phase(jobs,{'_custom_ctrl.pt'})
    check('fresh normal and edit latents reused, preview always regenerated',[x[1] for x in calls]==['_custom_ctrl.pt'])
    check('stale preview reference removed',not stale.exists())
    calls.clear();torch.save(torch.zeros(1,64,2,2,dtype=torch.bfloat16),ca/'normal_latent.pt')
    phase(jobs,{'_custom_ctrl.pt'})
    check('incorrect latent shape scheduled for regeneration',[x[1] for x in calls]==['normal_latent.pt','_custom_ctrl.pt'])
    calls.clear();os.utime(ca/'edit_ctrl.pt',(0,0))
    phase(jobs,{'_custom_ctrl.pt'})
    check('older cache scheduled for regeneration',[x[1] for x in calls]==['normal_latent.pt','edit_ctrl.pt','_custom_ctrl.pt'])
receipt={'scope':'Real CPU PyTorch serialization using source-extracted encoding functions and synthetic model outputs; not actual Qwen model or application end-to-end','torch':torch.__version__,'hip':torch.version.hip,'candidate_sha256':hashlib.sha256((p/'1_pre_cache_qwen_image21.py').read_bytes()).hexdigest(),'results':results,'passed':sum(x['pass'] for x in results),'failed':sum(not x['pass'] for x in results)}
(p/'CACHE_TEST_RECEIPT.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k!='results'}))
raise SystemExit(bool(receipt['failed']))
