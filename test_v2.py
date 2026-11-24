import ast, pathlib, tempfile, subprocess, sys, json, hashlib, os, time, ctypes
p=pathlib.Path(__file__).parent
a=ast.parse((p/'baseline.py').read_text(encoding='utf-8'))
b=ast.parse((p/'1_pre_cache_qwen_image21.py').read_text(encoding='utf-8'))
def fn(tree,name): return next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
def dump(x): return ast.dump(x,include_attributes=False)
results=[]
def check(name,condition):
    results.append({'name':name,'pass':bool(condition)})
    print(('PASS ' if condition else 'FAIL ')+name,flush=True)
for name in ('encode_and_save','encode_latents','encode_preview_prompts','load_text_encoder','bucket_size','find_samples','remove_stale_custom_ctrls'):
    check(name+' unchanged',dump(fn(a,name))==dump(fn(b,name)))
old=fn(a,'preprocess_qwen_image21').body
new=fn(b,'preprocess_qwen_image21').body
idx=next(i for i,x in enumerate(old) if isinstance(x,ast.Assign) and any(isinstance(y,ast.Name) and y.id=='pending' for y in x.targets))
check('text and job planning unchanged',list(map(dump,old[:idx]))==list(map(dump,new[:idx])))
check('VAE operations unchanged',list(map(dump,old[idx:]))==list(map(dump,fn(b,'run_vae_phase').body)))
source='import os,sys,subprocess,tempfile,json,time\n'+'\n'.join(ast.unparse(fn(b,n)) for n in ('contain_windows_workers','precache_config_identity','run_isolated_precache'))
with tempfile.TemporaryDirectory() as d:
    d=pathlib.Path(d)
    fixture=d/'fixture.py'
    fixture.write_text(source+'''
if len(sys.argv)==1:
    sys.exit(run_isolated_precache())
phase,manifest=sys.argv[1:]
with open(os.environ["TRACE"],"a") as f:
    f.write(json.dumps([phase,os.getpid(),os.getppid()])+"\\n")
if phase.endswith("text"):
    if os.environ["MODE"]=="textfail": sys.exit(7)
    if os.environ["MODE"]=="cancel": time.sleep(60)
    if os.environ["MODE"]=="configchange":
        with open(os.environ["LORALAB_PRECACHE_CONFIG"],"w") as f: f.write("changed")
    if os.environ["MODE"]!="missing":
        with open(manifest,"x") as f: json.dump({"jobs":[["image","a_latent.pt",[512,512]]]},f)
else:
    with open(manifest) as f: assert json.load(f)["jobs"][0][1]=="a_latent.pt"
    if os.environ["MODE"]=="vaefail": sys.exit(9)
''',encoding='utf-8')
    def env(mode):
        cfg=d/(mode+'.config')
        cfg.write_text('original')
        return {**os.environ,'TRACE':str(d/(mode+'.jsonl')),'MODE':mode,'LORALAB_PRECACHE_CONFIG':str(cfg)}
    for mode,expected,count in [('ok',0,2),('textfail',1,1),('missing',1,1),('vaefail',1,2),('configchange',1,1)]:
        e=env(mode)
        r=subprocess.run([sys.executable,str(fixture)],env=e,capture_output=True,text=True,timeout=10)
        trace=pathlib.Path(e['TRACE'])
        events=[json.loads(x) for x in trace.read_text().splitlines()] if trace.exists() else []
        check(mode+' status and phase count',r.returncode==expected and len(events)==count)
        if not events: print(r.stderr,r.stdout)
        if count==2 and len(events)==2: check(mode+' independent worker PIDs',events[0][1]!=events[1][1] and events[0][2]==events[1][2])
    e=env('cancel')
    proc=subprocess.Popen([sys.executable,str(fixture)],env=e)
    trace=pathlib.Path(e['TRACE'])
    try:
        deadline=time.monotonic()+8
        while not trace.exists() and proc.poll() is None and time.monotonic()<deadline: time.sleep(.05)
        pid=json.loads(trace.read_text().splitlines()[0])[1]
        from ctypes import wintypes as W
        k=ctypes.WinDLL('kernel32',use_last_error=True)
        k.OpenProcess.argtypes=[W.DWORD,W.BOOL,W.DWORD]; k.OpenProcess.restype=W.HANDLE
        k.WaitForSingleObject.argtypes=[W.HANDLE,W.DWORD]; k.WaitForSingleObject.restype=W.DWORD
        k.CloseHandle.argtypes=[W.HANDLE]
        handle=k.OpenProcess(0x00100000,False,pid)
        check('worker alive before cancellation',bool(handle) and k.WaitForSingleObject(handle,0)==258)
        proc.terminate(); proc.wait(timeout=5)
        check('direct GUI-style supervisor termination kills worker',bool(handle) and k.WaitForSingleObject(handle,5000)==0)
        if handle:k.CloseHandle(handle)
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
receipt={'scope':'AST equivalence, synthetic subprocess orchestration, Windows Job cancellation; no actual model/cache generation','results':results,'passed':sum(x['pass'] for x in results),'failed':sum(not x['pass'] for x in results),'candidate_sha256':hashlib.sha256((p/'1_pre_cache_qwen_image21.py').read_bytes()).hexdigest()}
(p/'TEST_RECEIPT_V2.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
sys.exit(bool(receipt['failed']))
