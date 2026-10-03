import torch, triton
from tensorfold.families.deepseek_v41.cuda import router as RT
dev="cuda"
g=torch.Generator(device=dev).manual_seed(0)
E=384
w=(torch.randn((E,5120),generator=g,device=dev)*0.02).to(torch.bfloat16)
b=(torch.randn(E,generator=g,device=dev)*0.1).float()
for R in (1, 128):
    x=torch.randn((R,5120),generator=g,device=dev).to(torch.bfloat16)
    p0,w0=RT.route(x,w,b,split=False)
    for BEB in (16,32):
        for warps in (2,4):
            for BK, ST in ((64,2),(64,3),(64,4),(128,2),(128,3),(32,4),(32,6)):
                lg=torch.empty((max(R,16),RT.BE),dtype=torch.float32,device=dev)
                def call():
                    RT._logits[(triton.cdiv(R,16), triton.cdiv(E,BEB))](x,x.stride(0),w,lg,R,D=5120,E=E,BE=RT.BE,BEB=BEB,BK=BK,num_warps=warps,num_stages=ST)
                try:
                    call(); torch.cuda.synchronize()
                except Exception as e:
                    print(R,BEB,warps,BK,ST,"fail",type(e).__name__); continue
                pick=torch.empty_like(p0); wt=torch.empty_like(w0)
                RT._select[(triton.cdiv(R,16),)](lg,b,pick,wt,R,E=E,K=6,SLOTS=7,SCALE=1.5,BE=RT.BE,num_warps=8)
                same=torch.equal(pick,p0) and torch.equal(wt.view(torch.int32),w0.view(torch.int32))
                for _ in range(3): call()
                torch.cuda.synchronize()
                a,bb=torch.cuda.Event(True),torch.cuda.Event(True)
                a.record()
                for _ in range(50): call()
                bb.record(); torch.cuda.synchronize()
                print(f"R={R} BEB={BEB} warps={warps} BK={BK} ST={ST}: {a.elapsed_time(bb)/50*1e3:.1f} us bitwise={same}",flush=True)
