import sys
sys.path.insert(0, "/dsv41-tf/tests")
import numpy as np, torch
import dsv41_kref as KR
from tensorfold.families.deepseek_v41.cuda import engram
g = np.random.default_rng(1)
v = g.integers(0, 256, size=(5, 12, 256), dtype=np.uint8)
v[np.isin(v, (0x7F, 0xFF))] = 0x38
v[0, 0] = np.array([b for b in range(256) if b not in (0x7F, 0xFF)] + [0, 0], dtype=np.uint8)
s = g.integers(118, 136, size=(5, 12, 8), dtype=np.uint8)
s[0, 1] = [0, 1, 2, 127, 200, 250, 254, 3]
s[0, 2] = 1
raw = torch.from_numpy(np.concatenate([v, s], axis=2)).contiguous()
out = torch.zeros((5, 24 * 256), dtype=torch.bfloat16, device="cuda")
engram.dequant(raw.cuda(), out, col0=12)
got = out[:, 12 * 256:].cpu()
want = KR.engram_dequant(raw)
gb, wb = got.view(torch.int16), want.view(torch.int16)
bad = (gb != wb).nonzero().tolist()
print("mismatches", len(bad), "of", gb.numel())
for r, c in bad[:30]:
    h, i = divmod(c, 256)
    byte, sb = int(v[r, h, i]), int(s[r, h, i // 32])
    print(f"r{r} h{h} i{i} byte {byte:#04x} scale {sb}: gpu {float(got[r,c])!r} ({int(gb[r,c]) & 0xffff:#06x}) "
          f"ref {float(want[r,c])!r} ({int(wb[r,c]) & 0xffff:#06x})")
# CPU float32 path in pieces
f = torch.tensor(v.reshape(-1)).view(torch.float8_e4m3fn).float()
print("torch flush denormal:", torch.get_num_threads())
x = torch.tensor([1.5], dtype=torch.float32) * torch.tensor([2.0 ** -126])
print("cpu 1.5*2^-126 =", x.item(), "gpu", (torch.tensor([1.5], device="cuda") * torch.tensor([2.0 ** -126], device="cuda")).item())
