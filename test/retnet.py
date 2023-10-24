import torch

'''
Original:
q[B, H, Q, Kd]
k[B, H, K, Kd]
v[B, H, K, D]
mask[H, Q, K]
------------
qk  = q @ k.transpose(-1, -2)
qkm = qk * m
r = qkm.detach().abs().sum(dim=-1).clamp(min=1)
s = qkm/r
o = s@v
-------------

Flash:
q[B, H, Br, Kd]
k[B, H, Bc, Kd]
v[B, H, Bc, D]
mask[H, Br, Bc]

r[B, H, Br]
acco[B, H, Br, D]
----------------------
r_new = 0
r_wo_clamp = 0
acco = 0
for (int j = 0; j < K/Bc; j ++)
{
  if (j != 0)
  { 
      acco = acco * r / r_new
      r = r_new
  }
  qkm = (q@k.transpose(-1,-2)) * m
  r_wo_clamp += qkm.detach().abs().sum(dim=-1)
  r_new = max(r_wo_clamp, 1)
  acco += (qkm/r_new)@v
}

-------------------------------
Best Config: {<Node, qkm>: {'block': [1, 1, 32, 64], 'warp': [1, 1, 16, 32], 'wmma': [16, 8, 16], 'use_cutlass': True, 'rstep': [128], 'use_tc': '80', 'strides': {3: <Stride, 2, 72>}}, <Node, reduce>: {'block': [1, 1, 32], 'thread': [1, 1, 32], 'rstep': [64], 'reduce_thread': [4], 'vectorize': {'input0': 8}}, <Node, acco>: {'block': [1, 1, 32, 512], 'warp': [1, 1, 16, 256], 'wmma': [8, 32, 16], 'use_cutlass': False, 'rstep': [64], 'use_tc': '80', 'strides': {3: <Stride, 2, 520>}}}
'''


