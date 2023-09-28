from loguru import logger
import torch

from torch._inductor.lowering import lowerings

black_list = ["tuned_mm", "tuned_bmm", "tuned_addmm"]
to_erase = []
for k in lowerings.keys():
    if lowerings[k].__name__ in black_list:
        to_erase.append(k)
for e in to_erase:
    del lowerings[e]
from .triton_kernels import mm, bmm

from welder import compile
from welder import graph

logger.debug("Init Welder for Pytorch Inductor...")
