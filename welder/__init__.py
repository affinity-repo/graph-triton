from loguru import logger
import torch

from .triton_kernels import mm, bmm

from welder import compile
from welder import graph

logger.debug("Init Welder for Pytorch Inductor...")
