import torch
import sys
import logging
import typing
from typing import List, Dict, Any, Optional
from torch._inductor.graph import GraphLowering
import torch.fx


class WelderGraphLowering(GraphLowering):
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        shape_env=None,
        num_static_inputs=None,
        graph_id=None,
    ):
        super().__init__(gm, shape_env, num_static_inputs, graph_id)