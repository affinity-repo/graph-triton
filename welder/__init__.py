from loguru import logger
import torch

from torch._inductor.lowering import lowerings

# black_list = ["tuned_mm", "tuned_bmm", "tuned_addmm"]
# to_erase = []
# for k in lowerings.keys():
#    if lowerings[k].__name__ in black_list:
#        to_erase.append(k)
# for e in to_erase:
#    del lowerings[e]
# from .triton_kernels import mm, bmm

from welder import compile
from welder import graph

logger.debug("Init Welder for Pytorch Inductor...")


def read_triton_code(src="triton_code.py", var="triton__0"):
    


def graph_compile(nn: torch.nn.Module, *dummy_inputs):
    nn_compiled = torch.compile(nn, backend="welder")
    _output = nn_compiled(*dummy_inputs)


def graph_compile_fa(
    FA_Body: torch.nn.Module,
    q: torch.tensor,
    k: torch.tensor,
    v: torch.tensor,
    m: torch.tensor,
):
    pass


def profiler(func):
    def wrapper(*args, **kwargs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        warmup_times = 10
        execution_times = 100

        for i in range(warmup_times):
            result = func(*args, **kwargs)

        start_event.record()
        for i in range(execution_times):
            result = func(*args, **kwargs)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / execution_times
        print(f"{func.__name__} Executed: {elapsed_time:.3f} ms")

        return result

    return wrapper
