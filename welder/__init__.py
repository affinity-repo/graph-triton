from loguru import logger
import torch
import shutil

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
    with open(src, "r", encoding="utf-8") as file:
        content = file.read()
        start_index = content.find("@triton.jit")
        end_index = content.find("'''", start_index)
        parts = [v.strip() for v in content[start_index:end_index].split("\n")]
        for part in parts[:10]:
            logger.info(part)
        logger.info("...")
        parts = extract_triton(parts)
        logger.info("FA Loop Body")
        for part in parts:
            logger.info(part)
        logger.info("")
        return parts


def write_triton_code(parts: list, cfg, src="fa_triton_code.py"):
    import os
    import hashlib
    import subprocess

    logger.info(f"Init from foler {os.path.dirname(__file__)}")

    configs = generate_configs()
    configs = generate_welder_configs(cfg, configs)
    parts[0] = parts[0].strip()

    for config in configs:
        logger.warning(
            f"Trying config: {hashlib.sha256(config['welder_cfg_str'].encode())}"
        )
        file_name = "/".join(
            [str(os.path.dirname(__file__)), "kernel", "fa_triton_code.py"]
        )
        i_data = open(file_name)
        data = "".join(i_data.readlines())
        data = data.replace("welder_fa_body", "\n".join(parts))
        for v in config:
            data = data.replace(v, config[v])
        o_data = open(src, "w")
        o_data.write(data)
        o_data.close()
        i_data.close()
        logger.info(f"Generated: {src}")

        result = subprocess.run(
            ["python", src], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if result.returncode == 0:
            os.system(f"cp {src} final_fa_triton_code.py")


def graph_compile(nn: torch.nn.Module, *dummy_inputs):
    nn_compiled = torch.compile(nn, backend="welder")
    _output = nn_compiled(*dummy_inputs)


def graph_compile_fa(
    FA_Body: torch.nn.Module,
    ConfigFile: str,
    *dummy_inputs,
):
    graph_compile(FA_Body, *dummy_inputs)
    parts = read_triton_code()
    write_triton_code(parts, ConfigFile)


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


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(model, example_inputs, times=1):
    import time

    synchronize()
    torch.manual_seed(1337)
    t0 = time.perf_counter()
    for _ in range(times):
        result = model(*example_inputs)
    synchronize()
    t1 = time.perf_counter()
    # GC the result after timing
    assert result is not None
    return t1 - t0


def print_performance(fn, args=(), times=10, repeat=10, baseline=1.0):
    timings = torch.tensor([timed(fn, args, times) for _ in range(repeat)])
    took = torch.median(timings) * 1000
    print(f"Triton Execution Time: {took/baseline:.6f} ms")
    return took


def extract_triton(parts: list):
    c = """         tmp2 = tmp0 * tmp1
        tmp3 = tl.abs(tmp2)
        tmp3_r += tl.sum(tmp3, axis=1)
        tmp5 = tl.where(tmp3_r > 1, tmp3_r , 1)
        tmp9 = tmp9 * tmp4 / tmp5
        tmp6 = tmp0 / tmp5
        tmp7 = tmp6.to(tl.float16)
        tmp9_r += tl.dot(tmp7, tmp8)
        tmp10_r = tmp5"""
    return c.split("\n")


def generate_configs():
    return [
        {
            "welder_cfg_str": "(4, 8, 2048, 128, 2048, 256, 2, 32,)",
            "tmp3_r": "r_wo_clamp",
            "tmp9_r": "acco",
            "tmp10_r": "r",
        }
    ]


def generate_welder_configs(cfg_path, configs):
    wldcfg = open(cfg_path, "r")
    content = wldcfg.read()
    start_index = content.find("[")
    end_index = content.find("]", start_index)
    blockcfg = content[start_index + 1 : end_index]
    cfg = [int(v) for v in blockcfg.split(",")]
    logger.info(f"Reading welder config: {cfg_path}, {blockcfg}")

    import copy

    str_cfg = ", ".join([str(v) for v in cfg[2:]])

    cfg_copy = copy.copy(configs[0])
    cfg_copy["welder_cfg_str"] = cfg_copy["welder_cfg_str"].replace("2, 32", str_cfg)

    configs.append(cfg_copy)
    return configs


block_m = 8
block_n = 8
