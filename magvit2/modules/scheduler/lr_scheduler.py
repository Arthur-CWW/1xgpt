import math
import torch
from functools import partial


# step scheduler
def fn_LinearWarmup(warmup_steps: int, step: int):
    if step < warmup_steps:  # linear warmup
        return float(step) / float(max(1, warmup_steps))
    else:
        return 1.0


def Scheduler_LinearWarmup(warmup_steps: int):
    return partial(fn_LinearWarmup, warmup_steps)


def fn_LinearWarmup_CosineDecay(warmup_steps: int, max_steps: int, multipler_min: float, step: int):
    if step < warmup_steps:  # linear warmup
        return float(step) / float(max(1, warmup_steps))
    else:  # cosine learning rate schedule
        multipler = 0.5 * (
            math.cos((step - warmup_steps) / (max_steps - warmup_steps) * math.pi) + 1
        )
        return max(multipler, multipler_min)


def Scheduler_LinearWarmup_CosineDecay(warmup_steps: int, max_steps: int, multipler_min: float):
    return partial(fn_LinearWarmup_CosineDecay, warmup_steps, max_steps, multipler_min)
