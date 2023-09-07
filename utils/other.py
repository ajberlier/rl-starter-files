import random
import numpy
import torch
import collections
import subprocess
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthesize(array):
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(array)
    d["std"] = numpy.std(array)
    d["min"] = numpy.amin(array)
    d["max"] = numpy.amax(array)
    return d


def get_gpu_memory():
    if device != "cuda":
        return [0], [0]
    else:
        command = "nvidia-smi --query-gpu=memory.used --format=csv"
        memory_used_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        command = "nvidia-smi --query-gpu=memory.total --format=csv"
        memory_total_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_used_info = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]
        memory_total_info = [int(x.split()[0]) for i, x in enumerate(memory_total_info)]
        memory_used_percent = np.asarray(memory_used_info) / np.asarray(memory_total_info)
        return memory_used_percent, memory_total_info