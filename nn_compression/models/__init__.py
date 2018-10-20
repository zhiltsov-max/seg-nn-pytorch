import models
from .models import names

import torch
import torch.cuda
import torch.nn as nn


def make_segmentation_model(model_name, class_count, **kwargs):
    model = getattr(models, model_name)(class_count, **kwargs)
    criterion = nn.CrossEntropyLoss()
    return model, criterion

def print_memory_stats(model, sample_input=None, mode='train'):
    if mode == 'train':
        model.train()
    else:
        model.eval()

    if sample_input is not None:
        model.forward(sample_input)

    print_model_info(model)

    print("Memory allocated: current {:,} bytes, max {:,} bytes".format(
        torch.cuda.memory_allocated(),
        torch.cuda.max_memory_allocated()))

def print_model_info(model):
    from functools import reduce
    from operator import mul
    def prod(seq):
        return reduce(mul, seq)

    model_parameters_count = 0
    model_buffers_size = 0
    model_modules_count = 0
    for module_name, module in model.named_modules():
        if len(module._modules) != 0:
            continue

        module_parameters_count = 0
        for param in module.parameters():
            module_parameters_count += prod(param.size())
        model_parameters_count += module_parameters_count

        module_buffers_size = 0
        module_state = module.state_dict()
        for buffer_name in module_state:
            buffer = module_state[buffer_name]
            buffer_size = buffer.numel() * buffer.element_size()
            module_buffers_size += buffer_size
        model_buffers_size += module_buffers_size

        model_modules_count += 1

        print("Module: {}, {:,} parameters, size {:,} bytes".format(
            module_name, module_parameters_count, module_buffers_size))
    print("Layers count:", model_modules_count)
    print("Total parameters: {:,}".format(model_parameters_count))
    print("Model size: {:,} bytes".format(model_buffers_size))