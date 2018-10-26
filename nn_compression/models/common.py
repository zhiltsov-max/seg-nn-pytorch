import torch
import torch.cuda
import torch.nn as nn


def prod(seq):
    from functools import reduce
    from operator import mul
    return reduce(mul, seq)

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

def print_model_info(model, show=False):
    model_parameters_count = 0
    model_buffers_size = 0
    model_modules_count = 0
    layer_parameter_counts = []
    layer_buffer_sizes = []
    for module_name, module in model.named_modules():
        if len(module._modules) != 0:
            continue

        module_parameters_count = 0
        for param in module.parameters():
            module_parameters_count += prod(param.size())
        layer_parameter_counts.append(module_parameters_count)
        model_parameters_count += module_parameters_count

        module_buffers_size = 0
        module_state = module.state_dict()
        for buffer_name in module_state:
            buffer = module_state[buffer_name]
            buffer_size = buffer.numel() * buffer.element_size()
            module_buffers_size += buffer_size
        layer_buffer_sizes.append(module_buffers_size)
        model_buffers_size += module_buffers_size

        model_modules_count += 1

        print("Module: {}, {:,} parameters, size {:,} bytes".format(
            module_name, module_parameters_count, module_buffers_size))
    print("Layers count:", model_modules_count)
    print("Total parameters: {:,}".format(model_parameters_count))
    print("Model size: {:,} bytes".format(model_buffers_size))

    if show:
        import matplotlib.pyplot as plt

        plt.subplot(211)
        plt.bar(range(len(layer_parameter_counts)), layer_parameter_counts)

        plt.subplot(212)
        plt.bar(range(len(layer_buffer_sizes)), layer_buffer_sizes)

        plt.show()