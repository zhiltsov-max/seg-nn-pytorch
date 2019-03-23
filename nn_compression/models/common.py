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

    print_model_info(model, True)

    print("Memory allocated: current {:,} bytes, max {:,} bytes".format(
        torch.cuda.memory_allocated(),
        torch.cuda.max_memory_allocated()))

def print_model_info(model, show=False):
    model_parameters_count = 0
    model_states_size = 0
    model_buffers_size = 0
    model_modules_count = 0
    layer_parameter_counts = []
    layer_states_sizes = []
    layer_buffers_sizes = []
    for module_name, module in model.named_modules():
        if len(module._modules) != 0:
            continue

        module_parameters_count = 0
        for param in module.parameters():
            module_parameters_count += param.numel()
        layer_parameter_counts.append(module_parameters_count)
        model_parameters_count += module_parameters_count

        module_states_size = 0
        module_state = module.state_dict()
        for buffer_name in module_state:
            buffer = module_state[buffer_name]
            buffer_size = buffer.numel() * buffer.element_size()
            module_states_size += buffer_size
        layer_states_sizes.append(module_states_size)
        model_states_size += module_states_size

        module_buffers_size = 0
        for buffer_name in module._buffers:
            buffer = module._buffers[buffer_name]
            module_buffers_size += buffer.numel() * buffer.element_size()
        layer_buffers_sizes.append(module_buffers_size)
        model_buffers_size += module_buffers_size

        model_modules_count += 1

        print("Module: {}, {:,} parameters, "
              "state size {:,} bytes, buffers size {:,} bytes".format(
            module_name, module_parameters_count, 
            module_states_size, module_buffers_size))
    print("Layers count:", model_modules_count)
    print("Total parameters: {:,}".format(model_parameters_count))
    print("Model size: {:,} bytes, buffers size: {:,}".format(
        model_states_size, model_buffers_size))

    if show:
        import matplotlib.pyplot as plt
        import numpy as np

        plt.subplot(211)
        plt.bar(range(len(layer_parameter_counts)), layer_parameter_counts)

        plt.subplot(212)
        plt.bar(np.arange(len(layer_states_sizes)), 
            layer_states_sizes, color='g', width=0.8)
        plt.bar(np.arange(len(layer_buffers_sizes)) + 0.5, 
            layer_buffers_sizes, color='r', width=0.8)

        plt.savefig('weights_distribution.png')

def print_weights_and_grads(model):
    for module_name, module in model.named_modules():
        if len(module._modules) != 0:
            continue

        module_norm = 0
        module_grad_norm = 0
        for param in module.parameters():
            norm = param.norm().item()
            grad_norm = param.grad.norm().item()
            module_norm += norm * norm
            module_grad_norm += grad_norm * grad_norm
        norm = norm ** 0.5
        grad_norm = grad_norm ** 0.5

        print("%s norm: %s" % (module_name, norm))
        print("%s grad norm: %s" % (module_name, grad_norm))