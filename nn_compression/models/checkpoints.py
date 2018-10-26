import torch
import os
import os.path as osp


def load_checkpoint(path):
    checkpoint = torch.load(path)
    model_path = osp.join(osp.dirname(path), checkpoint['model_state_path'])
    if osp.isfile(model_path):
        checkpoint['model_state'] = torch.load(model_path)
    else:
        raise Exception("Failed to load model state from '%s'" % (model_path))
    checkpoint.pop('model_state_path', None)
    return checkpoint

def save_checkpoint(state, path):
    checkpoint = { k: state[k] for k in state if k not in ['model_state'] }
    model_state = state['model_state']
    model_path = 'model_%d.pth' % (checkpoint['epoch'])
    checkpoint['model_state_path'] = model_path

    torch.save(model_state, osp.join(osp.dirname(path), model_path))
    torch.save(checkpoint, path)