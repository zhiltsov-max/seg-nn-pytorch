from .models import *
from .models_deeplabv3 import *
from .common import *
from .checkpoints import *
import torch.nn as nn


_model_prefix = 'model_'
names = [
    n[len(_model_prefix):] for n in globals() \
    if n.startswith(_model_prefix)
]

def make_segmentation_model(model_name, class_count, **kwargs):
    model = globals()[_model_prefix + model_name](class_count, **kwargs)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    return model, criterion