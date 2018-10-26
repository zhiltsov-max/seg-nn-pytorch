import models.models as models
from .common import *
from .checkpoints import *


_model_prefix = 'model_'
names = [
    n[len(_model_prefix):] for n in models.__dir__() \
    if n.startswith(_model_prefix)
]

def make_segmentation_model(model_name, class_count, **kwargs):
    model = getattr(models, _model_prefix + model_name)(class_count, **kwargs)
    criterion = nn.CrossEntropyLoss()
    return model, criterion
