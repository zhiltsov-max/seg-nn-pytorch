from .camvid import CamVid12, CamVid32
from .voc12 import VOC12

names = [ n for n in globals() if n and n[0].isupper() ]