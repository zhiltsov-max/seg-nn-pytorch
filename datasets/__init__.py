from .camvid import CamVid12, CamVid32
from .test import FakeDataset

names = [ n for n in globals() if n and n[0].isupper() ]