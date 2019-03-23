# Adapter for PASCAL VOC2012
# Home page: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html
# Direct link: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
#
# SBD extension:
# Home page: http://home.bharathh.info/pubs/codes/SBD/download.html
# Direct link: http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz

from PIL import Image
import numpy as np
from .segmentation_dataset import Subset, SegmentationDataset


class VOC12(SegmentationDataset):
    classes = [
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor'
    ]
    class_count = len(classes)

    class_colormap_index_to_rgb_table = np.array([
        0, 0, 0,
        128, 0, 0,
        0, 128, 0,
        128, 128, 0,
        0, 0, 128,
        128, 0, 128,
        0, 128, 128,
        128, 128, 128,
        64, 0, 0,
        192, 0, 0,
        64, 128, 0,
        192, 128, 0,
        64, 0, 128,
        192, 0, 128,
        64, 128, 128,
        192, 128, 128,
        0, 64, 0,
        128, 64, 0,
        0, 192, 0,
        128, 192, 0,
        0, 64, 128,
    ], dtype=np.int)

    crop_size = (512, 512)

    @staticmethod
    def class_colormap_index_to_r(p):
        if (VOC12.class_count <= p): return 255
        return VOC12.class_colormap_index_to_rgb_table[p * 3 + 0]

    @staticmethod
    def class_colormap_index_to_g(p):
        if (VOC12.class_count <= p): return 255
        return VOC12.class_colormap_index_to_rgb_table[p * 3 + 1]

    @staticmethod
    def class_colormap_index_to_b(p):
        if (VOC12.class_count <= p): return 255
        return VOC12.class_colormap_index_to_rgb_table[p * 3 + 2]

    @staticmethod
    def paint_inference(image):
        painted_channels = [None] * 3
        painted_channels[0] = image.point(VOC12.class_colormap_index_to_r)
        painted_channels[1] = image.point(VOC12.class_colormap_index_to_g)
        painted_channels[2] = image.point(VOC12.class_colormap_index_to_b)
        return Image.merge("RGB", painted_channels)

    def __init__(self, data_root, normalize=True):
        super(VOC12, self).__init__(
            data_root, ['train', 'val', 'test'], normalize)

    def get_train(self):
        return self.get_subset('train')

    def get_val(self):
        return self.get_subset('val')

    def get_test(self):
        return self.get_subset('test')