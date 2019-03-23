# Adapter for SegNet modified CamVid dataset and original one
# Dataset (CamVid12): https://github.com/alexgkendall/SegNet-Tutorial

from PIL import Image
import numpy as np
from .segmentation_dataset import Subset, SegmentationDataset


class CamVid32(SegmentationDataset):
    classes = [
        'Void',
        'Archway',
        'Bicyclist',
        'Bridge',
        'Building',
        'Car',
        'CartLuggagePram',
        'Child',
        'Column_Pole',
        'Fence',
        'LaneMkgsDriv',
        'LaneMkgsNonDriv',
        'Misc_Text',
        'MotorcycleScooter',
        'OtherMoving',
        'ParkingBlock',
        'Pedestrian',
        'Road',
        'RoadShoulder',
        'Sidewalk',
        'SignSymbol',
        'Sky',
        'SUVPickupTruck',
        'TrafficCone',
        'TrafficLight',
        'Train',
        'Tree',
        'Truck_Bus',
        'Tunnel',
        'VegetationMisc',
        'Animal',
        'Wall'
    ]
    class_count = len(classes)

    class_colormap_index_to_rgb_table = np.array([
        0, 0, 0,
        192, 0, 128,
        0, 128, 192,
        0, 128, 64,
        128, 0, 0,
        64, 0, 128,
        64, 0, 192,
        192, 128, 64,
        192, 192, 128,
        64, 64, 128,
        128, 0, 192,
        192, 0, 64,
        128, 128, 64,
        192, 0, 192,
        128, 64, 64,
        64, 192, 128,
        64, 64, 0,
        128, 64, 128,
        128, 128, 192,
        0, 0, 192,
        192, 128, 128,
        128, 128, 128,
        64, 128, 192,
        0, 0, 64,
        0, 64, 64,
        192, 64, 128,
        128, 128, 0,
        192, 128, 192,
        64, 0, 64,
        192, 192, 0,
        64, 128, 64,
        64, 192, 0,
    ], dtype=np.int)

    crop_size = (960, 720)

    @staticmethod
    def class_colormap_index_to_r(p):
        if (CamVid32.class_count <= p): return 255
        return CamVid32.class_colormap_index_to_rgb_table[p * 3 + 0]

    @staticmethod
    def class_colormap_index_to_g(p):
        if (CamVid32.class_count <= p): return 255
        return CamVid32.class_colormap_index_to_rgb_table[p * 3 + 1]

    @staticmethod
    def class_colormap_index_to_b(p):
        if (CamVid32.class_count <= p): return 255
        return CamVid32.class_colormap_index_to_rgb_table[p * 3 + 2]

    @staticmethod
    def paint_inference(image):
        painted_channels = [None] * 3
        painted_channels[0] = image.point(CamVid32.class_colormap_index_to_r)
        painted_channels[1] = image.point(CamVid32.class_colormap_index_to_g)
        painted_channels[2] = image.point(CamVid32.class_colormap_index_to_b)
        return Image.merge("RGB", painted_channels)

    def __init__(self, data_root, normalize=True):
        super(CamVid32, self).__init__(
            data_root, ['train', 'val', 'test'], normalize)

    def get_train(self):
        return self.get_subset('train')

    def get_val(self):
        return self.get_subset('val')

    def get_test(self):
        return self.get_subset('test')


class CamVid12(SegmentationDataset):
    classes = [
        'Sky',
        'Building',
        'Pole',
        'Road',
        'Pavement',
        'Tree',
        'SignSymbol',
        'Fence',
        'Car',
        'Pedestrian',
        'Bicyclist',
        'Unlabelled'
    ]
    class_count = len(classes)

    class_colormap_index_to_rgb_table = np.array([
        128, 128, 128,
        128, 0, 0,
        192, 192, 128,
        128, 64, 128,
        60, 40, 222,
        128, 128, 0,
        192, 128, 128,
        64, 64, 128,
        64, 0, 128,
        64, 64, 0,
        0, 128, 192,
        0, 0, 0,
    ], dtype=np.int)

    crop_size = (480, 360)

    @staticmethod
    def class_colormap_index_to_r(p):
        if (CamVid12.class_count <= p): return 255
        return CamVid12.class_colormap_index_to_rgb_table[p * 3 + 0]

    @staticmethod
    def class_colormap_index_to_g(p):
        if (CamVid12.class_count <= p): return 255
        return CamVid12.class_colormap_index_to_rgb_table[p * 3 + 1]

    @staticmethod
    def class_colormap_index_to_b(p):
        if (CamVid12.class_count <= p): return 255
        return CamVid12.class_colormap_index_to_rgb_table[p * 3 + 2]

    @staticmethod
    def paint_inference(image):
        painted_channels = [None] * 3
        painted_channels[0] = image.point(CamVid12.class_colormap_index_to_r)
        painted_channels[1] = image.point(CamVid12.class_colormap_index_to_g)
        painted_channels[2] = image.point(CamVid12.class_colormap_index_to_b)
        return Image.merge("RGB", painted_channels)

    def __init__(self, data_root, normalize=True):
        super(CamVid12, self).__init__(
            data_root, ['train', 'val', 'test'], normalize)

    def get_train(self):
        return self.get_subset('train')

    def get_val(self):
        return self.get_subset('val')

    def get_test(self):
        return self.get_subset('test')