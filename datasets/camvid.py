from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import os.path as osp
import numpy as np


def toLongTensor(img):
    return torch.as_tensor(np.array(img).transpose(2, 0, 1),
        dtype=torch.long)[0]

class Subset(data.Dataset):
    images_list = None
    base_dir = None

    def __init__(self, base_dir, images_list,
            input_transform=transforms.ToTensor(),
            target_transform=transforms.Lambda(toLongTensor),
            mean=None, std=None):
        super(Subset, self).__init__()
        self.base_dir = base_dir
        self.images_list = images_list
        self.input_transform = input_transform
        self.target_transform = target_transform
        self._mean = mean
        self._std = std

    def __getitem__(self, i):
        entry = self.images_list[i]
        input = Image.open(osp.join(self.base_dir, entry[0])).convert('RGB')
        target = Image.open(osp.join(self.base_dir, entry[1])).convert('RGB')
        input = self.input_transform(input)
        target = self.target_transform(target)
        return input, target

    def __len__(self):
        return len(self.images_list)

    @property
    def mean(self):
        if not self._mean:
            self.compute_stats()
        return self._mean

    @property
    def std(self):
        if not self._std:
            self.compute_stats()
        return self._std

    def compute_stats(self):
        if not self._mean or not self._std:
            mean = np.zeros(3, dtype=np.float)
            std = np.zeros(3, dtype=np.float)
            for i in range(len(self)):
                input, _ = self[i]
                input_array = np.array(input).transpose((2, 0, 1))
                mean += np.mean(input_array, axis=(1, 2))
                std += np.var(input_array, axis=(1, 2))
            self._mean = mean * (1.0 / len(self))
            self._std = np.sqrt(std) * (1.0 / (len(self) - 1))
        return self._mean, self._std


class CamVid32:
    """
    Class color mapping:
    00. 0 0 0 Void
    01. 192 0 128 Archway
    02. 0 128 192 Bicyclist
    03. 0 128 64 Bridge
    04. 128 0 0  Building
    05. 64 0 128 Car
    06. 64 0 192 CartLuggagePram
    07. 192 128 64 Child
    08. 192 192 128 Column_Pole
    09. 64 64 128 Fence
    10. 128 0 192 LaneMkgsDriv
    11. 192 0 64 LaneMkgsNonDriv
    12. 128 128 64 Misc_Text
    13. 192 0 192 MotorcycleScooter
    14. 128 64 64 OtherMoving
    15. 64 192 128 ParkingBlock
    16. 64 64 0 Pedestrian
    17. 128 64 128 Road
    18. 128 128 192 RoadShoulder
    19. 0 0 192 Sidewalk
    20. 192 128 128 SignSymbol
    21. 128 128 128 Sky
    22. 64 128 192 SUVPickupTruck
    23. 0 0 64 TrafficCone
    24. 0 64 64 TrafficLight
    25. 192 64 128 Train
    26. 128 128 0 Tree
    27. 192 128 192 Truck_Bus
    28. 64 0 64 Tunnel
    29. 192 192 0 VegetationMisc
    30. 64 128 64 Animal
    31. 64 192 0 Wall
    """

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

    @staticmethod
    def class_colormap_index_to_r(p):
        if (32 <= p): return 255
        return CamVid.class_colormap_index_to_rgb_table[p * 3 + 0]

    @staticmethod
    def class_colormap_index_to_g(p):
        if (32 <= p): return 255
        return CamVid.class_colormap_index_to_rgb_table[p * 3 + 1]

    @staticmethod
    def class_colormap_index_to_b(p):
        if (32 <= p): return 255
        return CamVid.class_colormap_index_to_rgb_table[p * 3 + 2]

    class_count = 32
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

    @staticmethod
    def paint_inference(image):
        painted_channels = [None] * 3
        painted_channels[0] = image.point(CamVid32.class_colormap_index_to_r)
        painted_channels[1] = image.point(CamVid32.class_colormap_index_to_g)
        painted_channels[2] = image.point(CamVid32.class_colormap_index_to_b)
        return Image.merge("RGB", painted_channels)

    def __init__(self, data_root):
        self.data_root = data_root
        self.train_list, self.val_list, self.test_list = self.get_lists()
        self.train_set = Subset(self.data_root, self.train_list)
        self.val_set = Subset(self.data_root, self.val_list)
        self.test_set = Subset(self.data_root, self.test_list)

    def get_lists(self):
        train_set_list = [
            line.split(' ') for line \
            in open(osp.join(self.data_root, 'list', 'train.txt'))
        ]
        val_set_list = [
            line.split(' ') for line \
            in open(osp.join(self.data_root, 'list', 'val.txt'))
        ]
        test_set_list = [
            line.split(' ') for line \
            in open(osp.join(self.data_root, 'list', 'test.txt'))
        ]
        return train_set_list, val_set_list, test_set_list

    def get_train(self):
        return self.train_set

    def get_val(self):
        return self.val_set

    def get_test(self):
        return self.test_set


class CamVid12:
    """
    Class color mapping:
    00. 0 0 0 Void
    01. 192 0 128 Archway
    02. 0 128 192 Bicyclist
    03. 0 128 64 Bridge
    04. 128 0 0  Building
    05. 64 0 128 Car
    06. 64 0 192 CartLuggagePram
    07. 192 128 64 Child
    08. 192 192 128 Column_Pole
    09. 64 64 128 Fence
    10. 128 0 192 LaneMkgsDriv
    11. 192 0 64 LaneMkgsNonDriv
    12. 128 128 64 Misc_Text
    13. 192 0 192 MotorcycleScooter
    14. 128 64 64 OtherMoving
    15. 64 192 128 ParkingBlock
    16. 64 64 0 Pedestrian
    17. 128 64 128 Road
    18. 128 128 192 RoadShoulder
    19. 0 0 192 Sidewalk
    20. 192 128 128 SignSymbol
    21. 128 128 128 Sky
    22. 64 128 192 SUVPickupTruck
    23. 0 0 64 TrafficCone
    24. 0 64 64 TrafficLight
    25. 192 64 128 Train
    26. 128 128 0 Tree
    27. 192 128 192 Truck_Bus
    28. 64 0 64 Tunnel
    29. 192 192 0 VegetationMisc
    30. 64 128 64 Animal
    31. 64 192 0 Wall
    """

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

    @staticmethod
    def class_colormap_index_to_r(p):
        if (32 <= p): return 255
        return CamVid.class_colormap_index_to_rgb_table[p * 3 + 0]

    @staticmethod
    def class_colormap_index_to_g(p):
        if (32 <= p): return 255
        return CamVid.class_colormap_index_to_rgb_table[p * 3 + 1]

    @staticmethod
    def class_colormap_index_to_b(p):
        if (32 <= p): return 255
        return CamVid.class_colormap_index_to_rgb_table[p * 3 + 2]

    class_count = 32
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

    @staticmethod
    def paint_inference(image):
        painted_channels = [None] * 3
        painted_channels[0] = image.point(CamVid32.class_colormap_index_to_r)
        painted_channels[1] = image.point(CamVid32.class_colormap_index_to_g)
        painted_channels[2] = image.point(CamVid32.class_colormap_index_to_b)
        return Image.merge("RGB", painted_channels)

    def __init__(self, data_root):
        self.data_root = data_root
        self.train_list, self.val_list, self.test_list = self.get_lists()
        self.train_set = Subset(self.data_root, self.train_list)
        self.val_set = Subset(self.data_root, self.val_list)
        self.test_set = Subset(self.data_root, self.test_list)

    def get_lists(self):
        train_set_list = [
            line.split(' ') for line \
            in open(osp.join(self.data_root, 'list', 'train.txt'))
        ]
        val_set_list = [
            line.split(' ') for line \
            in open(osp.join(self.data_root, 'list', 'val.txt'))
        ]
        test_set_list = [
            line.split(' ') for line \
            in open(osp.join(self.data_root, 'list', 'test.txt'))
        ]
        return train_set_list, val_set_list, test_set_list

    def get_train(self):
        return self.train_set

    def get_val(self):
        return self.val_set

    def get_test(self):
        return self.test_set