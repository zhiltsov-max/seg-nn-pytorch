from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import os.path as osp
import numpy as np
import hashlib
import json


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
        self._mean = mean
        self._std = std
        if (self._mean is not None) and (self._std is not None):
            self.input_transform = transforms.Compose([
                input_transform,
                transforms.Normalize(self._mean, self._std)
            ])
        else:
            self.input_transform = input_transform
        self.target_transform = target_transform

    def get(self, i):
        entry = self.images_list[i]
        input = Image.open(osp.join(self.base_dir, entry[0])).convert('RGB')
        target = Image.open(osp.join(self.base_dir, entry[1])).convert('RGB')
        return input, target, entry

    def __getitem__(self, i):
        input, target, _ = self.get(i)
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
                input, _, _ = self.get(i)
                input_array = np.array(input, dtype=float).transpose((2, 0, 1))
                input_array *= 1.0 / 255.0
                mean += np.mean(input_array, axis=(1, 2))
                std += np.var(input_array, axis=(1, 2))
            self._mean = mean * (1.0 / len(self))
            self._std = np.sqrt(std) * (1.0 / ((len(self) - 1) or 1))
        return self._mean, self._std


class Dataset:
    list_dir = 'list'
    stats_file_name = 'stats.json'

    def __init__(self, data_root, subsets, normalize=True):
        self.data_root = data_root
        self.subset_names = subsets
        self.subsets_lists = self.get_lists()

        stats_updated = False
        if normalize:
            stats_file_path = osp.join(self.data_root, self.stats_file_name)
            stats = {}
            if osp.isfile(stats_file_path):
                stats = self._load_stats(stats_file_path)

        self.subset = {}
        for subset_name in self.subset_names:
            mean, std = self._load_subset_stats(stats, subset_name)

            subset_list = self.subsets_lists[subset_name]
            if mean is None and std is None:
                mean, std = Subset(self.data_root, subset_list).compute_stats()
                self._save_subset_stats(stats, subset_name, [mean, std])
                stats_updated = True

            subset = Subset(self.data_root, subset_list, mean=mean, std=std)
            self.subset[subset_name] = subset

        if stats_updated:
            self._save_stats(stats_file_path, stats)

    def _get_list_file_path(self, subset_name):
        return osp.join(self.data_root, self.list_dir,
            '%s.txt' % (subset_name))

    def _get_subset_checksum(self, subset_name):
        list_path = self._get_list_file_path(subset_name)
        if not osp.isfile(list_path):
            raise Exception("Not found subset list file '%s'" % (list_path))

        with open(list_path, 'r') as f:
            file_data = f.read().encode('utf-8')
            return hashlib.md5(file_data).hexdigest()

    def _load_subset_stats(self, stats, subset_name):
        mean = None
        std = None
        if subset_name in stats:
            subset = stats[subset_name]

            their_checksum = subset['list_checksum']
            our_checksum = self._get_subset_checksum(subset_name)
            if their_checksum == our_checksum:
                mean = subset.get('mean', None)
                std = subset.get('std', None)
        return [mean, std]

    def _save_subset_stats(self, stats, subset_name, subset_stats):
        stats[subset_name] = {
            'mean': subset_stats[0],
            'std': subset_stats[1],
            'list_checksum': self._get_subset_checksum(subset_name),
        }

    def _load_stats(self, file_path):
        def read_array(d, key, default):
            entry = d.get(key, None)
            if entry is not None:
                entry = np.array(entry)
            return entry

        with open(file_path, 'r') as f:
            print("Loading dataset stats from '%s'" % (file_path))
            stats = json.load(f)
            for subset_name in stats:
                subset_stats = stats[subset_name]
                mean = read_array(subset_stats, 'mean', None)
                std = read_array(subset_stats, 'std', None)
            return stats

    def _save_stats(self, file_path, stats):
        for subset_name in stats:
            subset_stats = stats[subset_name]
            checksum = self._get_subset_checksum(subset_name)
            subset_stats['list_checksum'] = checksum
            if 'mean' in subset_stats:
                subset_stats['mean'] = list(subset_stats['mean'])
            else:
                subset_stats.pop('mean')
            if 'std' in subset_stats:
                subset_stats['std'] = list(subset_stats['std'])
            else:
                subset_stats.pop('std')
        try:
            with open(file_path, 'w') as f:
                json.dump(stats, f, indent=0)
                print("Saved dataset stats at '%s'" % (file_path))
        except Exception as e:
            if osp.isfile(file_path):
                os.remove(file_path)

    def get_lists(self):
        if hasattr(self, 'subsets_lists'):
            return self.subsets_lists

        self.subsets_lists = {}
        for subset_name in self.subset_names:
            subset_list = [
                line.split(' ') for line \
                    in open(self._get_list_file_path(subset_name))
            ]
            self.subsets_lists[subset_name] = subset_list
        return self.subsets_lists

    def get_subset(self, subset_name):
        return self.subset[subset_name]


class CamVid32(Dataset):
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
        return CamVid32.class_colormap_index_to_rgb_table[p * 3 + 0]

    @staticmethod
    def class_colormap_index_to_g(p):
        if (32 <= p): return 255
        return CamVid32.class_colormap_index_to_rgb_table[p * 3 + 1]

    @staticmethod
    def class_colormap_index_to_b(p):
        if (32 <= p): return 255
        return CamVid32.class_colormap_index_to_rgb_table[p * 3 + 2]

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


class CamVid12(Dataset):
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

    @staticmethod
    def class_colormap_index_to_r(p):
        if (32 <= p): return 255
        return CamVid12.class_colormap_index_to_rgb_table[p * 3 + 0]

    @staticmethod
    def class_colormap_index_to_g(p):
        if (32 <= p): return 255
        return CamVid12.class_colormap_index_to_rgb_table[p * 3 + 1]

    @staticmethod
    def class_colormap_index_to_b(p):
        if (32 <= p): return 255
        return CamVid12.class_colormap_index_to_rgb_table[p * 3 + 2]

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

    @staticmethod
    def paint_inference(image):
        painted_channels = [None] * 3
        painted_channels[0] = image.point(CamVid32.class_colormap_index_to_r)
        painted_channels[1] = image.point(CamVid32.class_colormap_index_to_g)
        painted_channels[2] = image.point(CamVid32.class_colormap_index_to_b)
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