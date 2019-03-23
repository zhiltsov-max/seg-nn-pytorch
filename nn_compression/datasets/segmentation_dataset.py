from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import os.path as osp
import numpy as np
import hashlib
import json
from nn_compression.utils import StatsCounter


def to_long_tensor(img):
    return torch.as_tensor(np.array(img), dtype=torch.long)

def crop_pad(tensor, crop_size):
    assert len(tensor.size()) in [2, 3], "Expected image input (C, H, W) or (H, W)"
    assert len(crop_size) == 2, "Expected a pair of numbers (w, h)"

    crop_w, crop_h = crop_size
    img_h, img_w = tensor.size()[-2], tensor.size()[-1]
    img = tensor.view(-1, img_h, img_w)

    real_crop_w, real_crop_h = min(img_w, crop_w), min(img_h, crop_h)

    buffer = torch.zeros((img.size()[0], crop_h, crop_w), dtype=img.dtype)
    buffer[:, :real_crop_h, :real_crop_w] = \
       img[:, :real_crop_h, :real_crop_w]

    if len(tensor.size()) == 2:
        buffer = buffer.view(crop_h, crop_w)
    return buffer

class Subset(data.Dataset):
    IMAGE_CHANNELS_COUNT = 3
    TARGET_CHANNELS_COUNT = 1

    def __init__(self, base_dir, images_list,
            input_transform=transforms.ToTensor(),
            target_transform=transforms.Lambda(to_long_tensor),
            mean=None, std=None, crop_size=None, cache=True):
        super(Subset, self).__init__()
        self.base_dir = base_dir

        self.images_list = images_list
        self.image_sizes = np.zeros((len(images_list), 2), dtype=int)
        for i in range(len(self.images_list)):
            image = self.get(i)[0]
            self.image_sizes[i][:] = [image.height, image.width]

        self.input_transform = input_transform
        self.target_transform = target_transform

        self._mean = mean
        self._std = std
        if (self._mean is not None) and (self._std is not None):
            self.input_transform = transforms.Compose([
                self.input_transform,
                transforms.Normalize(self._mean, self._std)
            ])

        self.crop_size = crop_size
        if self.crop_size:
            self.input_transform = transforms.Compose([
                self.input_transform,
                lambda t: crop_pad(t, self.crop_size),
            ])
            self.target_transform = transforms.Compose([
                self.target_transform,
                lambda t: crop_pad(t, self.crop_size),
            ])

    def load(self, input_path, target_path):
        input = Image.open(osp.join(self.base_dir, input_path)).convert('RGB')
        target = Image.open(osp.join(self.base_dir, target_path)).convert('P')
        return input, target

    def get(self, i):
        entry = self.images_list[i]
        input, target = self.load(entry[0], entry[1])
        return input, target, entry

    def get_path(self, i):
        return self.images_list[i]

    # Returns [height, width] of i-th sample
    def get_size(self, i):
        return self.image_sizes[i]

    def __getitem__(self, i):
        input, target, _ = self.get(i)
        input = self.input_transform(input)
        target = self.target_transform(target)

        return input, target, i

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
        if (not self._mean) or (not self._std):

            stats = torch.FloatTensor(len(self), 2, self.IMAGE_CHANNELS_COUNT).cuda()
            counts = torch.FloatTensor(len(self)).cuda()

            mean = lambda i, s: s[i][0]
            var = lambda i, s: s[i][1]

            for i in range(len(self)):
                input, _, _ = self.get(i)
                input_array = transforms.functional.to_tensor(input).cuda() \
                    .view(self.IMAGE_CHANNELS_COUNT, -1)
                count = input_array.size()[1]
                mean(i, stats)[:] = input_array.mean(dim=1)
                var(i, stats)[:] = input_array.var(dim=1) * count / (count - 1.0)
                counts[i] = count

            counter = StatsCounter()
            _, mean, var = counter.compute_stats(stats, counts, mean, var)
            self._mean = [v.item() for v in mean.cpu()]
            self._std = [v.item() for v in torch.sqrt(var).cpu()]
        return self._mean, self._std

    def cache(self, enable=True):
        if enable:
            self._cache = []


class SegmentationDataset:
    list_dir = 'list'
    stats_file_name = 'stats.json'

    def __init__(self, data_root, subsets, normalize=True):
        self.data_root = data_root
        self.subset_names = subsets
        self.subsets_lists = self.get_lists()

        stats = {}
        if normalize:
            stats_updated = False
            stats_file_path = osp.join(self.data_root, self.stats_file_name)
            if osp.isfile(stats_file_path):
                stats = self._load_stats(stats_file_path)

        self.subset = {}
        for subset_name in self.subset_names:
            mean, std = self._load_subset_stats(stats, subset_name)

            subset_list = self.subsets_lists[subset_name]
            if normalize and (not mean or not std):
                mean, std = Subset(self.data_root, subset_list).compute_stats()
                self._save_subset_stats(stats, subset_name, [mean, std])
                stats_updated = True

            subset = Subset(self.data_root, subset_list,
                mean=mean, std=std, crop_size=self.crop_size)
            self.subset[subset_name] = subset

        if normalize and stats_updated:
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
        with open(file_path, 'r') as f:
            print("Loading dataset stats from '%s'" % (file_path))
            stats = json.load(f)
            return stats

    def _save_stats(self, file_path, stats):
        def get_or_default(dic, key, convert=None):
            entry = dic.get(key, None)
            if entry is not None:
                return convert(entry)
            return entry

        for subset_name in stats:
            subset_stats = stats[subset_name]
            checksum = self._get_subset_checksum(subset_name)
            subset_stats['list_checksum'] = checksum
            subset_stats['mean'] = get_or_default(subset_stats, 'mean', list)
            subset_stats['std'] = get_or_default(subset_stats, 'std', list)

        try:
            with open(file_path, 'w') as f:
                json.dump(stats, f, indent=0)
                print("Saved dataset stats at '%s'" % (file_path))
        except Exception as e:
            if osp.isfile(file_path):
                os.remove(file_path)
            raise e

    def get_lists(self):
        if hasattr(self, 'subsets_lists'):
            return self.subsets_lists

        self.subsets_lists = {}
        for subset_name in self.subset_names:
            subset_list = [
                line.strip().split(' ') for line \
                    in open(self._get_list_file_path(subset_name))
            ]
            self.subsets_lists[subset_name] = subset_list
        return self.subsets_lists

    def get_subset(self, subset_name):
        return self.subset[subset_name]
