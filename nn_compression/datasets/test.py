import torch
import torch.utils.data as data


class Subset(data.Dataset):
    def __init__(self, input_shape, output_shape, count):
        super(Subset, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.count = count
        self.input = torch.Tensor(*self.input_shape)
        self.output = torch.Tensor(*self.output_shape)

    def __getitem__(self, i):
        return self.input, self.output

    def __len__(self):
        return self.count


class FakeDataset:
    def __init__(self, input_shape, output_shape, count, class_count):
        self.classes = ['class_%s' % (n) for n in class_count ]
        self.sample_set = Subset(input_shape, output_shape, count)

    def get_train(self):
        return self.sample_set

    def get_val(self):
        return self.sample_set

    def get_test(self):
        return self.sample_set