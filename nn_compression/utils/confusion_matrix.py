import numpy as np


class ConfusionMatrix:
    """
        Code adapted from VOC 2012 mIoU calculation script.
    """
    check_args = True
    classes = None
    class_count = 0
    confusion = None
    dirty = False
    sumim = None

    def __init__(self, classes, check_args=True):
        self.classes = classes[:]
        self.class_count = len(classes)
        self.check_args = check_args
        self.confusion = np.zeros((self.class_count, self.class_count))

    def check_segmentation(self, image_array):
        has_invalid_values = (np.any(image_array < 0)) or \
                             (np.any(image_array >= self.class_count))
        if has_invalid_values:
            raise Exception(
                'Array contains unexpected values beyond interval [0; %d]: %s.' \
                % (self.class_count - 1, np.unique(image_array)))

    def update(self, output, target):
        """
            Inputs:
                outputs: float tensor of shape (C, H, W) or (C, H * W)
                      or long tensor of shape (H * W)
                targets: long tensor of shape (1, H, W) or (H, W) or (H * W)
        """
        confusion = self.confusion

        assert(
            (output.dim() == 1) or \
            (output.dim() == 2) and (output.size()[0] == self.class_count) or \
            (output.dim() == 3) and (output.size()[0] == self.class_count))
        assert(
            (target.dim() == 2) or \
            (target.dim() == 3 and target.size()[0] == 1))

        if output.dim() != 1:
            output = output.view(self.class_count, -1).argmax(dim=0)
        output = output.cpu().numpy()
        target = target.view(-1).cpu().numpy()

        if output.size != target.size:
            raise Exception("Array sizes are different: %s, %s" % \
                (output.size, target.size))

        if self.check_args:
            self.check_segmentation(output)
            self.check_segmentation(target)

        if self.sumim is None:
            self.sumim = np.array(output, dtype=int)
        sumim = self.sumim

        sumim[:] = output
        sumim *= self.class_count
        sumim += target
        hist, _ = np.histogram(sumim[target < 255],
            bins=range(0, confusion.size + 1))
        confusion += hist.reshape(confusion.shape)

        self.dirty = True

    def compute_overall_accuracy(self):
        self.overall_acc = 100.0 * \
            np.sum(np.diagonal(self.confusion)) / np.sum(self.confusion)
        return self.overall_acc

    def compute_class_accuracies(self):
        self.class_acc = np.zeros(self.class_count)
        denoms = np.sum(self.confusion, axis=0)
        for i in range(self.class_count):
            if (denoms[i] == 0):
                denoms[i] = 1
            self.class_acc[i] = 100.0 * self.confusion[i][i] / denoms[i]
        self.avg_class_acc = np.sum(self.class_acc) / self.class_count
        return self.class_acc, self.avg_class_acc

    def compute_class_IoU(self):
        self.class_iou = np.zeros(self.class_count)
        gtj = np.sum(self.confusion, axis=1)
        resj = np.sum(self.confusion, axis=0)
        for j in range(self.class_count):
            gtjresj = self.confusion[j][j]
            denom = max(gtj[j] + resj[j] - gtjresj, 1.0)
            self.class_iou[j] = 100.0 * gtjresj / denom
        self.avg_class_iou = np.sum(self.class_iou) / self.class_count
        return self.class_iou, self.avg_class_iou

    def update_results(self):
        if self.dirty == True:
            self.compute_overall_accuracy()
            self.compute_class_accuracies()
            self.compute_class_IoU()
            self.dirty = False

    def get_results(self):
        self.update_results()
        return self

    def show_results(self):
        self.update_results()

        print('----------------------------------')
        print('Percentage of pixels correctly labelled overall (mean accuracy): %6.3f%%' %
            self.overall_acc)
        print('----------------------------------')
        print('Mean class accuracy: %6.3f%%' % self.avg_class_acc)
        print('Class accuracies:')
        for i in range(self.class_count):
            print('  %18s: %6.3f%%' % (self.classes[i], self.class_acc[i]))
        print('----------------------------------')
        print('Mean IoU accuracy: %6.3f%%' % self.avg_class_iou)
        print('IoU class accuracies:')
        for i in range(self.class_count):
            print('  %18s: %6.3f%%' % (self.classes[i], self.class_iou[i]))
        print('----------------------------------')