import torch
import torch.optim
import torch.backends.cudnn as cudnn
import torch.utils.data

import nn_compression.datasets as datasets
import nn_compression.models as models
from nn_compression.utils import ConfusionMatrix

import argparse
import os
import os.path as osp
import time
import PIL.Image as Image
import numpy as np
import random
import warnings


class ModelScore:
    mean_iou = 0
    mean_accuracy = 0
    class_ious = []
    class_accuracies = []

    def __lt__(self, other):
        return self.mean_iou < other.mean_iou

def evaluate(args, dataset, subset, model, save_dir=None):
    model.eval()
    with torch.no_grad():
        confusion = ConfusionMatrix(dataset.classes)

        print("Running evaluation...")

        eval_time = time.time()

        for i, (inputs, targets) in enumerate(subset):
            inputs = inputs.to(args.device, non_blocking=True)
            targets = targets.to(args.device, non_blocking=True)

            outputs = model.forward(inputs)
            outputs = outputs[:, :, 0:targets.size()[-2], 0:targets.size()[-1]]
            for output, target in zip(outputs, targets):
                confusion.update(output, target)

            if args.save_inference:
                # TODO: save inference with correct filenames
                os.makedirs(osp.join(save_dir, 'raw'), exist_ok=True)
                os.makedirs(osp.join(save_dir, 'painted'), exist_ok=True)
                for output, target in zip(outputs, targets):
                    output = output.argmax(dim=0).cpu().numpy().astype(np.int8)
                    output_image = Image.fromarray(output, mode='L')
                    output_image_painted = dataset.paint_inference(output_image)

                    output_image.save(
                        osp.join(save_dir, 'raw', "image_%d.png" % (i)) )
                    output_image_painted.save(
                        osp.join(save_dir, 'painted', "image_%d.png" % (i)) )


        eval_time = time.time() - eval_time

        print("Evaluation time: %.3fs., processed %d images" \
            % (eval_time, len(subset)))

        confusion.show_results()
        results = confusion.get_results()

        score = ModelScore()
        score.mean_iou = results.avg_class_iou
        score.mean_accuracy = results.overall_acc
        score.class_ious = results.class_iou
        score.class_accuracies = results.class_acc
        return score

def load_checkpoint(path):
    checkpoint = torch.load(path)
    model_path = osp.join(osp.dirname(path), checkpoint['model_state_path'])
    if osp.isfile(model_path):
        weights = torch.load(checkpoint['model_state_path'])
        checkpoint['model_state'] = weights
    else:
        raise Exception("Failed to load model state from '%s'" % (model_path))
    checkpoint.pop('model_state_path', None)
    return checkpoint

def save_checkpoint(state, path):
    state['model_state_path'] = 'model_%d.pth' % (state['epoch'])
    torch.save(state['model_state'],
        osp.join(osp.dirname(path), state['model_state_path']))
    state.pop('model_state', None)
    torch.save(state, path)

def adjust_optimizer_params(args, optimizer, iteration):
    lr = args.lr / (1.0 + args.lrd * iteration)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args, dataset, subsets, model, criterion, optimizer, checkpoint=None):
    train_subset = subsets['train']
    val_subset = subsets['val']

    best_model_score = ModelScore()
    global_iteration = 0

    if checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        args.start_epoch = checkpoint['epoch']
        best_model_score = checkpoint['best_model_score']
        global_iteration = checkpoint['global_iteration']
        print("Resuming from epoch '%d'" % (args.start_epoch))
        checkpoint = None

    for epoch in range(args.start_epoch, args.epochs):
        model.train()

        epoch_time = time.time()
        step_time = epoch_time
        for i, (inputs, targets) in enumerate(train_subset):
            adjust_optimizer_params(args, optimizer, global_iteration)

            inputs = inputs.to(args.device, non_blocking=True)
            targets = targets.to(args.device, non_blocking=True)

            outputs = model.forward(inputs)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_iteration == 0 and args.verbose:
                models.print_memory_stats(model, mode='train')

            global_iteration += 1

            # Print status
            if (args.print_freq) and (i % args.print_freq == 0):
                current_time = time.time()
                print("Epoch {epoch}, {iter}/{iters}, time {epoch_time:.4f}s"
                    " | step time {step_time:.4f}s, loss {loss:.5f},"
                    " lr {lr:.3e}" \
                    .format(
                        epoch=epoch, iter=i, iters=len(train_subset),
                        epoch_time=current_time - epoch_time,
                        step_time=current_time - step_time,
                        loss=loss.item(), lr=optimizer.param_groups[0]['lr']
                    ))
                step_time = current_time

        # Update best model
        is_best = False
        model_score = None
        if (args.eval_freq) and (epoch % args.eval_freq == 0):
            model_score = evaluate(args, dataset, val_subset, model,
                osp.join(args.inference_dir, epoch, 'val'))
            if model_score < best_model_score:
                best_model_score = model_score
                is_best = True
            else:
                is_best = False
            print('Epoch %d | model score: %.3f, best score: %.3f' \
                % (epoch, model_score, best_model_score))

        # Save model
        if (args.checkpoint_freq) and (epoch % args.checkpoint_freq == 0):
            state = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_model_score': best_model_score,
                'global_iteration': global_iteration,
            }
            save_checkpoint(state, osp.join(
                args.checkpoint_save_dir, 'checkpoint_%d.pth' % (epoch + 1)))

            if args.save_best and is_best:
                save_checkpoint(state,
                    osp.join(args.checkpoint_save_dir, 'best.pth'))
            state = None

def parse_args():
    def str2bool(value):
        return value.lower() in ['1', 'true', 'yes', 'y', 't']

    parser = argparse.ArgumentParser()

    parser.add_argument_group('General parameters')
    parser.add_argument('--model', default='resnet18_seg',
        choices=models.names,
        help='model architecture (default: %(default)s)')
    parser.add_argument('--backend', default='cuda',
        choices=['cpu', 'cuda'],
        help='backend (default: %(default)s)')
    parser.add_argument('--test', default=False, type=bool,
        help='do model testing (default: %(default)s)')
    parser.add_argument('--train', default=True, type=bool,
        help='do model training (default: %(default)s)')
    parser.add_argument('--verbose', default=False, type=str2bool,
        help='be verbose (default: %(default)s)')

    parser.add_argument_group('Data related')
    parser.add_argument('--data_dir', default='data',
        help='path to dataset (default: %(default)s)')
    parser.add_argument('--dataset', default='CamVid12',
        choices=datasets.names,
        help='dataset name (default: %(default)s)')
    parser.add_argument('--workers', default=1, type=int,
        help='number of data loading workers (default: %(default)s)')
    parser.add_argument('--normalize', default=True, type=str2bool,
        help='do data normalization before use (default: %(default)s)')

    parser.add_argument_group('Optimizer parameters')
    parser.add_argument('--batch_size', default=1, type=int,
        help='batch size (default: %(default)s)')
    parser.add_argument('--lr', default=0.1, type=float,
        help='initial learning rate (default: %(default)s)')
    parser.add_argument('--lrd', default=0, type=float,
        help='basic learning rate decay (default: %(default)s)')
    parser.add_argument('--momentum', default=0.9, type=float,
        help='momentum (default: %(default)s)')
    parser.add_argument('--wd', default=1e-4, type=float,
        help='weight decay (default: %(default)s)')

    parser.add_argument_group('Training related')
    parser.add_argument('--checkpoint', default='', type=str,
        help='path to latest checkpoint')
    parser.add_argument('--weights', default='', type=str,
        help='path to initial weights')
    parser.add_argument('--seed', default=None, type=int,
        help='seed for random values')
    parser.add_argument('--epochs', default=10, type=int,
        help='number of total epochs to run (default: %(default)s)')
    parser.add_argument('--start_epoch', default=0, type=int,
        help='epoch to start training from (default: %(default)s)')
    parser.add_argument('--print_freq', default=10, type=int,
        help='logging frequency (default: %(default)s)')
    parser.add_argument('--eval_freq', default=0, type=int,
        help='evaluation frequency on validation set (default: %(default)s)')
    parser.add_argument('--checkpoint_freq', default=0, type=int,
        help='checkpoint save frequency (default: %(default)s)')
    parser.add_argument('--checkpoint_save_dir', default='checkpoints',
        help='directory for checkpoints (default: %(default)s)')
    parser.add_argument('--save_best', default=False, type=bool,
        help='track best model (default: %(default)s)')

    parser.add_argument_group('Evaluation related')
    parser.add_argument('--inference_dir', default='inference',
        help='inference save directory (default: %(default)s)')
    parser.add_argument('--save_inference', default=False, type=bool,
        help='save inference during evaluation (default: %(default)s)')

    return parser.parse_args()

def main():
    args = parse_args()
    print("Executed with parameters:", args)
    args.device = torch.device(args.backend)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            'You have chosen to seed training. '
            'This will turn on the CUDNN deterministic setting, '
            'which can slow down your training considerably! '
            'You may see unexpected behavior when restarting '
            'from checkpoints.')
    else:
        cudnn.benchmark = True

    if (args.checkpoint_freq != 0) or (args.checkpoint_save_dir is not '') \
       or (args.save_best):

        assert(0 <= args.checkpoint_freq)
        assert(args.checkpoint_save_dir is not '')
        os.makedirs(args.checkpoint_save_dir, exist_ok=True)

    dataset = datasets.__dict__[args.dataset](args.data_dir, args.normalize)
    train_dataset = dataset.get_train()
    val_dataset = dataset.get_val()
    test_dataset = dataset.get_test()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model, criterion = models.make_segmentation_model(args.model,
        dataset.class_count)
    model = model.to(args.device)
    criterion = criterion.to(args.device)

    if args.weights:
        if osp.isfile(args.weights):
            model.load_state_dict(torch.load(args.weights))
        else:
            raise Exception("Failed to load model weights from '%d'" \
                % (args.weights))

    if args.train:
        optimizer = torch.optim.SGD(model.parameters(),
            args.lr, momentum=args.momentum, weight_decay=args.wd)

        checkpoint = None
        if args.checkpoint:
            if osp.isfile(args.checkpoint):
                print("Loading checkpoint from '%s'" % (args.checkpoint))
                checkpoint = torch.load(args.checkpoint)
            else:
                raise Exception("No checkpoint found at '%s'" \
                    % (args.checkpoint))

        subsets = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
        train(args, dataset, subsets, model, criterion, optimizer, checkpoint)

    if args.test:
        if args.verbose:
            sample_input = next(iter(test_loader))[0].to(args.device)
            models.print_memory_stats(model, sample_input,
                'train' if args.train else 'test')
        evaluate(args, dataset, subsets['test'], model)


if __name__ == '__main__':
    main()