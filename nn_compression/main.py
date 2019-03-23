import torch
import torch.optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn as nn

import nn_compression.datasets as datasets
import nn_compression.models as models
from nn_compression.models import save_checkpoint, load_checkpoint
from nn_compression.utils import ConfusionMatrix

import apex.fp16_utils

import argparse
import os
import os.path as osp
import time
import PIL.Image as Image
import numpy as np
import random
import warnings


def fast_collate(batch):
    print(batch)
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1][0] for target in batch], dtype=torch.int64)
    rest = [target[1:] for target in batch]
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)
        
    return tensor, targets, rest

class ModelScore:
    mean_iou = 0
    mean_accuracy = 0
    class_ious = []
    class_accuracies = []

    def __lt__(self, other):
        return self.mean_iou < other.mean_iou

def mask_targets(targets, real_sizes, mask_value=255):
    for target, real_size in zip(targets, real_sizes):
        target[:real_size[0], real_size[1]:].fill_(mask_value)
        target[real_size[0]:, :].fill_(mask_value)
    return targets

def evaluate(args, dataset, subset, model, save_dir=None):
    if args.save_inference and save_dir is not None:
        raw_save_dir = osp.join(save_dir, 'raw')
        painted_save_dir = osp.join(save_dir, 'painted')
        os.makedirs(raw_save_dir, exist_ok=True)
        os.makedirs(painted_save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        confusion = ConfusionMatrix(dataset.classes)

        print("Running evaluation...")

        samples_count = 0

        eval_time = time.time()

        for i, (inputs, targets, indices) in enumerate(subset):
            samples_count += len(inputs)

            if args.fp16:
                inputs = inputs.half()
            inputs = inputs.to(args.device, non_blocking=True)

            outputs = model.forward(inputs)

            if targets is not None:
                for output, target, idx in zip(outputs, targets, indices):
                    real_size = subset.dataset.get_size(idx)
                    output = output[:, :real_size[0], :real_size[1]].contiguous()
                    target = target[:real_size[0], :real_size[1]].contiguous()
                    confusion.update(output, target)

            if args.save_inference and save_dir is not None:
                for output, idx in zip(outputs, indices):
                    real_size = subset.dataset.get_size(idx)
                    output = output[:, :real_size[0], :real_size[1]]
                    output = output.argmax(dim=0).cpu().numpy().astype(np.uint8)
                    output_image_raw = Image.fromarray(output, mode='L')
                    output_image_painted = \
                        dataset.paint_inference(output_image_raw)

                    input_path = osp.basename(subset.dataset.get_path(idx)[0])
                    filename = input_path[:input_path.rfind('.')]
                    output_image_raw.save(osp.join(raw_save_dir,
                        filename + '.png'))
                    output_image_painted.save(osp.join(painted_save_dir,
                        filename + '.png'))


        eval_time = time.time() - eval_time

        print("Evaluation time: %.3fs., processed %d images" \
            % (eval_time, samples_count))

        confusion.show_results()
        results = confusion.get_results()

        score = ModelScore()
        score.mean_iou = results.avg_class_iou
        score.mean_accuracy = results.overall_acc
        score.class_ious = results.class_iou
        score.class_accuracies = results.class_acc
        return score

def get_model_outputs(args, model, subset):
    with torch.no_grad():
        model.eval()

        results = None
        for _, (inputs, _, indices) in enumerate(subset):
            for input, index in zip(inputs, indices):
                input = input.to(args.device, non_blocking=True)
                input = input.view(1, *input.size())
                output = model.forward(input)[0]

                if results is None:
                    results = torch.empty(len(subset.dataset), *output.size(), 
                        device='cpu')
                results[index] = output
    return results


LRD_TYPES = ['smooth', 'exp']

def adjust_optimizer_params(args, optimizer, iteration, epoch):
    if iteration < args.lr_warmup_iter:
        return

    if args.lrd_type == 'exp': 
        lr = args.lr * (2.0 ** -(1.0 + args.lrd * epoch))
    elif args.lrd_type == 'smooth':
        lr = args.lr / (1.0 + args.lrd * (iteration - args.lr_warmup_iter))
    else:
        assert False, "Unexpected learning rate decay type"

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args, dataset, subsets, model, criterion, optimizer, checkpoint=None):
    train_subset = subsets['train']
    val_subset = subsets['val']

    best_score = ModelScore()
    global_iteration = 0

    args.lrd /= len(train_subset)

    if checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        args.start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint['best_score']
        global_iteration = checkpoint['global_iteration']
        print("Resuming from epoch '%d'" % (args.start_epoch))
        del checkpoint

    if args.distill:
        distillation_model = args.distillation_model
        distillation_criterion = args.distillation_criterion
        distillation_loss_scale = args.distillation_loss_scale
        distillation_model.eval()

        if args.cache_distillation_outputs:
            distillation_model.to(args.device)
            
            print('Checking distillation model results')
            score = evaluate(args, dataset, val_subset, distillation_model)
            print("Model score on '%s': mIoU %.3f, mAcc %.3f" % \
                ('val', score.mean_iou, score.mean_accuracy))

            print('Building the distillation model outputs cache')
            distillation_cache = get_model_outputs(args, distillation_model, train_subset)
            def take(tensor, indices):
                return torch.stack([tensor[idx] for idx in indices])
            distillation_targets = lambda indices: take(distillation_cache, indices)
            print('Cache built')

            distillation_model.to('cpu')

    for epoch in range(args.start_epoch, args.epochs):
        model.train()

        epoch_time = time.time()
        step_time = epoch_time
        for i, (inputs, targets, indices) in enumerate(train_subset):
            adjust_optimizer_params(args, optimizer, global_iteration, epoch)

            if args.fp16:
                inputs = inputs.half()
                targets = targets.half()
            inputs = inputs.to(args.device, non_blocking=True)
            targets = targets.to(args.device, non_blocking=True)

            # if dataset has different image sizes:
            # mask_targets(targets,
            #     [train_subset.dataset.get_size(i) for i in indices])

            optimizer.zero_grad()

            outputs = model.forward(inputs)

            loss = criterion(outputs, targets)
            if args.distill:
                if args.cache_distillation_outputs:
                    dtargets = distillation_targets(indices).to(args.device)
                else:
                    dtargets = distillation_model.forward(inputs)

                dloss = distillation_criterion(outputs, dtargets)
                if (args.print_freq) and (i % args.print_freq == 0):
                    print('Loss: %.3f, distillation loss: %.3f' % \
                        (loss.item(), dloss.item()))

                if args.distillation_rescale:
                    current_ratio = (loss + dloss).item()
                    current_ratio = [loss.item() / current_ratio + 1e-6, 
                                     1 - loss.item() / current_ratio]
                    loss = (1.0 - distillation_loss_scale) / current_ratio[0] * loss + \
                           distillation_loss_scale / current_ratio[1] * dloss
                else:
                    loss += distillation_loss_scale * dloss

            assert torch.all(torch.isfinite(loss)), 'Optmization has diverged.'

            if args.fp16:
                optimizer.backward(loss)
            else:
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
        current_score = None
        if args.eval_freq and (epoch != 0) and (epoch % args.eval_freq == 0):
            current_score = evaluate(args, dataset, val_subset, model,
                osp.join(args.inference_dir, 'epoch_%d' % (epoch), 'val'))
            if best_score < current_score:
                best_score = current_score
                is_best = True
            else:
                is_best = False
            print('Epoch {epoch} | current score: '
                'mIoU {current_miou:.3f}, mAcc {current_macc:.3f}'
                ', best score: mIoU {best_miou:.3f}, mAcc {best_macc:.3f}' \
                .format(
                    epoch=epoch,
                    current_miou=current_score.mean_iou,
                    current_macc=current_score.mean_accuracy,
                    best_miou=best_score.mean_iou,
                    best_macc=best_score.mean_accuracy
                ))

            for subset_name in ['train', 'test']:
                print("Testing the model on '%s' subset" % (subset_name)) 
                evaluate(args, dataset, subsets[subset_name], model)

        # Save the model
        if args.checkpoint_freq and (epoch % args.checkpoint_freq == 0) or \
           args.save_best and is_best:

            state = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'best_score': best_score,
                'global_iteration': global_iteration,
            }

            if args.checkpoint_freq and (epoch % args.checkpoint_freq == 0):
                path = osp.join(
                    args.checkpoint_save_dir, 'checkpoint_%d.pth' % (epoch))
                print('Saving checkpoint as "%s"' % (path))
                save_checkpoint(state, path)

            if args.save_best and is_best:
                path = osp.join(args.checkpoint_save_dir, 'best.pth')
                print('Saving new best model as "%s"' % (path))
                save_checkpoint(state, path)

            state = None

def parse_args():
    def str2bool(value):
        return value.lower() in ['1', 'true', 'yes', 'y', 't']

    parser = argparse.ArgumentParser()

    parser.add_argument_group('General parameters')
    parser.add_argument('-f', '--config', default=None, metavar='PATH',
        help='Load experiment settings from the specified file')

    parser.add_argument('--model', default='resnet18_seg',
        choices=models.names,
        help='model architecture (default: %(default)s)')
    parser.add_argument('--backend', default='cuda',
        choices=['cpu', 'cuda'],
        help='backend (default: %(default)s)')
    parser.add_argument('--test', default=False, type=str2bool,
        help='do model testing (default: %(default)s)')
    parser.add_argument('--train', default=True, type=str2bool,
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
    parser.add_argument('--lrd_type', default='smooth', 
        choices=LRD_TYPES, 
        help='learning rate decay type (default: %(default)s)')
    parser.add_argument('--momentum', default=0.9, type=float,
        help='momentum (default: %(default)s)')
    parser.add_argument('--wd', default=1e-4, type=float,
        help='weight decay (default: %(default)s)')
    parser.add_argument('--lr_warmup_iter', default=1, type=int,
        help='warmup iterations (without decay) (default: %(default)s)')

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
    parser.add_argument('--save_best', default=False, type=str2bool,
        help='track best model (default: %(default)s)')
    parser.add_argument('--optimizer', default='SGD', choices=['Adam', 'SGD'],
        help='track best model (default: %(default)s)')
    parser.add_argument('--fp16', action='store_true',
        help='Run model fp16 mode.')
    parser.add_argument('--static-loss-scale', type=float, default=1,
        help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--dynamic-loss-scale', action='store_true',
        help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
        '--static-loss-scale.')
    parser.add_argument('--prof', dest='prof', action='store_true',
        help='Only run 10 iterations for profiling.')
    parser.add_argument('-dm', '--distillation_model', default=None,
        help='The trained model for distillation learning')
    parser.add_argument('-dw', '--distillation_model_weights', default=None,
        help='The trained model weights')
    parser.add_argument('-ds', '--distillation_loss_scale', default=1.0, type=float,
        help='The multiplier for distillation loss')
    parser.add_argument('-dr', '--distillation_rescale', default=True, type=str2bool,
        help='Rescale distillation loss using the multiplier (default: %(default)s)')
    parser.add_argument('--cache_distillation_outputs', 
        default=False, type=str2bool,
        help='Run distillation model before training and save the outputs (default: %(default)s)')
    parser.add_argument('--test_on_best', type=str2bool, default=False,
        help='test on best model after training (default: %(default)s)')

    parser.add_argument_group('Evaluation related')
    parser.add_argument('--inference_dir', default='inference',
        help='inference save directory (default: %(default)s)')
    parser.add_argument('--save_inference', default=False, type=str2bool,
        help='save inference during evaluation (default: %(default)s)')
    parser.add_argument('--testing_subsets', default='test', 
        type=lambda s: s.split(','),
        help='subsets to use for model evaliation during testing'
             ' (comma separated, default: %(default)s')


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
            'You have chosen to do seed training. '
            'This will turn on the CUDNN deterministic setting, '
            'which can slow down your training considerably! '
            'You may see unexpected behavior when restarting '
            'from checkpoints.')
    else:
        cudnn.benchmark = True

    if args.fp16:
        assert torch.backends.cudnn.enabled, \
            "fp16 mode requires cudnn backend to be enabled."

    if args.static_loss_scale != 1.0:
        if not args.fp16:
            warnings.warn(
                "If --fp16 is not used, static_loss_scale will be ignored.")

    if (args.checkpoint_freq != 0) or (args.checkpoint_save_dir is not '') \
       or (args.save_best):

        assert(0 <= args.checkpoint_freq)
        assert(args.checkpoint_save_dir is not '')
        os.makedirs(args.checkpoint_save_dir, exist_ok=True)

    if args.distillation_model:
        assert osp.isfile(args.distillation_model_weights)
    args.distill = args.distillation_model is not None

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

    subsets = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

    print("Creating model '%s'" % (args.model))
    model, criterion = models.make_segmentation_model(args.model,
        dataset.class_count)

    if args.fp16:
        model = apex.fp16_utils.network_to_half(model)

    if args.weights:
        def load_weights(model, weights):
            if osp.isfile(weights):
                print("Loading model weights from '%s'" % (weights))
                model.load_state_dict(torch.load(weights), False)
            else:
                raise Exception("Failed to load model weights from '%s'" \
                    % (weights))
        load_weights(model, args.weights)

    model = model.to(args.device)
    criterion = criterion.to(args.device)

    if args.train:
        if args.distill:
            print("Distilling the knowledge of '%s'" % (args.distillation_model))
            distillation_model, _ = models.make_segmentation_model(
                args.distillation_model, dataset.class_count)
            print("Loading distillation model weights from '%s'" % \
                (args.distillation_model_weights))
            distillation_model.load_state_dict(
                torch.load(args.distillation_model_weights), False)
            distillation_model = distillation_model.to(args.device)
            args.distillation_model = distillation_model

            def MSELoss(x, y):
                # required to reduce cancellation during math operations
                d = (x - y) ** 2.0
                while 0 < d.dim():
                    d = d.mean(d.dim() - 1)
                return d
            args.distillation_criterion = MSELoss

        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(),
                args.lr, momentum=args.momentum, weight_decay=args.wd)
        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(),
                args.lr, weight_decay=args.wd)

        if args.fp16:
            optimizer = FP16_Optimizer(optimizer,
                static_loss_scale=args.static_loss_scale,
                dynamic_loss_scale=args.dynamic_loss_scale)

        checkpoint = None
        if args.checkpoint:
            if osp.isfile(args.checkpoint):
                print("Loading checkpoint from '%s'" % (args.checkpoint))
                checkpoint = load_checkpoint(args.checkpoint)
            else:
                raise Exception("Not found checkpoint at '%s'" \
                    % (args.checkpoint))

        torch.cuda.empty_cache()
        train(args, dataset, subsets, model, criterion, optimizer, checkpoint)

        if args.save_best and args.test_on_best:
            model.load_state_dict(torch.load(
            	osp.join(args.checkpoint_save_dir, 'best_model.pth')))
            for subset_name in ['train', 'val', 'test']:
                print("Testing the model on '%s' subset" % (subset_name)) 
                evaluate(args, dataset, subsets[subset_name], model)

    if args.test:
        testing_subsets = args.testing_subsets
        if args.verbose and len(testing_subsets) != 0:
            subset_name = testing_subsets[0]
            sample_input = next(iter(test_loader))[0].to(args.device)
            models.print_memory_stats(model, sample_input, mode='test')
        for subset_name in testing_subsets:
            print("Testing the model on '%s' subset" % (subset_name)) 
            evaluate(args, dataset, subsets[subset_name], model,
                osp.join(args.inference_dir, subset_name))


if __name__ == '__main__':
    main()