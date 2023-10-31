# Credits: https://github.com/thuml/Transfer-Learning-Library
import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
import os
# import wandb

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append('../')
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss, ImageClassifier
from dalib.adaptation.mcc import MinimumClassConfusionLoss
# from dalib.modules.masking import Masking
# from dalib.modules.teacher import EMATeacher
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance
from common.utils.sam import SAM

from dalib.modules.p_infill import P_Infill
from dalib.modules.p_shuffle import P_Shuffle, P_Selective_Shuffle
from dalib.modules.p_rotate import P_Rotate, P_Selective_Rotate
from dalib.modules.p_shuffle_rotate import P_Shuffle_Rotate
from dalib.modules.randcrop import RandCropForBatch

from dalib.adaptation.triplet import (TripletLossInLatentSpaceCosSim, TripletLossInLatentSpaceCosSimv2,
                                      TripletLossInLatentSpaceCosSimv2_plus1, TripletLossInLatentSpaceCosSimv2_plus2, TripletLossInLatentSpaceCosSimv2_plus3,
                                      TripletLossInLatentSpaceCosSimv3, TripletLossInLogisticDist)

from torch.utils.tensorboard import SummaryWriter

sys.path.append('.')
import utils
# breakpoint()
import numpy as np

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase, args)
    training_start_time = time.time()
    print(args)

    # if args.log_results:
    #     wandb.init(
    #         project="MIC",
    #         name=args.log_name)
    #     wandb.config.update(args)
    # print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        # warnings.warn('You have chosen to seed training. '
        #               'This will turn on the CUDNN deterministic setting, '
        #               'which can slow down your training considerably! '
        #               'You may see unexpected behavior when restarting '
        #               'from checkpoints.')
        np.random.seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True
    device = args.device

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source,
                          args.target, train_transform, val_transform)
    if args.phase=='test':
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    
    else:
        train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
        train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        train_source_iter = ForeverDataIterator(train_source_loader)
        train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    print(backbone)
    pool_layer = nn.Identity() if args.no_pool else None
    # classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
    #                              pool_layer=pool_layer, finetune=not args.scratch).to(device)
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                             pool_layer=pool_layer, finetune=not args.scratch)
    classifier = torch.nn.DataParallel(classifier).to(device)
    classifier_feature_dim = classifier.module.features_dim

    if args.randomized:
        domain_discri = DomainDiscriminator(
            args.randomized_dim, hidden_size=1024).to(device)
    else:
        domain_discri = DomainDiscriminator(
            classifier_feature_dim * num_classes, hidden_size=1024).to(device)

    # define optimizer and lr scheduler
    base_optimizer = torch.optim.SGD
    ad_optimizer = SGD(domain_discri.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    optimizer = SAM(classifier.module.get_parameters(), base_optimizer, rho=args.rho, adaptive=False,
                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr *
                            (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    lr_scheduler_ad = LambdaLR(
        ad_optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    domain_adv = ConditionalDomainAdversarialLoss(
        domain_discri, entropy_conditioning=args.entropy,
        num_classes=num_classes, features_dim=classifier_feature_dim, randomized=args.randomized,
        randomized_dim=args.randomized_dim
    ).to(device)

    mcc_loss = MinimumClassConfusionLoss(temperature=args.temperature)

    if args.neg_aug_type == 'shuffle':
        if args.neg_aug_ratio is None:
            negative_aug = P_Shuffle(args.neg_aug_patch_size)
        else:
            print('\nargs.neg_aug_ratio is {}!!\nP_Selective_Shuffle is used!\n'.format(args.neg_aug_ratio))
            negative_aug = P_Selective_Shuffle(args.neg_aug_patch_size, args.neg_aug_ratio)
    elif args.neg_aug_type == 'rotate':
        if args.neg_aug_ratio is None:
            negative_aug = P_Rotate(args.neg_aug_patch_size)
        else:
            print('\nargs.neg_aug_ratio is {}!!\nP_Selective_Rotate is used!\n'.format(args.neg_aug_ratio))
            negative_aug = P_Selective_Rotate(args.neg_aug_patch_size, args.neg_aug_ratio)
    elif args.neg_aug_type == 'infill':
        negative_aug = P_Infill(args.neg_aug_replace_rate)
    elif args.neg_aug_type == 'shuffle_rotate' or args.neg_aug_type == 'shufrot':
        print('\nP_Shuffle_Rotate is used with args.neg_aug_ratio {}!!\n'.format(args.neg_aug_ratio))
        negative_aug = P_Shuffle_Rotate(args.neg_aug_patch_size, args.neg_aug_ratio)
    else:
        raise Exception("Invalid value of args.neg_aug_type: {}".format(args.neg_aug_type))

    # define triplet loss
    if args.triplet_type == 'latent':
        triplet_loss = TripletLossInLatentSpaceCosSim(args.triplet_entropy_cond, args.triplet_temp)
    elif args.triplet_type == 'latentv2' or args.triplet_type == 'NVC' or args.triplet_type == 'nvc' or args.triplet_type == 'latentv2_bd':
        triplet_loss = TripletLossInLatentSpaceCosSimv2(args.triplet_temp)
    elif args.triplet_type == 'latentv2_plus1':
        triplet_loss = TripletLossInLatentSpaceCosSimv2_plus1(args.triplet_temp)
    elif args.triplet_type == 'latentv2_plus2':
        triplet_loss = TripletLossInLatentSpaceCosSimv2_plus2(args.triplet_temp)
    elif args.triplet_type == 'latentv2_plus3':
        triplet_loss = TripletLossInLatentSpaceCosSimv2_plus3(args.triplet_temp)
    elif args.triplet_type == 'latentv3':
        triplet_loss = TripletLossInLatentSpaceCosSimv3(args.triplet_temp, args.triplet_temp_weight, args.triplet_lambda, args.triplet_thres_weight)
    elif args.triplet_type == 'logistic':
        triplet_loss = TripletLossInLogisticDist(args.triplet_entropy_cond, args.triplet_beta, args.triplet_dist)
    else:
        raise NotImplementedError("Invalid args.triplet_type: {}".format(args.triplet_type))

    # define random crop and other rand augs
    random_crop = RandCropForBatch(args.target_size, args.denorm_and_toPIL,
                                   args.s_colorjitter,
                                   args.p_colorjitter,
                                   args.p_grayscale,
                                   args.gaussian_blur,
                                   args.p_horizontalflip)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(
            logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.module.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(
            classifier.module.backbone, classifier.module.pool_layer, classifier.module.bottleneck).to(device)
        source_feature = collect_feature(
            train_source_loader, feature_extractor, device)
        target_feature = collect_feature(
            train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.pdf')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(
            source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    # start training
    tb_writer = SummaryWriter(log_dir = os.path.join(args.log, 'tensorboard'))
    best_acc1 = 0.
    for epoch in range(args.epochs):
        print("lr_bbone:", lr_scheduler.get_last_lr()[0])
        print("lr_btlnck:", lr_scheduler.get_last_lr()[1])
        # if args.log_results:
        #     wandb.log({"lr_bbone": lr_scheduler.get_last_lr()[0],
        #                "lr_btlnck": lr_scheduler.get_last_lr()[1]})
        # train for one epoch

        train(train_source_iter, train_target_iter, classifier,
              domain_adv, mcc_loss, random_crop, #masking, 
              negative_aug, triplet_loss,
              optimizer, ad_optimizer,
              lr_scheduler, lr_scheduler_ad, epoch, args, tb_writer)
        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)
        tb_writer.add_scalar("Validation/acc1", acc1, epoch)
        # if args.log_results:
        #     wandb.log({'epoch': epoch, 'val_acc': acc1})

        # remember best acc@1 and save checkpoint
        torch.save(classifier.module.state_dict(),
                   logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'),
                        logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)
        tb_writer.add_scalar("Validation/best_acc1", best_acc1, epoch)

    #print("best_acc1 = {:3.1f}".format(best_acc1))
    if not args.per_class_eval:
        print("best_acc1 = {:3.1f}".format(best_acc1))
    else:
        print("mean_acc = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.module.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = utils.validate(test_loader, classifier, args, device)
    if not args.per_class_eval:
        print("test_acc1 = {:3.1f}".format(acc1))
    else:
        print("test_mean_acc = {:3.1f}".format(acc1))
    # if args.log_results:
    #     wandb.log({'epoch': epoch, 'test_acc': acc1})
    training_end_time = time.time()
    time_consumption = training_end_time-training_start_time
    import datetime
    time_consumption = datetime.timedelta(seconds=time_consumption)
    
    print("[Info] Time consumption (hh:mm:ss.ms) is {}".format(str(time_consumption)))
    
    tb_writer.close()
    logger.close()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model,
          domain_adv: ConditionalDomainAdversarialLoss, mcc, random_crop,
          negative_aug, triplet_loss,
          optimizer, ad_optimizer,
          lr_scheduler: LambdaLR, lr_scheduler_ad, epoch: int, args: argparse.Namespace, tb_writer):
    data_time = AverageMeter('Data_Time(unit: sec)', ':3.1f')
    batch_time = AverageMeter('Batch_Time(unit: sec)', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    triplet_losses = AverageMeter('Triplet Loss ({})'.format(args.triplet_type))
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [data_time, batch_time, losses, trans_losses, triplet_losses, cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    domain_adv.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_t, _ = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)

        labels_s = labels_s.to(device)

        x_t_negaug = negative_aug(x_t)
        x_t_rand = random_crop(x_t)

        # measure data loading time
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        ad_optimizer.zero_grad()

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        # breakpoint()
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)
        cls_loss = F.cross_entropy(y_s, labels_s)
        mcc_loss_value = mcc(y_t)

        y_t_negaug, f_t_negaug = model(x_t_negaug)
        y_t_rand, f_t_rand = model(x_t_rand)
        
        if args.triplet_type == 'latent':
            if args.triplet_entropy_cond:
                triplet_loss_value = triplet_loss(anchor=f_t, pos=f_t_rand, neg=f_t_negaug, pred_anchor=y_t)
            else:
                triplet_loss_value = triplet_loss(anchor=f_t, pos=f_t_rand, neg=f_t_negaug)
        elif args.triplet_type == 'latentv2' or args.triplet_type == 'NVC' or args.triplet_type == 'nvc' or args.triplet_type == 'latentv3' or 'latentv2_plus' in args.triplet_type:
            triplet_loss_value = triplet_loss(anchor=f_t, pos=f_t_rand, neg=f_t_negaug, anchor_detach=False)
        elif args.triplet_type == 'latentv2_bd':
            triplet_loss_value = 0.5 * (triplet_loss(anchor=f_t, pos=f_t_rand, neg=f_t_negaug, anchor_detach=False) + triplet_loss(anchor=f_t_rand, pos=f_t, neg=f_t_negaug, anchor_detach=False))
        elif args.triplet_type == 'logistic':
            triplet_loss_value = triplet_loss(pred_anchor=y_t, pred_pos=y_t_rand, pred_neg=y_t_negaug)

        loss = cls_loss + mcc_loss_value + triplet_loss_value * args.triplet_coef

        loss.backward()

        # Calculate ϵ̂ (w) and add it to the weights
        optimizer.first_step(zero_grad=True)

        # Calculate task loss and domain loss
        y, f = model(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s)

        y_t_negaug, f_t_negaug = model(x_t_negaug)
        y_t_rand, f_t_rand = model(x_t_rand)

        if args.triplet_type == 'latent':
            if args.triplet_entropy_cond:
                triplet_loss_value = triplet_loss(anchor=f_t, pos=f_t_rand, neg=f_t_negaug, pred_anchor=y_t)
            else:
                triplet_loss_value = triplet_loss(anchor=f_t, pos=f_t_rand, neg=f_t_negaug)
        elif args.triplet_type == 'latentv2' or args.triplet_type == 'NVC' or args.triplet_type == 'nvc' or args.triplet_type == 'latentv3' or 'latentv2_plus' in args.triplet_type:
            triplet_loss_value = triplet_loss(anchor=f_t, pos=f_t_rand, neg=f_t_negaug, anchor_detach=False)
        elif args.triplet_type == 'latentv2_bd':
            triplet_loss_value = 0.5 * (triplet_loss(anchor=f_t, pos=f_t_rand, neg=f_t_negaug, anchor_detach=False) + triplet_loss(anchor=f_t_rand, pos=f_t, neg=f_t_negaug, anchor_detach=False))
        elif args.triplet_type == 'logistic':
            triplet_loss_value = triplet_loss(pred_anchor=y_t, pred_pos=y_t_rand, pred_neg=y_t_negaug)
        
        adv_loss_value = domain_adv(y_s, f_s, y_t, f_t)
        mcc_loss_value = mcc(y_t)
        # transfer_loss = domain_adv(y_s, f_s, y_t, f_t) + mcc(y_t)
        transfer_loss = adv_loss_value + mcc_loss_value
        domain_acc = domain_adv.domain_discriminator_accuracy
        loss = cls_loss + transfer_loss * args.trade_off + triplet_loss_value * args.triplet_coef

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc, x_s.size(0))
        domain_accs.update(domain_acc, x_s.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))
        triplet_losses.update(triplet_loss_value.item(), x_s.size(0))

        if (i+1) % args.print_freq == 0 or i == 0:
            global_iter = args.iters_per_epoch * epoch + i
            tb_writer.add_scalar("Train/cls_acc(source-domain)", cls_acc, global_iter)
            tb_writer.add_scalar("Train/domain_acc", domain_acc, global_iter)
            tb_writer.add_scalar("Train/total_loss(weighted sum)", loss.item(), global_iter)
            tb_writer.add_scalar("Train/adv_loss", adv_loss_value.item(), global_iter)
            tb_writer.add_scalar("Train/mcc_loss", mcc_loss_value.item(), global_iter)
            tb_writer.add_scalar("Train/triplet_loss", triplet_loss_value.item(), global_iter)

        loss.backward()
        # Update parameters of domain classifier
        
        ad_optimizer.step()
        # Update parameters (Sharpness-Aware update)
        optimizer.second_step(zero_grad=True)

        lr_scheduler.step()
        lr_scheduler_ad.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0 or i == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CDAN+MCC with SDAT for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true',
                        help='whether train from scratch.')
    parser.add_argument('-r', '--randomized', action='store_true',
                        help='using randomized multi-linear-map (default: False)')
    parser.add_argument('-rd', '--randomized-dim', default=1024, type=int,
                        help='randomized dimension when using randomized multi-linear-map (default: 1024)')
    parser.add_argument('--entropy', default=False,
                        action='store_true', help='use entropy conditioning')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001,
                        type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75,
                        type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9,
                        type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='cdan',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    # parser.add_argument('--log_results', action='store_true',
    #                     help="To log results in wandb")
    # parser.add_argument('--gpu', type=str, default="0", help="GPU ID")
    # parser.add_argument('--log_name', type=str,
    #                     default="log", help="log name for wandb")
    parser.add_argument('--rho', type=float, default=0.05, help="GPU ID")
    parser.add_argument('--temperature', default=2.0,
                        type=float, help='parameter temperature scaling')
    # random crop and other rand augs
    parser.add_argument('--target_size', default=224, type=int, help="The size of crop obtained after random crop")
    parser.add_argument('--denorm_and_toPIL', action='store_true', help="Whether or not to denorm and transform image to PIL before applying random crop & other rand augs.")
    parser.add_argument('--p_horizontalflip', default=0, type=float, help="scale factor multiplied to parameters used in colorjitter")
    parser.add_argument('--s_colorjitter', default=1, type=float, help="scale factor multiplied to parameters used in colorjitter")
    parser.add_argument('--p_colorjitter', default=0.8, type=float, help="probability of colorjitter. If you don't want to colorjitter, please set 0.")
    parser.add_argument('--p_grayscale', default=0.2, type=float, help="probability of grayscale. If you don't want to grayscale, please set 0.")
    parser.add_argument('--gaussian_blur', action='store_true', help="Whether or not you want to apply gaussian blur in the end of augmentation.")
    # negative augmentation
    parser.add_argument('--neg_aug_type', default='shuffle', type=str, help="Choose from ['shuffle', 'rotate', 'infill', 'shuffle_rotate', 'shufrot']") # 'shuffle_rotate' and 'shufrot' are the same
    parser.add_argument('--neg_aug_replace_rate', default=0.25, type=str, help="Choose from [0.25, 0.375]. When --neg_aug_type is 'infill', this flag must be specified.")
    parser.add_argument('--neg_aug_patch_size', default=32, type=int, help="When --neg_aug_type is 'shuffle' or 'rotate', this flag must be specified. Recommendation: multiple of size of a patch of input image.")
    parser.add_argument('--neg_aug_ratio', default=1.0, type=float, help="When --neg_aug_type is 'shuffle', this flag can be used for define how many patches are selected to be shuffled. Recommendation: quotient of the total number of patches.")
    # triplet loss
    parser.add_argument('--triplet_type', default='NVC', type=str, help="Choose from ['latent', 'latentv2', 'NVC', 'nvc', 'latentv2_bd', 'latentv2_plus1', 'latentv2_plus2', 'latentv2_plus3', 'latentv3', 'logistic']") # 'latentv2', 'NVC', and 'nvc' are all the same.
    parser.add_argument('--triplet_entropy_cond', action='store_true', help="Whether or not entropy conditioning is used for computing triplet loss.")
    parser.add_argument('--triplet_dist', default='l2_square', type=str, help="Choose from ['l2_square', 'l2', 'l1', 'smooth_l1']")
    parser.add_argument('--triplet_coef', default=0.5, type=float, help="The weight for triplet loss term")
    parser.add_argument('--triplet_temp', default=0.5, type=float, help="The value of temperature, which is used when --triplet_type is 'latent'.")
    parser.add_argument('--triplet_temp_weight', default=13.5, type=float, help="The value of temperature for weights of pseudo-positive terms.")
    parser.add_argument('--triplet_lambda', default=1.0, type=float, help="hyperparameter value of pseudo-positive terms in latentv3. This flag can be used only when --triplet_type is 'latentv3'.")
    parser.add_argument('--triplet_beta', default=0.5, type=float, help="hyperparameter value for smooth L1 loss. This is used when --triplet type is 'logistic' and --triplet_dist is 'smooth_l1'.")
    parser.add_argument('--triplet_thres_weight', default=None, type=float, help="threshold value for weights of pseudo-positive terms in latentv3. This flag can be used only when --triplet_type is 'latentv3'.")
    

    # When using 'Product' dataset
    parser.add_argument('--test_difficulty', default='', type=str, choices=['', '1', '2', '3'], help="Only when using 'Product' dataset")

    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    main(args)
