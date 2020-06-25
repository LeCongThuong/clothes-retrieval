import os
import sys
import shutil
import argparse
from datetime import datetime
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import glob

from backbones.resnet_models import ResNetModels
from backbones.densenet_models import DenseNetModels
from backbones import model_factory
from dataloader import ClothesDataset, augmentation
from samplers import pk_sampler, pk_sample_full_coverage_epoch
from utils import (get_lr, set_lr, log_experience,
                   get_scheduler, get_sampler, parse_arguments,
                   compute_predictions, get_summary_writer)
from losses.triplet_loss import TripletLoss
from losses.arcface import ArcMarginProduct


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parse_arguments()

    image_paths_list = args.image_paths_list
    labels_file = args.labels_file
    writer, date_id = get_summary_writer(args)
    data_transform = augmentation(args.image_size, train=True)

    output_folder = log_experience(args, date_id, data_transform)

    # read labels from file
    with open(labels_file, 'r') as f:
        content = f.readlines()
    classes = [x.strip() for x in content]

    # mapping labels to id number
    mapping_label_id = dict(zip(classes, range(len(classes))))
    num_classes = len(classes)

    # read all image paths for training
    with open(image_paths_list, 'r') as f:
        file_list = f.readlines()
    paths = [x.strip() for x in file_list]

    # mapping file paths to id number
    mapping_files_to_global_id = dict(zip(paths, range(len(paths))))

    # read dictionary that maps from label to image paths
    with open(args.label_to_samples_file, 'r') as f:
        labels_to_samples = dict(json.load(f))

    # read eval image paths from file
    with open(args.eval_paths_file, 'r') as f:
        eval_file_list = f.readlines()
        eval_paths = [x.strip() for x in eval_file_list]

    # generate model architecture
    model_params = {
        'embedding_dim': args.embedding_dim,
        'num_classes': num_classes,
        'image_size': args.image_size,
        'archi': args.archi,
        'pretrained': bool(args.pretrained),
        'dropout': args.dropout,
        'alpha': args.alpha
    }
    model = model_factory.get_model(**model_params)

    # define loss
    if args.loss == 'arcface':
        arcface_loss = ArcMarginProduct(args.embedding_dim, num_classes)
        arcface_loss.to(device)
    else:
        if args.margin == -1:
            triplet_loss = TripletLoss(margin='soft', sample=False)
        else:
            triplet_loss = TripletLoss(margin=args.margin, sample=False)
        triplet_loss.to(device)
    # set init epoch
    start_epoch = args.start_epoch
    # load checkpoint if args.weights parameter is not None
    if args.checkpoint is not None:
        print('loading pre-trained weights and changing input size ...')
        checkpoint = torch.load(args.checkpoint)
        # start_epoch = checkpoint['epoch']
        if 'state_dict' in checkpoint.keys():
            backbone = checkpoint['state_dict']
        try:
            backbone.pop('model.classification_layer.weight')
            backbone.pop('model.classification_layer.bias')
        except KeyError:
            print('Key Error: No classification layer, skipp')
        model.load_state_dict(backbone, strict=False)
    model.to(device)

    # define optimizer
    my_list = ['model.classification_layer.weight', 'model.classification_layer.bias']
    params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))))
    base_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))))

    if args.loss == 'arcface':
        optimizer = Adam([{'params': base_params},
                          {'params': params, "lr": args.fs_lr},
                          {'params': arcface_loss.parameters()}],
                            lr=args.lr,
                         weight_decay=args.wd)
    else:
        optimizer = Adam([{'params': base_params},
                          {'params': params, "lr": args.fs_lr},
                          {'params': triplet_loss.parameters()}],
                           lr=args.lr,
                           weight_decay=args.wd)

    # define scheduler
    scheduler = get_scheduler(args, optimizer)
    # if args.checkpoint is not None:
    # #    weights = torch.load(args.weights)
    #     print('loading pre-trained scheduler and optimizer state ...')
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     scheduler.load_state_dict(checkpoint['scheduler'])

    dataset = ClothesDataset(paths=paths, mapping_label_id=mapping_label_id, transform=data_transform)

    sampler = get_sampler(args,
                          dataset,
                          classes,
                          labels_to_samples,
                          mapping_files_to_global_id)

    dataloader = DataLoader(dataset,
                            sampler=sampler,
                            batch_size=args.batch_size,
                            drop_last=True,
                            num_workers=args.num_workers)
    model.train()

    # start training loop
    for epoch in tqdm(range(start_epoch, args.epochs + start_epoch)):
        if args.loss == 'arcface':
            params = {
                'model': model,
                'dataloader': dataloader,
                'optimizer': optimizer,
                'criterion': arcface_loss,
                'logging_step': args.logging_step,
                'epoch': epoch,
                'epochs': args.epochs,
                'writer': writer,
                'date_id': date_id,
                'scheduler': scheduler,
                'output_folder': output_folder
            }
            arcface_train(**params)
        else:
            params = {
                'model': model,
                'dataloader': dataloader,
                'optimizer': optimizer,
                'criterion': triplet_loss,
                'logging_step': args.logging_step,
                'epoch': epoch,
                'epochs': args.epochs,
                'writer': writer,
                'date_id': date_id,
                'scheduler': scheduler,
                'output_folder': output_folder
            }

            triplet_train(**params)

        if epoch <= args.warmup_epochs:
            print("In warmup process, not save model")
        if (args.checkpoint_period != -1) & ((epoch + 1) % args.checkpoint_period == 0):
            state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scheduler': scheduler.state_dict()

            }
            save_epoch = epoch % 3
            torch.save(state,
                       os.path.join(output_folder,
                                    f'model_{save_epoch}.pth'))

        scheduler.step()
        # if epoch % 5 == 0:
        #     compute_predictions(args,
        #                         model,
        #                         paths,
        #                         eval_paths,
        #                         mapping_label_id,
        #                         date_id,
        #                         writer,
        #                         epoch)


def arcface_train(model, dataloader, optimizer, criterion, logging_step, epoch, epochs, writer, date_id, output_folder, scheduler):
    losses = []
    losses_arcface = []
    losses_ce = []

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
        images = batch['image'].cuda()
        targets = batch['label'].cuda()
        optimizer.zero_grad()
        features, logits = model.forward_classifier(images)

        output = criterion(features, targets)
        loss_arcface = nn.CrossEntropyLoss()(output, targets)

        loss_ce = nn.CrossEntropyLoss()(logits, targets)
        loss = loss_arcface + loss_ce

        loss.backward()

        current_lr = get_lr(optimizer)

        optimizer.step()
        losses.append(loss.item())
        losses_arcface.append(loss_arcface.item())
        losses_ce.append(losses_ce)

        writer.add_scalar(f'total_loss',
                          loss.item(),
                          epoch * len(dataloader) + i
                          )

        writer.add_scalar(f'arface_loss',
                          loss_arcface.item(),
                          epoch * len(dataloader) + i)

        writer.add_scalar(f'entropy_loss',
                          loss_ce.item(),
                          epoch * len(dataloader) + i)

        running_avg_loss = np.mean(losses)
        running_avg_arcface_loss = np.mean(losses_arcface)
        # running_avg_ce_loss = np.mean(losses_ce)
        print(f'[Epoch {epoch+1}][Batch {i} / {len(dataloader)}][lr: {current_lr}]: [total_loss: {running_avg_loss}][arcface_loss: {running_avg_arcface_loss}]]')

    average_total_loss = np.mean(losses)
    average_arcface_loss = np.mean(losses_arcface)
    #average_ce_loss = np.mean(losses_ce)

    writer.add_scalar(f'total-loss-epoch',
                      average_total_loss,
                      epoch
                      )

    writer.add_scalar(f'arcface-loss-epoch',
                      average_arcface_loss,
                      epoch
                      )

    # writer.add_scalar(f'ce-loss-epoch',
    #                   average_ce_loss,
    #                   epoch
    #                   )


def triplet_train(model, dataloader, optimizer, criterion, logging_step, epoch, epochs, writer, date_id, output_folder, scheduler):
    losses = []

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
        images = batch['image'].cuda()
        targets = batch['label'].cuda()
        optimizer.zero_grad()

        predictions = model(images)
        loss = criterion(predictions, targets)
        loss.backward()

        current_lr = get_lr(optimizer)

        optimizer.step()
        losses.append(loss.item())

        writer.add_scalar(f'loss',
                          loss.item(),
                          epoch * len(dataloader) + i
                          )

        # if (i % logging_step == 0) & (i > 0):
        running_avg_loss = np.mean(losses)
        print(f'[Epoch {epoch+1}][Batch {i} / {len(dataloader)}][lr: {current_lr}]: loss {running_avg_loss}')

    average_loss = np.mean(losses)
    writer.add_scalar(f'loss-epoch',
                      average_loss,
                      epoch
                      )


if __name__ == "__main__":
    main()
