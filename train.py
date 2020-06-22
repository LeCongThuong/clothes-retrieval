import os
import sys
import shutil
import argparse
from datetime import datetime

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

    image_paths_list = args.image_paths_listls
    labels_file = args.labels_file
    writer = get_summary_writer(args)
    data_transform = augmentation(args.image_size, train=True)

    time_id, output_folder = log_experience(args, data_transform)
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
    arcface_loss = ArcMarginProduct(args.embedding_dim, num_classes)
    arcface_loss.to(device)
    start_epoch = args.start_epoch

    # load checkpoint if args.weights parameter is not None
    if args.weights is not None:
        print('loading pre-trained weights and changing input size ...')
        weights = torch.load(args.weights)
        start_epoch = weights['epoch']
        if 'state_dict' in weights.keys():
            weights = weights['state_dict']

        if args.archi != 'densenet121':
            if bool(args.pop_fc):
                weights.pop('model.fc.weight')
                weights.pop('model.fc.bias')
            try:
                weights.pop('model.classifier.weight')
                weights.pop('model.classifier.bias')
            except:
                print('no classifier. skipping.')

        model.load_state_dict(weights, strict=False)

    model.to(device)

    # define optimizer
    my_list = ['fc1.weight', 'fc1.bias']
    params = list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam([{'params': parameters},
                      {'params': arcface_loss.parameters()}],
                     lr=args.lr,
                     weight_decay=args.wd)

    # define scheduler
    scheduler = get_scheduler(args, optimizer)
    if args.weights is not None:
        weights = torch.load(args.weights)
        print('loading pre-trained scheduler and optimizer state ...')
        optimizer.load_state_dict(weights['optimizer'])
        scheduler.load_state_dict(weights['scheduler'])

    dataset = ClothesDataset(paths=paths, mapping_label_id=mapping_label_id, transform=data_transform)

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=args.num_workers)

    model.train()

    # start training loop
    for epoch in tqdm(range(start_epoch, args.epochs + start_epoch)):
        params = {
            'model': model,
            'dataloader': dataloader,
            'optimizer': optimizer,
            'criterion': arcface_loss,
            'logging_step': args.logging_step,
            'epoch': epoch,
            'epochs': args.epochs,
            'writer': writer,
            'time_id': time_id,
            'scheduler': scheduler,
            'output_folder': output_folder
        }
        _ = train(**params)

        scheduler.step()

    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scheduler': scheduler.state_dict()

    }
    torch.save(state,
               os.path.join(output_folder,
                            f'{time_id}.pth'))

    # compute_predictions(args,
    #                     data_files,
    #                     model,
    #                     mapping_label_id,
    #                     mapping_pseudo_files_folders,
    #                     time_id,
    #                     output_folder)


def train(model, dataloader, optimizer, criterion, logging_step, epoch, epochs, writer, time_id, output_folder, scheduler):
    current_lr = get_lr(optimizer)
    losses = []
    min_loss = 1000000000

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), leave=False):
        images = batch['image'].cuda()
        targets = batch['label'].cuda()

        features, logits = model.forward_classifier(images)

        output = criterion(features, targets)
        loss_arcface = nn.CrossEntropyLoss()(output, targets)

        loss_ce = nn.CrossEntropyLoss()(logits, targets)
        loss = loss_arcface + loss_ce

        optimizer.zero_grad()
        loss.backward()

        current_lr = get_lr(optimizer)

        optimizer.step()
        losses.append(loss.item())

        writer.add_scalar(f'loss',
                          loss.item(),
                          epoch * len(dataloader) + i
                          )

        if (i % logging_step == 0) & (i > 0):
            running_avg_loss = np.mean(losses)
            print(f'[Epoch {epoch+1}][Batch {i} / {len(dataloader)}][lr: {current_lr}]: loss {running_avg_loss}')

    average_loss = np.mean(losses)
    writer.add_scalar(f'loss-epoch',
                      average_loss,
                      epoch
                      )


   # if (args.checkpoint_period != -1) & (args.checkpoint_period % (epoch+1) == 0):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scheduler': scheduler.state_dict()

    }

    if average_loss < min_loss:
        min_loss = average_loss
        torch.save(state,
                   os.path.join(output_folder,
                                f'model_best.pth'))

    torch.save(state,
               os.path.join(output_folder,
                            f'{datetime.now()}.pth'))

    return average_loss


if __name__ == "__main__":
    main()
