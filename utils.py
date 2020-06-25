import argparse
from configparser import ConfigParser
import math
import os
import shutil
import json
import simplejson
from datetime import datetime
from pprint import pprint
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloader import ClothesDataset, augmentation
from warmup_scheduler import GradualWarmupScheduler
from samplers import pk_sampler, pk_sample_full_coverage_epoch


# utility function to parse the arguments from the command line

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_paths_list', type=str, default='/content/clothes-retrieval/metadata/data_train_list_file.txt')
    parser.add_argument('--labels_file', type=str, default='/content/clothes-retrieval/metadata/labels_without_one_image_items.txt')
    parser.add_argument('--eval_paths_file', type=str, default='/content/clothes-retrieval/metadata/data_eval_list_file.txt')
    parser.add_argument('--label_to_samples_file', type=str, default='/content/clothes-retrieval/metadata/label_to_samples.txt')
    parser.add_argument('--archi', default='densenet121',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnext',
                                 'densenet121',
                                 'mobilenet',
                                 'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'], type=str)
    parser.add_argument('--embedding-dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--alpha', type=int, default=8)
    parser.add_argument('--pretrained', type=int, choices=[0, 1], default=0)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--loss', default='arcface', choices=['arcface', 'triplet'])

    parser.add_argument('--margin', type=float, default=-1)
    parser.add_argument('-p', type=int, default=16)
    parser.add_argument('-k', type=int, default=4)
    parser.add_argument('--sampler', type=int, default=2, choices=[1, 2])

    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--fs_lr', type=float, default=6e-4)
    parser.add_argument('--wd', type=float, default=0.00001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=12)

    parser.add_argument('--logging-step', type=int, default=20)

    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--pop-fc', type=int, default=1)
    parser.add_argument('--log_path', type=str, default='/content/clothes-retrieval/training_output/')
    parser.add_argument('--tag', type=str, default='')

    parser.add_argument('--checkpoint-period', type=int, default=2)

    parser.add_argument('--scheduler', type=str,
                        choices=['multistep', 'cosine', 'warmup'], default='warmup')
    parser.add_argument('--step-size', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--milestones', nargs='+', type=int)
    parser.add_argument('--lr-end', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=2)

    args = parser.parse_args()
    return args


# function to get a tensorboard summary writer

def get_summary_writer(args):
    now = datetime.now()
    date_id = now.strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(args.log_path, args.tag + '----' + date_id)
    os.makedirs(logdir)
    writer = SummaryWriter(logdir)
    return writer, date_id


# function to log parameters
def log_experience(args, date_id, data_transform):
    data_transform_repr = data_transform.indented_repr()
    arguments = vars(args)
    arguments['date'] = date_id
    arguments['leaderboard_score'] = None
    arguments['tag'] = args.tag
    arguments['_augmentation'] = data_transform_repr

    print('logging these arguments for the experience ...')
    pprint(arguments)
    print('----')

    output_folder = f'{args.log_path}/{args.tag}-model-and-parameters-{date_id}/'
    os.makedirs(output_folder)

    with open(os.path.join(output_folder, f'{date_id}.json'), 'w') as f:
        f.write(simplejson.dumps(simplejson.loads(
            json.dumps(arguments)), indent=4, sort_keys=True))
    return output_folder


# utility function to convert an image to a square while keeping its aspect ratio

def expand2square(pil_img):
    background_color = (0, 0, 0)
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


# utlity function to get and set learning rates in optimizer

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# utility function to define learning rate schedulers

def get_scheduler(args, optimizer):
    args = vars(args)

    if args['scheduler'] == "warmup":
        print(f'Using warmup scheduler with cosine annealing')
        print(
            f"warmup epochs : {args['warmup_epochs']} | total epochs {args['epochs']}")
        print(f"lr_start : {args['lr']} ---> lr_end : {args['lr_end']}")

        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                      args['epochs'],
                                                                      eta_min=args['lr_end'])
        scheduler = GradualWarmupScheduler(optimizer,
                                           multiplier=1,
                                           total_epoch=args['warmup_epochs'],
                                           after_scheduler=scheduler_cosine)

    elif args['scheduler'] == "multistep":
        print(
            f"Using multistep scheduler with gamma = {args['gamma']} and milestones = {args['milestones']}")

        scheduler = MultiStepLR(optimizer,
                                milestones=args['milestones'],
                                gamma=args['gamma'])

    elif args['scheduler'] == "cosine":
        print(f"Using cosine annealing from {args['lr']} to {args['lr_end']}")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               args['epochs'],
                                                               eta_min=args['lr_end'])

    return scheduler

# utility function to define Dataset Samplers


def get_sampler(args,
                dataset,
                classes,
                labels_to_samples,
                mapping_files_to_global_id):

    args = vars(args)
    if args['sampler'] == 1:
        sampler = pk_sampler.PKSampler(data_source=dataset,
                                       classes=classes,
                                       labels_to_samples=labels_to_samples,
                                       mapping_files_to_global_id=mapping_files_to_global_id,
                                       p=args['p'],
                                       k=args['k'])

    elif args['sampler'] == 2:
        sampler = pk_sample_full_coverage_epoch.PKSampler(data_source=dataset,
                                                          classes=classes,
                                                          labels_to_samples=labels_to_samples,
                                                          mapping_files_to_global_id=mapping_files_to_global_id,
                                                          p=args['p'],
                                                          k=args['k'])

    return sampler


# find top n images that have same gave category
def find_n_images_same_cate(given_cate: str, paths: list, sorted_index_list: list, n=20):
    sorted_paths = [paths[i] for i in sorted_index_list]
    n_same_cate_images = []
    n_image_types = []
    total_images_collected = 0

    for path in reversed(sorted_paths):
        image_cate = path.split('/')[-1].split('~')[2:4]
        image_type = path.split('/')[-1].split('~')[-4]
        if image_cate == given_cate:
            n_same_cate_images.append(path)
            n_image_types.append(image_type)
            total_images_collected += 1
            if total_images_collected >= n:
                return n_same_cate_images, n_image_types

    return n_same_cate_images, n_image_types


def calculate_similarity(eval_path, paths, embeddings, test_embedding):
    eval_cate = eval_path.split('/')[-1].split('~')[2:4]
    # eval_type = eval_path.split('/')[-1].split('~')[-4]
    cate_path_index = []
    for index, path in enumerate(paths):
        path_cate = path.split('/')[-1].split('~')[2:4]
        if eval_cate == path_cate:
            cate_path_index.append(index)
    cate_embeddings = embeddings[cate_path_index, :]
    return cate_path_index, cosine_similarity(test_embedding[np.newaxis, :], cate_embeddings)


def compute_predictions(args, model, paths: list, eval_paths: list, mapping_label_id, date_id, writer:SummaryWriter, output_folder, epoch):
    model.eval()
    print("generating predictions ......")

    data_transform_test = augmentation(args.image_size, train=False)
    scoring_dataset = ClothesDataset(paths,
                                     mapping_label_id,
                                     data_transform_test,
                                     test=True)

    scoring_dataloader = DataLoader(scoring_dataset,
                                    shuffle=False,
                                    num_workers=11,
                                    batch_size=args.batch_size)

    embeddings = []
    for batch in tqdm(scoring_dataloader, total=len(scoring_dataloader)):
        with torch.no_grad():
            embedding = model(batch['image'].cuda())
            embedding = embedding.cpu().detach().numpy()
            embeddings.append(embedding)
    embeddings = np.concatenate(embeddings)

    # np.save(os.path.join(output_folder,
    #                      f'embeddings_{date_id}.npy'),
    #         embeddings)

    test_dataset = ClothesDataset(eval_paths,
                                  mapping_label_id,
                                  data_transform_test,
                                  test=True)

    test_dataloader = DataLoader(test_dataset,
                                 num_workers=11,
                                 shuffle=False,
                                 batch_size=args.batch_size)


    test_embeddings = []
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        with torch.no_grad():
            test_embedding = model(batch['image'].cuda())
            test_embedding = test_embedding.cpu().detach().numpy()
            test_embeddings.append(test_embedding)
    test_embeddings = np.concatenate(test_embeddings)

    # np.save(os.path.join(output_folder,
    #                      f'test_embeddings_{date_id}.npy'),
    #         test_embeddings)

    # eval_labels = [eval_path.split('/')[-1].split('~')[-4] for eval_path in eval_paths]
    # eval_label_indexes = np.array([mapping_label_id[eval_label] for eval_label in eval_labels])
    # dataset_labels = [path.split('/')[-1].split('~')[-4] for path in paths]
    # dataset_label_indexes = np.array([mapping_label_id[dataset_label] for dataset_label in dataset_labels])

    # dataset_index_matrix = dataset_label_indexes[np.newaxis, :] + np.zeros((len(eval_paths), 1))

    # csm = cosine_similarity(test_embeddings, embeddings)
    # sorted_index = np.argsort(csm)
    # sorted_res = np.array(list(map(lambda x, y: y[x], sorted_index, dataset_index_matrix)))
    acc_top_1 = 0
    acc_top_5 = 0
    acc_top_10 = 0
    acc_top_20 = 0

    # list of list
    n_same_cate_images_total = []
    for i, eval_path in enumerate(eval_paths):
        # eval_cate = eval_path.split('/')[-1].split('~')[2:4]
        eval_type = eval_path.split('/')[-1].split('~')[-4]
        cate_path_index, eval_path_csm = calculate_similarity(eval_path, paths, embeddings, test_embeddings[i])
        new_paths = [paths[index] for index in cate_path_index]
        sorted_eval_path_csm_index = np.argsort(eval_path_csm)
        n_same_cate_images = [new_paths[index] for index in reversed(sorted_eval_path_csm_index[0])][:20]
        n_image_types = [image_name.split('/')[-1].split('~')[-4] for image_name in n_same_cate_images]
        n_same_cate_images_total.append(n_same_cate_images)
        if eval_type == n_image_types[0]:
            acc_top_1 += 1
        if eval_type in n_image_types[: 5]:
            acc_top_5 += 1
        if eval_type in n_image_types[:10]:
            acc_top_10 += 1
        if eval_type in n_image_types[:20]:
            acc_top_20 += 1

    print("---------------------------------------------")
    print("acc_top_1: ", acc_top_1 / len(eval_paths))
    print("acc_top_5: ", acc_top_5 / len(eval_paths))
    print("acc_top_10: ", acc_top_10 / len(eval_paths))
    print("acc_top_20: ", acc_top_20 / len(eval_paths))
    print("---------------------------------------------")
    print("predictions generated...")
    writer.add_scalar(f'accuracy_top_1',
                      acc_top_1,
                      epoch
                      )

    writer.add_scalar(f'accuracy_top_5',
                      acc_top_5,
                      epoch
                      )

    writer.add_scalar(f'accuracy_top_10',
                      acc_top_10,
                      epoch
                      )

    writer.add_scalar(f'accuracy_top_20',
                      acc_top_20,
                      epoch
                      )
    # sorted array according
    for i, same_cate_images in enumerate(n_same_cate_images_total[:10]):
        fig = plt.figure(figsize=(24, 48))
        query_image = eval_paths[i]
        # image_result_index = sorted_index[i, :]
        # sorted_paths = []
        # for value in image_result_index:
        #     sorted_paths.append(paths[value])
        # image_result = sorted_paths[-10:]

        images_show = [query_image]
        for image_re in same_cate_images:
            images_show.append(image_re)
        for idx in np.arange(11):
            ax = fig.add_subplot(1, 11, idx + 1, xticks=[], yticks=[])
            image = mpimg.imread(images_show[idx])
            plt.imshow(image)
        writer.add_figure("Query_{}".format(i), fig, global_step=epoch)




