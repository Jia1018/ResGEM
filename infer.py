import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch_geometric.loader import DataLoader, DataListLoader
import torch_geometric.transforms as T
from torch_geometric.nn import DataParallel
import time

from Dataloader import InferDataset_full_model
from models import ResGEM
from train_eval import infer_full_model

def parse_arguments():
    parser = argparse.ArgumentParser()
    # naming / file handling
    parser.add_argument('--testset', type=str, default='./data/test', help='testing set file name')
    parser.add_argument('--save_dir', type=str, default='./data/results', help='')
    # training parameters
    parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--device_idx', type=int, default=0)
    parser.add_argument('--manualSeed', type=int, default=3627473, help='manual seed')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--decay_step', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--data_type', type=str, default='Synthetic', help='')
    parser.add_argument("--nfeat", type=int, default=7, help="Dimension of input feature.")
    parser.add_argument('--graph_type', type=str, default='DENSE_FACE', help='graph node type, DENSE_FACE or VERTEX')
    parser.add_argument('--max_patch_size', type=int, default=1000, help='')
    parser.add_argument('--min_patch_size', type=int, default=1000, help='')
    parser.add_argument('--max_neighbor_num', type=int, default=25, help='')
    parser.add_argument('--max_one_ring_num', type=int, default=15, help='max number of faces in the one ring area of a vertex')
    parser.add_argument('--neighbor_sizes', type=list, default=[8, 16, 32], help='max number of faces in the one ring area of a vertex')
    parser.add_argument('--local_patch_size', type=int, default=100, help='')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    parser.add_argument('--model_interval', type=int, default=5, metavar='N', help='how many batches to wait before logging training status')

    # others parameters
    parser.add_argument('--resume', type=str, default='', help='refine model at this path')

    return parser.parse_args()

args = parse_arguments()

device = torch.device('cuda', args.device_idx)
print(device)
torch.set_num_threads(args.workers)

model = ResGEM(args.nfeat, 3, args.neighbor_sizes, args.heads).to(device)
print(model)
device_ids=[7]
model = DataParallel(model, device_ids=device_ids)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      args.decay_step,
                                      gamma=args.lr_decay)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=device)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))



infer_list = os.listdir(os.path.join(args.testset, args.data_type))
print(infer_list)
avg_time = 0.
for model_name in infer_list:
    print(model_name)
    t = time.time()
    infer_dataset = InferDataset_full_model('./', args, model_name, 38000)
    infer_loader = DataListLoader(infer_dataset, batch_size=1)
    infer_full_model(args, model, model_name, infer_loader, 'ResGCN_dense', 33)
    t_duration = time.time() - t
    print("Duration:", t_duration)
    avg_time += t_duration
print("Average time:", avg_time/len(infer_list))