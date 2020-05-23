from __future__ import print_function
import argparse
"""import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn as nn 
from torch.utils.data import DataLoader
#from warpctc_pytorch import CTCLoss
import os
import utils
from torchvision import transforms"""
from model import CRNN
from dataset import ImageLoader
from train import weights_init, trainBatch, val
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchsize', type=int, help='input batch size', default=2)
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nhidden', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--crnn_checkpoints', default='', help="path to crnn (to continue training)")
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--checkpoints_folder', default=None, help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=3, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=3, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=3, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=3, help='Interval to be displayed')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')

opt=parser.parse_args()

print(opt)

nclass = len(opt.alphabet)+1
nc = 1
crnn = CRNN(imgH = 32, nc = 1, nclass = nclass, nh = opt.nhidden, n_rnn=2, leakyRelu=False)







