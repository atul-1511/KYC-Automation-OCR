from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn as nn 
from torch.utils.data import DataLoader
#from warpctc_pytorch import CTCLoss
import os
import utils
from torchvision import transforms
from model import CRNN
from dataset import ImageLoader
from train import weights_init, trainBatch, val
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
#parser.add_argument('--trainroot', required=True, help='path to dataset')
#parser.add_argument('--valroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nhidden', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--crnn_checkpoints', default='', help="path to crnn (to continue training)")
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz. ')
parser.add_argument('--checkpoints_folder', default=None, help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=3, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=3, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=3, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=3, help='Interval to be displayed')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')

opt = parser.parse_args()

print (opt)

#loading other parameters of opt
nclass = len(opt.alphabet) + 1
nc = 1

#adding model
crnn = CRNN(imgH = 32, nc = 1, nclass = nclass, nh = opt.nhidden, n_rnn=2, leakyRelu=False)

#loading model weights
if opt.crnn_checkpoints != '' and (os.path.exists(opt.crnn_checkpoints)) :
    try :
        crnn.load_state_dict(torch.load(opt.crnn_checkpoints))
        print('loaded pretrained model from %s' % opt.crnn_checkpoints)
    except :
        crnn.apply(weights_init)
        print ('Corrupt checkpoint given. Training from scratch')


if opt.checkpoints_folder == None:
    opt.checkpoints_folder = 'model_checkpoints'

print (opt.checkpoints_folder)

if not (os.path.exists(opt.checkpoints_folder)) :
    os.system('mkdir {0}'.format(opt.checkpoints_folder))

print ('Model checkpoints directory created')

##checking for GPU if available
cudnn.benchmark = True

batch_size = opt.batchSize

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#Loading training and test data
traindataset = ImageLoader(image_folder = 'samples_generated', label_csv = 'sample_text.csv', \
    transform =transforms.Compose([transforms.Resize((32,100)), transforms.ToTensor()]))

assert traindataset

train_dataloader = DataLoader(traindataset, batch_size=batch_size, \
    shuffle=True, num_workers=0,drop_last = True)

testdataset = ImageLoader(image_folder = 'samples_generated', label_csv = 'sample_text.csv', \
    transform =transforms.Compose([transforms.Resize((32,100)), transforms.ToTensor()]))

assert testdataset

test_dataloader = DataLoader(testdataset, batch_size=batch_size, num_workers=0, drop_last = True)

converter = utils.strLabelConverter(opt.alphabet)
#criterion = CTCLoss()
criterion = nn.MSELoss()

#If GPU usage is allowed
if opt.cuda:
    crnn.cuda()
    crnn = nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    criterion = criterion.cuda()


# loss averager
loss_avg = utils.averager()

#creating optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)


for epoch in range(opt.niter):
    train_iter = iter(train_dataloader)
    i = 0
    train_iters = len(train_dataloader)
    print ('No. of epoch: '+str(epoch))
    while i < train_iters:
        data = train_iter.next()
        cost = trainBatch(crnn, criterion, optimizer,data, converter, use_cuda = opt.cuda)
        loss_avg.add(cost)
        i += 1

        if i % opt.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, opt.niter, i, len(train_dataloader), loss_avg.val()))
            loss_avg.reset()

        if i % opt.valInterval == 0:
            val(crnn, test_dataloader, criterion, converter, opt.n_test_disp, opt.batchSize, use_cuda = opt.cuda)

        # do checkpointing
        if i % opt.saveInterval == 0:
            #save_checkpoint()
            torch.save(crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(opt.checkpoints_folder, epoch, i))
