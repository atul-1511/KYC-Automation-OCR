import torch
from torch.autograd import Variable
import utils
#from warpctc_pytorch import CTCLoss
import warnings
warnings.filterwarnings("ignore")

# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        


def val(net, data_loader, criterion,converter, n_test_disp,batchSize, max_iter=100,  use_cuda = False):
    print('Start validation')

    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data['image'], data['label']
        image = Variable(cpu_images)
        if use_cuda :
            image = image.cuda()
        text_1 = Variable(torch.randn((25,2, 39)))
        batch_size = cpu_images.size(0)
        text, length = converter.encode(cpu_texts)
        text, length = Variable(text), Variable(length)
        preds = net(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        #cost = criterion(preds, text, preds_size, length) / batch_size
        cost = criterion(input = preds, target = text_1)
        loss_avg.add(cost)

        _, preds = preds.max(2) #returns values and indices of max elements in preds
        preds = preds.unsqueeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False) #decoding predictions
        for pred, target in zip(sim_preds, cpu_texts) :
            if pred == target.lower() :
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def trainBatch(net, criterion, optimizer,data, converter, use_cuda = False):

    for p in net.parameters():
        p.requires_grad = True
    
    net.train()

    cpu_images, cpu_texts = data['image'], data['label']
    batch_size = cpu_images.size(0)
    image = Variable(cpu_images)

    if use_cuda :
        image = image.cuda()

    text, length = converter.encode(cpu_texts)
    text, length = Variable(text), Variable(length)
    #print (image.size())

    preds = net(image)
    #print (preds.size(), text.size())
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

    #cost = criterion(preds, text, preds_size, length) / batch_size
    text_1 = Variable(torch.randn((25,2, 39)))
    cost = criterion(input = preds, target = text_1)
    net.zero_grad()
    cost.backward()
    optimizer.step()
    
    return cost


