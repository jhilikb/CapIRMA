

from __future__ import print_function
import argparse
from timeit import default_timer as timer
import os
import sys
import torch
import torch.optim as optim
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision.utils as vutils
# from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from loaddata import DLibdata
from torch.utils.data import DataLoader

from torchvision import transforms, datasets


from skimage import io, transform
from matplotlib import pyplot as pl
from PIL import Image
import utils
from model import Net
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
# counn=sys.argv[2]
# counn='1516'
countrn=12677-46
countst=1733-94

cu=1
kwargs = {'num_workers': 4, 'pin_memory': True} if cu else {}

cl=57
print('loading train now')
training_set = DLibdata(train=True)
train_loader = DataLoader(training_set, batch_size=6, shuffle=True, **kwargs)
print('loading test')
testing_set = DLibdata(train=False)
test_loader = DataLoader(testing_set, batch_size=6, shuffle=True, **kwargs)
features = torch.FloatTensor(countst, 57, 16, 1)
orilab=torch.IntTensor(countst)
predlab = torch.IntTensor(countst)
fileind=torch.IntTensor(countst)
matches= torch.FloatTensor(countst, 57,1,1)
y_true=torch.IntTensor(countst,cl)
y_scores=torch.IntTensor(countst,cl)

def train(model, optimizer, epoch):
    
    print('===> Training mode')


    num_batches = len(train_loader) # iteration per epoch. e.g: 469
    total_step = args.epochs * num_batches
    epoch_tot_acc = 0

    # Switch to train mode
    model.train()

    if args.cuda:
        # When we wrap a Module in DataParallel for multi-GPUs
        model = model.module

    start_time = timer()
    n=0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, unit='batch')):
        batch_size = data.size(0)
        global_step = batch_idx + (epoch * num_batches) - num_batches

        labels = target
        target_one_hot = utils.one_hot_encode(target, length=args.num_classes)
        assert target_one_hot.size() == torch.Size([batch_size, args.num_classes])

        data, target = Variable(data), Variable(target_one_hot)
        # print(data)

        if args.cuda:
            data = data.cuda()
            target = target.cuda()

        # Train step - forward, backward and optimize
        optimizer.zero_grad()
        output = model(data) # output from DigitCaps (out_digit_caps)
        loss, margin_loss, recon_loss = model.loss(data, output, target)
        loss.backward()
        optimizer.step()
        # if epoch==25:
        #     features[n:n + batch_size, :, :, :] = output.detach().clone().cpu()
        #
        #     fileind[n:n + batch_size] = ind.clone()
        #     n = n + batch_size


        # Calculate accuracy for each step and average accuracy for each epoch
        acc = utils.accuracy(args,output, labels, args.cuda)
        epoch_tot_acc += acc
        epoch_avg_acc = epoch_tot_acc / (batch_idx + 1)

        

    # Print time elapsed for an epoch
    # end_time = timer()
    # print('Time elapsed for epoch {}: {:.0f}s.'.format(epoch, end_time - start_time))
    # torch.save(features, 'features_trn.pt')
    # # print(features[1])
    # torch.save(fileind, 'fileindtrn.pt')

    
    

def test(model, num_train_batches, epoch):
    
    print('===> Evaluate mode')

    # Switch to evaluate mode
    model.eval()

    if args.cuda:
        # When we wrap a Module in DataParallel for multi-GPUs
        model = model.module

    loss = 0
    margin_loss = 0
    recon_loss = 0

    correct = 0
    countst = 1733 - 94
    num_batches = len(test_loader)

    global_step = epoch * num_train_batches + num_train_batches
    n=0
    # matches= torch.FloatTensor(countst, 57,1,1)
    # orilab=torch.FloatTensor(countst)
    # predlab = torch.FloatTensor(countst)
    # fileind=torch.FloatTensor(countst)
    b=1
    for data, target,ind in test_loader:
        print(b)
        b=b+1
        batch_size = data.size(0)
        target_indices = target
        target_one_hot = utils.one_hot_encode(target_indices, length=args.num_classes)
        y_true[n:n + batch_size, :] = target_one_hot.clone()
        assert target_one_hot.size() == torch.Size([batch_size, args.num_classes])

        with torch.no_grad():
            data, target= Variable(data, volatile=True), Variable(target_one_hot)


        if args.cuda:
            data = data.cuda()
            target = target.cuda()

        # Output predictions
        output = model(data) # output from DigitCaps (out_digit_caps)

        # Sum up batch loss
        with torch.no_grad():
            t_loss, m_loss, r_loss = model.loss(data, output, target, size_average=False)
        loss += t_loss.data[0]
        margin_loss += m_loss.data[0]
        recon_loss += r_loss.data[0]
        
        # Count number of correct predictions
        # v_magnitude shape: [128, 10, 1, 1]
        features[n:n + batch_size, :, :, :] = output.detach().clone().cpu()
        with torch.no_grad():
            v_magnitude = torch.sqrt((output**2).sum(dim=2, keepdim=True))
        # print(v_magnitude.data)
        # pred shape: [128, 1, 1, 1]
            pred = v_magnitude.data.max(1, keepdim=True)[1].cpu()
        # for loop in range(pred.shape[0]):
        #     print(target_indices[loop],pred[loop])

        #
        #     fileind[n:n + batch_size] = ind.clone()
        #     n = n + batch_size
        if epoch>0:
            matches[n:n + batch_size, :,:,:] =  v_magnitude.data.detach().clone()
            # print(target_indices)
            orilab[n:n+batch_size]=target_indices.clone()
            # y_true[n:n + batch_size, :] = target.data.clone()
            predlab[n:n + batch_size] = pred[:,0,0,0].clone()
            fileind[n:n + batch_size] = ind.clone()
            n = n + batch_size
            # print(y_true[7])

        correct += pred.eq(target_indices.view_as(pred)).sum()

    # Get the reconstructed images of the last batch
    if args.use_reconstruction_loss:
        reconstruction = model.decoder(output, target)
        # Input image size and number of channel.
        # By default, for MNIST, the image width and height is 28x28 and 1 channel for black/white.
        image_width = args.input_width
        image_height = args.input_height
        image_channel = args.num_conv_in_channel
        recon_img = reconstruction.view(-1, image_channel, image_width, image_height)
        assert recon_img.size() == torch.Size([batch_size, image_channel, image_width, image_height])

        # Save the image into file system
        # utils.save_image(recon_img, 'results/recons_image_test_{}_{}.png'.format(epoch, global_step))
        # utils.save_image(data, 'results/original_image_test_{}_{}.png'.format(epoch, global_step))

        # Add and visualize the image in TensorBoard
        # recon_img = vutils.make_grid(recon_img.data, normalize=True, scale_each=True)
        # original_img = vutils.make_grid(data.data, normalize=True, scale_each=True)
        # writer.add_image('test/recons-image-{}-{}'.format(epoch, global_step), recon_img, global_step)
        # writer.add_image('test/original-image-{}-{}'.format(epoch, global_step), original_img, global_step)

    # Log test losses
    loss /= num_batches
    margin_loss /= num_batches
    recon_loss /= num_batches

    # Log test accuracies
    num_test_data = len(test_loader.dataset)
    accuracy = float(correct) /float(num_test_data)
    accuracy_percentage = 100. * accuracy

    

    
    y_scores = torch.squeeze(matches)
    
    torch.save(matches, 'match323.pt')
    torch.save(orilab, 'orilab323.pt')
    torch.save(predlab, 'predlab323.pt')
    torch.save(fileind, 'fileind323.pt')
    torch.save(features, 'features_tst.pt')


def main():
    """The main function
    Entry point.
    """
    global args


    # Setting the hyper parameters
    parser = argparse.ArgumentParser(description='Example of Capsule Network')
    parser.add_argument('--epochs', type=int, default=15,
                        help='number of training epochs. default=20')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate. default=0.01')
    parser.add_argument('--batch-size', type=int, default=6,
                        help='training batch size. default=128')
    parser.add_argument('--test-batch-size', type=int,
                        default=6, help='testing batch size. default=128')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status. default=10')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training. default=false')
    parser.add_argument('--threads', type=int, default=4,
                        help='number of threads for data loader to use. default=4')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for training. default=42')
    parser.add_argument('--num-conv-out-channel', type=int, default=64,
                        help='number of channels produced by the convolution. default=128')
    parser.add_argument('--file', type=str, default=" ",
                       help='filename. default= ')
    parser.add_argument('--num-conv-in-channel', type=int, default=3,
                        help='number of input channels to the convolution. default=64')
    parser.add_argument('--num-primary-unit', type=int, default=8,
                        help='number of primary unit. default=8')
    parser.add_argument('--primary-unit-size', type=int,
                        default=3872, help='primary unit size is 32 * 6 * 6. default=1152,18432')
    parser.add_argument('--num-classes', type=int, default=57,
                        help='number of classes. 1 unit for one class . default=57')
    parser.add_argument('--output-unit-size', type=int,
                        default=16, help='output unit size. default=16')
    parser.add_argument('--num-routing', type=int,
                        default=3, help='number of routing iteration. default=3')
    parser.add_argument('--pc', type=int,
                        default=32, help='number of prim channels. default=32')
    parser.add_argument('--use-reconstruction-loss', type=utils.str2bool, nargs='?', default=True,
                        help='use an additional reconstruction loss. default=True')
    parser.add_argument('--regularization-scale', type=float, default=0.0005,
                        help='regularization coefficient for reconstruction loss. default=0.0005')
    parser.add_argument('--dataset', help='the name of dataset (mnist, cifar10,irma)', default='b')
    parser.add_argument('--input-width', type=int,
                        default=220, help='input image width to the convolution. ')
    parser.add_argument('--input-height', type=int,
                        default=220, help='input image height to the convolution. ')
    parser.add_argument('--nets', type=int,
                        default='3', help='net number')
    # parser.add_argument('--fnt', type=str,
    #                     default='23', help='file number')

    args = parser.parse_args()

    #print(args)
    # counn1=args.fnt
    # counn=args.fn
    #print('******************************************************' + counn + '*****************************************************************************')    # Check GPU or CUDA is available
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Get reproducible results by manually seed the random number generator
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    # kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    #
    # print('loading train now')
    # training_set = DLibdata(train=True)
    # train_loader = DataLoader(training_set, batch_size=64, shuffle=True, **kwargs)
    # print('loading test')
    # testing_set = DLibdata(train=False)
    # test_loader = DataLoader(testing_set, batch_size=64, shuffle=True, **kwargs)

    # Build Capsule Network
    #print('===> Building model')
    model = Net(pc=args.pc,
                nets=args.nets,
                num_conv_in_channel=args.num_conv_in_channel,
                num_conv_out_channel=args.num_conv_out_channel,num_primary_unit=args.num_primary_unit,
                primary_unit_size=args.primary_unit_size,
                num_classes=args.num_classes,
                output_unit_size=args.output_unit_size,
                num_routing=args.num_routing,
                use_reconstruction_loss=args.use_reconstruction_loss,
                regularization_scale=args.regularization_scale,
                input_width=args.input_width,
                input_height=args.input_height,
                cuda_enabled=args.cuda)
    


    if args.cuda:
        print('Utilize GPUs for computation')
        print('Number of GPU available', torch.cuda.device_count())
        model.cuda()
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model)

    #pretr_dict = torch.load('/media/jhilik/DATA/irma/code/capsule/results/trained_model/model_epoch_15_32_3.pth')
    
    #pp = pretr_dict['state_dict']

    #model.load_state_dict(pp)
    

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Make model checkpoint directory
    if not os.path.exists('results/trained_model'):
        os.makedirs('results/trained_model')

    # Set the logger
    # writer = SummaryWriter()

    # Train and test
    #epoch=30
    start_time = timer()
    #test(model, len(train_loader), 30)
    end_time = timer()
    print('Time elapsed for epoch {}: {:.0f}s.'.format(epoch, end_time - start_time))
    for epoch in range(1, args.epochs + 1):
         train(model,  optimizer, epoch)
         test(model, len(train_loader), epoch)
         # testsin(model,args.file)
    #
    #     # Save model checkpoint
         utils.checkpoint({'epoch': epoch + 1,'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}, epoch)

    # writer.close()
    print(classification_report(orilab, predlab))
    print(precision_recall_fscore_support(orilab, predlab, average='weighted'))
    print(roc_auc_score(y_true, y_scores, average='weighted'))


if __name__ == "__main__":
    main()
