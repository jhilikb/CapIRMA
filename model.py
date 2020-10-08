
import torch
import torch.nn as nn
from torch.autograd import Variable

from conv_layer import ConvLayer
from capsule_layer import CapsuleLayer
from decoder import Decoder
import torch.nn.functional as F
from torchvision import models



class Net(nn.Module):

    

    def __init__(self, pc,nets,num_conv_in_channel, num_conv_out_channel, num_primary_unit,
                 primary_unit_size, num_classes, output_unit_size, num_routing,
                 use_reconstruction_loss, regularization_scale, input_width, input_height,
                 cuda_enabled):
       
        super(Net, self).__init__()
	
        self.cuda_enabled = cuda_enabled

        # Configurations used for image reconstruction.
        self.use_reconstruction_loss = use_reconstruction_loss
        self.pc=pc
        self.nets=nets
        
        self.image_width = input_width
        self.image_height = input_height
        self.image_channel = num_conv_in_channel

        
        self.regularization_scale = regularization_scale

        # inception = models.inception_v3(pretrained=True)
        if nets==1: #alexnet
            vgg_pretrained_features = models.alexnet(pretrained=True).features
            modified_pretrained = nn.Sequential(*list(vgg_pretrained_features.children())[:-1])
            num_conv_out_channel1 = 256
            for param in modified_pretrained.parameters():
                param.requires_grad = False

        if nets == 2: #vgg256
            vgg_pretrained_features = models.vgg11(pretrained=True).features
            modified_pretrain = nn.Sequential(*list(vgg_pretrained_features.children())[:-10])
            num_conv_out_channel1 = 256
            for param in modified_pretrain.parameters():
                param.requires_grad = False
            modified_pretrained = nn.Sequential(modified_pretrain,nn.MaxPool2d(2))

        if nets==3:  #vgg512
            vgg_pretrained_features = models.vgg11(pretrained=True).features
            modified_pretrained = nn.Sequential(*list(vgg_pretrained_features.children())[:-1])
            num_conv_out_channel1 = 512
            for param in modified_pretrained.parameters():
                param.requires_grad = False

        if nets==4:  #none
            self.conv1 = ConvLayer(in_channel=num_conv_in_channel, out_channel=num_conv_out_channel, kernel_size=11)
            self.conv2 = ConvLayer(in_channel=num_conv_out_channel, out_channel=num_conv_out_channel, kernel_size=11)
            # #ss=8
            num_conv_out_channel1 = 128
            self.conv3 = ConvLayer(in_channel=num_conv_out_channel,out_channel=num_conv_out_channel1,kernel_size=11)
            modified_pretrained=nn.Sequential(self.conv1, nn.ReLU(), nn.MaxPool2d(2),self.conv2,nn.ReLU(),nn.MaxPool2d(2),self.conv3,nn.ReLU(),nn.MaxPool2d(2))



        self.features = modified_pretrained
        
        self.primary = CapsuleLayer(p=pc,
                                    in_unit=0,
                                    in_channel=num_conv_out_channel1,
                                    num_unit=num_primary_unit,
                                    unit_size=primary_unit_size, # capsule outputs
                                    use_routing=False,
                                    num_routing=num_routing,
                                    cuda_enabled=cuda_enabled)

        
        self.digits = CapsuleLayer(p=pc,
                                   in_unit=num_primary_unit,
                                   in_channel=primary_unit_size,
                                   num_unit=num_classes,
                                   unit_size=output_unit_size, 
                                   use_routing=True,
                                   num_routing=num_routing,
                                   cuda_enabled=cuda_enabled)

        # Reconstruction network
        if use_reconstruction_loss:
            self.decoder = Decoder(num_classes, output_unit_size, input_width,
                                   input_height, num_conv_in_channel, cuda_enabled)

    def forward(self, x):
        """
        Defines the computation performed at every forward pass.
        """
        

        out_conv1 = self.features(x)
        
        out_primary_caps = self.primary(out_conv1)
        
        out_digit_caps = self.digits(out_primary_caps)
        return out_digit_caps

    def loss(self, image, out_digit_caps, target, size_average=True):
        
        recon_loss = 0
        m_loss = self.margin_loss(out_digit_caps, target)
        if size_average:
            m_loss = m_loss.mean()

        total_loss = m_loss

        if self.use_reconstruction_loss:
            # Reconstruct the image from the Decoder network
            reconstruction = self.decoder(out_digit_caps, target)
            recon_loss = self.reconstruction_loss(reconstruction, image)

            # Mean squared error
            if size_average:
                recon_loss = recon_loss.mean()

            # In order to keep in line with the paper,
            # they scale down the reconstruction loss by 0.0005
            # so that it does not dominate the margin loss.
            total_loss = m_loss + recon_loss * self.regularization_scale

        return total_loss, m_loss, (recon_loss * self.regularization_scale)

    def margin_loss(self, input, target):
        
        batch_size = input.size(0)

        # ||vc|| also known as norm.
        v_c = torch.sqrt((input**2).sum(dim=2, keepdim=True))

        # Calculate left and right max() terms.
        zero = Variable(torch.zeros(1))
        if self.cuda_enabled:
            zero = zero.cuda()
        m_plus = 0.9
        m_minus = 0.1
        loss_lambda = 0.5
        max_left = torch.max(m_plus - v_c, zero).view(batch_size, -1)**2
        max_right = torch.max(v_c - m_minus, zero).view(batch_size, -1)**2
        t_c = target
        # Lc is margin loss for each digit of class c
        l_c = t_c * max_left + loss_lambda * (1.0 - t_c) * max_right
        l_c = l_c.sum(dim=1)

        return l_c

    def reconstruction_loss(self, reconstruction, image):
        
        # Calculate reconstruction loss.
        batch_size = image.size(0) # or another way recon_img.size(0)
        # error = (recon_img - image).view(batch_size, -1)
        image = image.view(batch_size, -1) 
        error = reconstruction - image
        squared_error = error**2

        # Scalar Variable
        recon_error = torch.sum(squared_error, dim=1)

        return recon_error
