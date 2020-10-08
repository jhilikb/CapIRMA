"""Capsule layer
Modified the Capsule Implementation of Cedric Chee according to the requirements of the paper. 
Main paper: PyTorch implementation of CapsNet in Sabour, Hinton et al.'s paper
Dynamic Routing Between Capsules. NIPS 2017.
https://arxiv.org/abs/1710.09829

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import utils


class CapsuleLayer(nn.Module):
    """
    The core implementation of the idea of capsules
    """

    def __init__(self,p, in_unit, in_channel, num_unit, unit_size, use_routing,
                 num_routing, cuda_enabled):
        super(CapsuleLayer, self).__init__()

        self.in_unit = in_unit
        self.p=p
        self.in_channel = in_channel
        self.num_unit = num_unit
        self.use_routing = use_routing
        self.num_routing = num_routing
        self.cuda_enabled = cuda_enabled

        if self.use_routing:
            
            self.weight = nn.Parameter(torch.randn(1, in_channel, num_unit, unit_size, in_unit))
        else:
            """
            According to the CapsNet architecture section in the paper,
            we have routing only between two consecutive capsule layers (e.g. PrimaryCapsules and ClassCaps).
            No routing is used between Conv1 and PrimaryCapsules.

            This means PrimaryCapsules is composed of several convolutional units.
            """
            # Define  convolutional units.
            self.conv_units = nn.ModuleList([
                nn.Conv2d(self.in_channel, self.p, 3, 1) for u in range(self.num_unit)
            ])

    def forward(self, x):
        if self.use_routing:
            # Currently used by DigitCaps layer.
            return self.routing(x)
        else:
            # Currently used by PrimaryCaps layer.
            return self.no_routing(x)

    def routing(self, x):
        """
        Routing algorithm for capsule.

        

        :return: vector output of capsule j
        """
        batch_size = x.size(0)

        x = x.transpose(1, 2) 
        x = torch.stack([x] * self.num_unit, dim=2).unsqueeze(4)

        
        batch_weight = torch.cat([self.weight] * batch_size, dim=0)

        
        u_hat = torch.matmul(batch_weight, x)

        
        b_ij = Variable(torch.zeros(1, self.in_channel, self.num_unit, 1))
        if self.cuda_enabled:
            b_ij = b_ij.cuda()

        
        num_iterations = self.num_routing

        for iteration in range(num_iterations):
            
            c_ij = F.softmax(b_ij, dim=2)  # Convert routing logits (b_ij) to softmax.
            
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            # Implement equation 2 in the paper.
            # s_j is total input to a capsule, is a weigthed sum over all "prediction vectors".
            # u_hat is weighted inputs, prediction ˆuj|i made by capsule i.
           
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            # Squash the vector output of capsule j.
            # v_j shape: [batch_size, weighted sum of PrimaryCaps output,
            #             num_classes, output_unit_size from u_hat, 1]
            
            v_j = utils.squash(s_j, dim=3)

            
            v_j1 = torch.cat([v_j] * self.in_channel, dim=1)

            # The agreement.
            
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)

            # Update routing (b_ij) by adding the agreement to the initial logit.
            b_ij = b_ij + u_vj1

        return v_j.squeeze(1) # shape: [128, 10, 16, 1]

    def no_routing(self, x):
        """
        Get output for each unit.
        A unit has batch, channels, height, width.
        

        :return: vector output of capsule j
        """
        # Create  convolutional unit.
        # A convolutional unit uses normal convolutional layer with a non-linearity (squash).
        unit = [self.conv_units[i](x) for i, l in enumerate(self.conv_units)]

        # Stack all unit outputs.
        
        unit = torch.stack(unit, dim=1)

        batch_size = x.size(0)

        
        unit = unit.view(batch_size, self.num_unit, -1)

        # Add non-linearity
        
        return utils.squash(unit, dim=2) 
