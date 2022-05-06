"""
Function for performing GAT operation on node features.

Function initialisation: (in_features, out_features, alpha, concat)
- in_features: Feature dimension for each node
- out_features: Output feature dimension of the embedding
- alpha: Negative slope for LeakyReLU in GAT
- concat: Whether to concatenate the features coming from nheads computations

Feature inputs: (input,adj)
- input: Input node embeddings or input features
- adj: Corresponding adjacency matrix

Function output: Node embedding of dimension (No. of nodes,out_features)

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GraphAttentionLayer_Ind(nn.Module):

    def __init__(self, in_features, out_features, alpha, concat=True):
        super(GraphAttentionLayer_Ind, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))

        self.reset_param(self.W)
        self.reset_param(self.a)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def reset_param(self,t):
        nn.init.kaiming_normal_(t.data)

    def forward(self,input, adj):
        h = torch.mm(input, self.W)

        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        zero_vec = -9e15*torch.ones_like(e)
        att = torch.where(adj > 0, e*adj, zero_vec)
        attention = F.softmax(att, dim=1)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
