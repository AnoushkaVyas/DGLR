"""
Function to compute node embeddings for a particular window. GRU is independent for all the regions.
The output of the GAT is feeded to the the GRU to generate node embeddings to pass it to the
second GAT layer. GRU is temporal as it decides between the new node embedding and the output of
the previous time step.

Function initilisation: (infeat,outfeat,hiddenfeat, device,alpha,nheads)
- infeat: Feature dimension for each node
- outfeat: Output feature dimension of the embedding
- hiddenfeat: Feature dimension for the hidden node embedding
- alpha: Negative slope for LeakyReLU in GAT
- nheads: Number of parallel computations in GAT

Function inputs: (A_list, feature_list,index)
- A_list: List of adjacency matrices for all the time steps in a window
- feature_list: List of all the features for all the time steps in a window
- index: Corresponding region for the window

Function output: Node embedding of dimension (No. of nodes *windowsize, outfeat)

"""

import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import numpy as np
from models.gnn.gat_ind import GAT_IND

class DGLR(torch.nn.Module):
    def __init__(self,nodes,infeat,outfeat,hiddenfeat, device,alpha,nheads):
        super().__init__()

        feat = [infeat,hiddenfeat,outfeat]
        self.device = device
        self.GRCU_layers = []
        self.alpha=alpha
        self.nheads=nheads
        self.nodes=nodes

        self._parameters = nn.ParameterList()
        
        for i in range(1,len(feat)):
            if i==len(feat)-1:
                grcu_i = GRCU(self.nodes,feat[i-1]*self.nheads,feat[i],feat[i],self.alpha,self.nheads)
                self.GRCU_layers.append(grcu_i.to(self.device))
                self._parameters.extend(list(self.GRCU_layers[-1].parameters()))
            else:
                grcu_i = GRCU(self.nodes,feat[i-1],feat[i],feat[i]*self.nheads,self.alpha,self.nheads)
                self.GRCU_layers.append(grcu_i.to(self.device))
                self._parameters.extend(list(self.GRCU_layers[-1].parameters()))

    def parameters(self):
        return self._parameters

    def forward(self,A_list, feature_list):
        for i,unit in enumerate(self.GRCU_layers):

            if i==len(self.GRCU_layers)-1:
                feature_list = unit(A_list,feature_list,concat=False)
            else:
                feature_list = unit(A_list,feature_list,concat=True)

        out=torch.cat(feature_list,dim=0)
        return out

class GRCU(torch.nn.Module):
    def __init__(self,nodes,infeat,outfeat,finaloutfeat,alpha,nheads):
        super().__init__()
        self.rows = infeat
        self.cols = outfeat
        self.finaloutfeat=finaloutfeat
        self.alpha=alpha
        self.nheads=nheads
        self.nodes=nodes

        self.evolve_features=[]

        for i in range(self.nodes):
            efeat_i = mat_GRU_cell(self.finaloutfeat)
            self.evolve_features.append(efeat_i)

        self.GAT_init_features=Parameter(torch.Tensor(self.nodes,self.finaloutfeat))
        self.reset_param(self.GAT_init_features)
        self.GAT=GAT_IND(self.rows, self.cols, self.alpha, self.nheads)

    def reset_param(self,t):
        nn.init.xavier_normal_(t.data)

    def forward(self,A_list,node_embs_list,concat):
        GAT_features = self.GAT_init_features
        out_seq = []
        for t,Ahat in enumerate(A_list):
            gat_features=[]
            node_embs = node_embs_list[t]
            node_embs=self.GAT(node_embs,Ahat,concat)

            for i in range(self.nodes):
                gat_features.append(self.evolve_features[i](GAT_features[i],node_embs[i]))
            
            GAT_features=torch.stack(gat_features,dim=0)
            out_seq.append(GAT_features)

        return out_seq

class mat_GRU_cell(torch.nn.Module):
    def __init__(self,Dim):
        super().__init__()
        self.Dim = Dim
        
        self.update = mat_GRU_gate(self.Dim,torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(self.Dim,torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(self.Dim,torch.nn.Tanh())
        
    def forward(self,prev_Q,prev_Z):

        update = self.update(prev_Z,prev_Q)
        reset = self.reset(prev_Z,prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(prev_Z, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q


class mat_GRU_gate(torch.nn.Module):
    def __init__(self,Dim,activation):
        super().__init__()
        self.activation = activation
        self.W = Parameter(torch.Tensor(Dim,Dim))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(Dim,Dim))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(Dim))

    def reset_param(self,t):
        nn.init.xavier_normal_(t.data)
    
    def forward(self,x,hidden):
        out = self.activation(self.W.matmul(x) +self.U.matmul(hidden) + self.bias)
        return out


