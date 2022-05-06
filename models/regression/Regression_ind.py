"""

Function to calculate predicted soil moisture. After receiving the node embeddings, the embeddings for a particular node is extracted
for all the time steps and passed through a fully connected layer to predict the soil moisture value. Each region has a FC layer for itself.

Function initialisation: (ntime,nfeatures,negslope)
-ntime: Number of time steps before prediction
-nfeatures: Input feature dimensions for FC or the output feature dimensions of node embeddings
-negslope: Negative slope for the LeakyReLU 

Function inputs: (node_emb,ID)
- node_embd: Input node embeddings
- ID: Region for which soil moisture is to be predicted

Function output: Predicted soil moisture value

"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np

class FC_IND(nn.Module):
    def __init__(self, nodes,ntime,nfeatures,negslope,device):
        super(FC_IND, self).__init__()

        self.window=ntime
        self.nodes=nodes
        self.negslope=negslope
        self.device=device
        self.fclist=[]

        for i in range(self.nodes):
            self.fclist.append(nn.Linear(ntime*nfeatures,1).to(self.device))
            self.reset_param(self.fclist[i].weight) 

        self.fclist=nn.ModuleList(self.fclist)

    def reset_param(self,t):
        nn.init.kaiming_normal_(t.data)

    def forward(self, node_emb,ID):

        node_emb_id=[]
        for i in range(0,self.nodes*self.window,self.nodes):
            node_emb_id.append(node_emb[i+ID])
    
        x=torch.cat((node_emb_id),dim=0)
        out = F.leaky_relu(self.fclist[ID](x),negative_slope=self.negslope)
        return out