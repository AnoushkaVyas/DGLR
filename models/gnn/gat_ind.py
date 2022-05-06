"""
Function for performing GAT operation on node features. This function calls GraphAttentionLayer_Ind 
to calculate the node features. 

Function initialisation: (infeat, outfeat, alpha, nheads)
- infeat: Feature dimension for each node
- outfeat: Output feature dimension of the embedding
- alpha: Negative slope for LeakyReLU in GAT
- nheads: Number of parallel computations in GAT

Function inputs: (x, adj,concat)
- x: Input node embeddings or input features
- adj: Corresponding adjacency matrix
- concat: Whether to perform concatenation of the features from the nheads 

Function output: Node embedding of dimension (No. of nodes,outfeat)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gnn.ind_layer import GraphAttentionLayer_Ind


class GAT_IND(nn.Module):
    def __init__(self, infeat, outfeat, alpha, nheads):
        super(GAT_IND, self).__init__()
        self.outfeat=outfeat
        self.nheads=nheads

        self.attentions = [GraphAttentionLayer_Ind(infeat, outfeat, alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = [GraphAttentionLayer_Ind(infeat, outfeat, alpha, concat=False) for _ in range(nheads)]
        for i, attention in enumerate(self.out_att):
            self.add_module('attention_{}'.format(i), attention)
        

    def forward(self, x, adj,concat):
        if concat:
            out = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        else:
            H=[]
            for att in self.out_att:
                H.append(att(x,adj))

            x=torch.stack(H).sum(dim=0)/self.nheads
            out=F.elu(x)

        return out



