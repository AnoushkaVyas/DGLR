"""
Code File for:
- Train-Test division
- Loading dataset
- Making normalized adjacency matrix 

"""

import numpy as np
import torch
import csv
import os
import networkx as nx
from scipy.linalg import fractional_matrix_power
from sklearn import preprocessing

def load_data_usa(dataset="USA"):
    """Load USA dataset"""
    print('Loading {} dataset for 2018 and 2019...'.format(dataset))

    #train-test division
    train_set=518
    test_set=212

    # build adjacency
    path_for_edge="./Datasets/USA/graph/edges.txt"

    adj=np.zeros((68,68))
    f=open(path_for_edge, "r")
    f1=f.readlines()

    for line in f1:
        i=int(line.rstrip().split(',')[0])
        j=int(line.rstrip().split(',')[1])
        if adj[i][j]==0 or adj[j][i]==0:
            adj[i][j]=1
            adj[j][i]=1

    # normalize adjacency matrix
    adj = adj + np.eye(adj.shape[0])

    G = nx.from_numpy_matrix(adj)
    L=nx.laplacian_matrix(G).toarray()
    D=L+adj

    frac=fractional_matrix_power(D,-1)
    adj=frac @ adj 
    adj =  torch.FloatTensor(adj)

    #features and ID
    path_for_feature="./Datasets/USA/features/"
    features=[]
    AList=[]

    for files in sorted(os.listdir(path_for_feature)):
        reader = csv.reader(open(path_for_feature+ str(files), "r"), delimiter=",")
        x = list(reader)
        idx_features = np.array(x)
        f=idx_features[1:,2:].astype(float)
        f=preprocessing.normalize(f,norm='l2')
        f = torch.FloatTensor(f)
        features.append(f)
        AList.append(adj)

    #soilmoisture
    path_for_SM="./Datasets/USA/SM/"
    SM=[]

    for files in sorted(os.listdir(path_for_SM)):
        reader = csv.reader(open(path_for_SM+str(files), "r"), delimiter=",")
        x = list(reader)
        SoilMoisturedata = np.array(x)
        sm=SoilMoisturedata[1:,-1].astype(float)
        sm = torch.FloatTensor(sm)
        SM.append(sm)

    return train_set ,test_set ,features, SM, AList

