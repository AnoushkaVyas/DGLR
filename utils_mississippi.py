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

def load_data_mississippi(dataset="Mississippi"):
    """Load Mississippi dataset"""
    print('Loading {} dataset for 2017...'.format(dataset))

    #train-test division
    train_set=22
    test_set=6

    # build adjacency
    path_for_edge="./Datasets/Mississippi/graph/edges.txt"

    adj=np.zeros((5,5))
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
    path_for_feature="./Datasets/Mississippi/features/"
    features=[]
    AList=[]
    date=[]

    for files in sorted(os.listdir(path_for_feature)):
        reader = csv.reader(open(path_for_feature+ str(files), "r"), delimiter=",")
        x = list(reader)
        idx_features = np.array(x)
        f=idx_features[1:,3:].astype(float)
        date.append(idx_features[1:,2][0])
        f=preprocessing.normalize(f,norm='l2')
        f = torch.FloatTensor(f)
        features.append(f)
        AList.append(adj)
        idx=idx_features[1:,0].astype(int)
        name=idx_features[1:,1]

    print("Region: ",name)
    print("ID: ",idx)

    #soilmoisture
    path_for_SM="./Datasets/Mississippi/groundtruth_SM/"
    SM=[]

    for files in sorted(os.listdir(path_for_SM)):
        reader = csv.reader(open(path_for_SM+str(files), "r"), delimiter=",")
        x = list(reader)
        SoilMoisturedata = np.array(x)
        sm=SoilMoisturedata[1:,-1].astype(float)
        sm = torch.FloatTensor(sm)
        SM.append(sm)

    return train_set ,test_set ,features, SM, AList,date


