"""
Implementation for a temporal graph representation learning based solution by taking different 
input features as NDVI, SAR coefficients, Weather values to predict (both estimating missing values and forecasting) 
soil moisture which can leverage both spatial and temporal dependency of soil moisture values. Graph Reconstruction module is added
along with the soil moisture prediction module with feature and ground truth smoothness regularisation.

"""

from __future__ import division
from __future__ import print_function

import time
import argparse
import glob
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Importing function to load data
from utils_usa import load_data_usa
from utils_spain import load_data_spain
from utils_mississippi import load_data_mississippi
from utils_alabama import load_data_alabama

#Importing GAT based models
from models.dglr.dglr import DGLR

#Importing Regression model
from models.regression.Regression_ind import FC_IND

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='USA', help='Dataset to be used.')

parser.add_argument('--nofstations', type=int, default=68,help='Number of nodes in the graph.')

parser.add_argument('--noftimes', type=int, default=10,help='Number of times to run the experiment.')

parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')

parser.add_argument('--epochs', type=int, default=20,help='Number of epochs to train.')   

parser.add_argument('--lr', type=float, default=0.01,help='Initial learning rate.')

parser.add_argument('--hidden', type=int, default=20,help='Number of hidden units.')

parser.add_argument('--out', type=int, default=20,help='Number of output units.')

parser.add_argument('--window', type=int, default=4,help='Window Length.')

parser.add_argument('--stride', type=int, default=1,help='Stride for the Window.')

parser.add_argument('--nheads', type=int, default=3, help='Number of head attentions.')

parser.add_argument('--alpha', type=float, default=0.1, help='Alpha for the leaky_relu.')

parser.add_argument('--negslope', type=float, default=0.1, help='Negative slope for the leaky_relu of regression.')

args = parser.parse_args()
args.cuda = False

# Load data
if args.dataset == "USA":
    train_set ,test_set ,  features, SM, AList = load_data_usa()

if args.dataset == "Spain":
    train_set ,test_set ,  features, SM, AList = load_data_spain()

if args.dataset == "Mississippi":
    train_set ,test_set ,  features, SM, AList,date = load_data_mississippi()

if args.dataset == "Alabama":
    train_set ,test_set ,  features, SM, AList,date = load_data_alabama()

#RMSE
def error(actual, predicted):
    return actual - predicted

def mse(actual, predicted):
    return np.mean(np.square(error(actual, predicted)))

def rmse(actual, predicted):
    return np.sqrt(mse(actual, predicted))

#SMAPE
def smape(A, F):
    return (100 * np.sum(np.abs(F - A) / (np.abs(A) + np.abs(F))))/len(A)

#Correlation
def correlation(a,b):
    corr=stats.pearsonr(a,b)[0]
    return corr

#Average over iterations
rmse_std=np.zeros((args.noftimes,args.nofstations))
corr_std=np.zeros((args.noftimes,args.nofstations))
smape_std=np.zeros((args.noftimes,args.nofstations))

for itr in range(args.noftimes):

    # Model and optimizer
    model_egnn = DGLR(nodes=args.nofstations,infeat=features[0].shape[1],outfeat=args.out,hiddenfeat=args.hidden,device="cuda" if args.cuda else "cpu",alpha=args.alpha,nheads=args.nheads)
    model_regression = FC_IND(nodes=args.nofstations,ntime=args.window,nfeatures=args.out,negslope=args.negslope,device="cuda" if args.cuda else "cpu")

    optimizer_egnn = optim.Adam(model_egnn.parameters(),lr=args.lr)

    optimizer_regression = optim.Adam(model_regression.parameters(),lr=args.lr)

    #Results arrays
    groundtruth=np.zeros((args.nofstations,train_set+test_set-args.window))
    prediction=np.zeros((args.nofstations,train_set+test_set-args.window))
    trainloss=[]
    predictionloss=[]
    graphrefineloss=[]
    targetsmoothnessloss=[]
    featuresmoothnessloss=[]

    # Train model
    gamma=2
    beta=2000
    phi=1
    delta=2000

    t_total = time.time()
    A=AList

    mse_loss=nn.MSELoss()
    crossentropyloss=nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        loss_train=0
        loss_pred=0
        loss_graph_refine=0
        loss_target_smoothness=0
        loss_feature_smoothness=0
        Ait = torch.zeros(size=(train_set,args.nofstations,args.out))
        
        for i in range(args.nofstations):
            model_egnn.train()
            model_regression.train()

            optimizer_egnn.zero_grad()
            optimizer_regression.zero_grad()

            print('Epoch ',epoch,' ID ', i)

            loss_wd=0
            loss_refine_wd=0
            loss_feature=0
            loss_sm_smoothness=0

            for wd in range(0,train_set-args.window,args.stride):
                
                train_AList=A[wd:wd+args.window+1]
                train_data=features[wd:wd+args.window+1]
                target_SM=torch.FloatTensor([SM[wd+args.window][i]])

                nodeemb = model_egnn(train_AList, train_data)
                predict_SM=model_regression(nodeemb,i)

                recons_adj_id=torch.cat(([F.relu(nodeemb[j:j+args.nofstations] @ nodeemb[j:j+args.nofstations].T)[i]/torch.sum(F.relu(nodeemb[j:j+args.nofstations] @ nodeemb[j:j+args.nofstations].T)[i])  for j in range(0,args.nofstations*(args.window+1),args.nofstations)]),dim=0)
                target_adj_id=torch.cat(([AList[j][i] for j in range(wd,wd+args.window+1)]),dim=0)

                diff_x=torch.cat(([torch.square(torch.norm(features[j]-features[j][i],dim=1)) for j in range(wd,wd+args.window+1)]),dim=0)
                diff_sm=torch.cat(([torch.square(SM[j]-SM[j][i]) for j in range(wd,wd+args.window+1)]),dim=0)
                
                loss_wd=loss_wd + mse_loss(predict_SM, target_SM)
                loss_refine_wd=loss_refine_wd+crossentropyloss(recons_adj_id,target_adj_id)
                loss_feature=loss_feature+torch.dot(recons_adj_id,diff_x)
                loss_sm_smoothness=loss_sm_smoothness+torch.dot(recons_adj_id,diff_sm)

                print("Target: ",target_SM)
                print("Prediction: ",predict_SM)
                print('------')

                if epoch==args.epochs-1:
                    prediction[i][wd]=predict_SM
                    groundtruth[i][wd]=target_SM 

                Ait[wd:wd+args.window+1,i,:]=torch.stack(([nodeemb[j:j+args.nofstations][i].clone().detach() for j in range(0,args.nofstations*(args.window+1),args.nofstations)]),dim=0)

            print('ID Loss ',loss_wd)
            print('Graph Refine ID Loss ',loss_refine_wd)
            print('Feature Smoothness Loss', loss_feature)
            print('Soil Moisture Smoothness Loss', loss_sm_smoothness)

            loss_train=loss_train+loss_wd.item()+loss_refine_wd.item()+loss_feature.item()+loss_sm_smoothness.item()
            loss_pred=loss_pred+loss_wd.item()
            loss_graph_refine=loss_graph_refine+loss_refine_wd.item()
            loss_target_smoothness=loss_target_smoothness+loss_sm_smoothness.item()
            loss_feature_smoothness=loss_feature_smoothness+loss_feature.item()
            
            loss_total= gamma* loss_wd+ delta* loss_refine_wd + beta*loss_feature + phi*loss_sm_smoothness
            loss_total.backward()

            optimizer_egnn.step()
            optimizer_regression.step()   
            print('------------------')

        A=[ F.relu(Ait[j] @ Ait[j].T)/torch.sum(F.relu(Ait[j] @ Ait[j].T),dim=1) for j in range(train_set)] 
            
        trainloss.append(loss_train)
        predictionloss.append(loss_pred)
        graphrefineloss.append(loss_graph_refine)
        targetsmoothnessloss.append(loss_target_smoothness)
        featuresmoothnessloss.append(loss_feature_smoothness)

        print('Training Loss ',loss_train)
        print('Prediction Training Loss ',loss_pred)
        print('Graph Refine Training Loss ',loss_graph_refine)
        print('Target Smoothness Training Loss ',loss_target_smoothness)
        print('Feature Smoothness Training Loss ',loss_feature_smoothness)

        print('================')

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print('=========')

    # Test
    for i in range(args.nofstations):
        for wd in range(train_set-args.window,train_set+test_set-args.window,args.stride):

            test_AList=[A[train_set-1]]* args.window
            test_data=features[wd:wd+args.window]
            target_SM=torch.FloatTensor([SM[wd+args.window][i]])

            model_egnn.eval()
            model_regression.eval()

            nodeemb = model_egnn(test_AList, test_data)
            predict_SM=model_regression(nodeemb,i)

            loss_test = mse_loss(predict_SM, target_SM)

            print("Test set results:","loss= {:.4f}".format(loss_test.item()))

            print('For ID ',i)
            print("Target: ",target_SM)
            print("Prediction: ",predict_SM)
            print('------')
            
            prediction[i][wd]=predict_SM
            groundtruth[i][wd]=target_SM

        rmse_std[itr][i]=rmse(groundtruth[i][train_set-args.window:train_set+test_set-args.window], prediction[i][train_set-args.window:train_set+test_set-args.window])
        corr_std[itr][i]=correlation(groundtruth[i][train_set-args.window:train_set+test_set-args.window], prediction[i][train_set-args.window:train_set+test_set-args.window])
        smape_std[itr][i]=smape(groundtruth[i][train_set-args.window:train_set+test_set-args.window], prediction[i][train_set-args.window:train_set+test_set-args.window])


mean_rmse_itr=np.mean(rmse_std,axis=0)
mean_smape_itr=np.mean(smape_std,axis=0)
mean_corr_itr=np.mean(corr_std,axis=0)
std_rmse_itr=np.std(rmse_std,axis=0)
std_smape_itr=np.std(smape_std,axis=0)
std_corr_itr=np.std(corr_std,axis=0)

print('Average Test RMSE: ',np.mean(mean_rmse_itr),'+/-',np.mean(std_rmse_itr))
print('Average Test Correlation: ',np.mean(mean_corr_itr),'+/-',np.mean(std_corr_itr))
print('Average Test SMAPE: ',np.mean(mean_smape_itr),'+/-',np.mean(std_smape_itr))
