# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 14:26:12 2021

@author: Sotiris
"""
import os
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import imageio
from random import seed
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score


##############################################################################
# NN network
##############################################################################  
class NN(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim, depth = None, Ogden_BWD = False, Holz_BWD = False):
        
        super(NN, self).__init__()
        self.depth = depth
        self.Holz_BWD = Holz_BWD
        self.Ogden_BWD = Ogden_BWD
        
        if self.depth is None:
			# Default depth = 3 if DEPTH not defined
            self.fc1 = nn.Linear(feature_dim,hidden_dim)
            #self.drop1 = nn.Dropout(0.1)
            self.fc2 = nn.Linear(hidden_dim,hidden_dim)
            self.fc3 = nn.Linear(hidden_dim,hidden_dim)
            self.fc4 = nn.Linear(hidden_dim,output_dim)
        else:
            self.layers = nn.ModuleDict() # a collection that will hold your layers
    
            self.layers['input'] = nn.Linear(feature_dim, hidden_dim)
            for i in range(1, depth):
                self.layers['hidden_'+str(i)] = torch.nn.Linear(hidden_dim, hidden_dim)
    
            self.layers['output'] = torch.nn.Linear(hidden_dim, output_dim)
        
        
        # Define sigmoid layer
        if self.Holz_BWD or self.Ogden_BWD:
            self.sigmoid = nn.Sigmoid()
        
    
    def forward(self, x):
        
        if self.depth is None:
            x = F.elu(self.fc1(x))
            #x = self.drop1(x)
            x = F.elu(self.fc2(x))
            x = F.elu(self.fc3(x))
            x = self.fc4(x)
            
        else:
                                
            for layer in self.layers:
                
                if layer not in set(['output']):
                    
                    x = F.elu(self.layers[layer](x))
            
            x = self.layers['output'](x)  
        
        
        if self.Holz_BWD:
            # scale_tensor = torch.zeros(8, 8)
            # scale_tensor.fill_diagonal_(1e2)
            
            # af and afs where sampled one order lower than the others
            # Max limits output for a's: 50kPa and for b's: 30
            scale_tensor = torch.diag(torch.FloatTensor([500,30,50,30,50,30,500,30]))
            x = torch.matmul(self.sigmoid(x), scale_tensor)
            
        if self.Ogden_BWD:
            # scale_tensor = torch.zeros(2, 2)
            # scale_tensor.fill_diagonal_(1e2)
            scale_tensor = torch.diag(torch.FloatTensor([50,30]))
            x = torch.matmul(self.sigmoid(x), scale_tensor)
        
        
        return x
    

    
def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x, y):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat = model(x)
        # Computes loss
        loss = loss_fn(y, yhat)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()
    
    # Returns the function that will be called inside the train loop
    return train_step
    

def scaled_to_tensor(device, Xscaler, X, Y, XY_BatchSize, Yscaler = None, X_noise = None):
    
    X_scaled = Xscaler.transform(X)
    
    if X_noise is not None:
        X_scaled = X_scaled + X_noise
        
    if Yscaler is not None:
        Y = Yscaler.transform(Y)
        
    X_tensor = torch.from_numpy(X_scaled).float().to(device)
    Y_tensor = torch.from_numpy(Y).float().to(device)
    data = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset=data, batch_size = XY_BatchSize)  
    
    return X_tensor, data, loader

def convert_to_ref(TestData, n_val, fX_val):
    
    if TestData.isshear:
        fX_val_ref = np.array([]).reshape(0, 100)
        for i in range(0,n_val):
            
            sam_range = i*50
            Fx = fX_val[sam_range:sam_range+50,0]
            Fz = fX_val[sam_range:sam_range+50,1]
            
            temp = np.hstack((Fx,Fz))
            fX_val_ref = np.vstack((fX_val_ref,temp))
            
    else:
        
        fX_val_ref = np.array([]).reshape(0, 50)
        for i in range(0,n_val):
            
            sam_range = i*50
            Fz = fX_val[sam_range:sam_range+50,0]
                        
            fX_val_ref = np.vstack((fX_val_ref,Fz))
      
    
    return fX_val_ref

class CustomLoss(torch.nn.Module):
    
    def __init__(self, alpha):
        super(CustomLoss,self).__init__()
        self.alpha = alpha #regularization for monotonicity constraint
        
    def MonotonicityLoss(self,yhat):
        
        yhat_np = yhat.data.numpy()
        yhat_pd = pd.DataFrame(yhat_np[:,0:50])
        y_cummax = yhat_pd.cummax(axis=1)
        
        y_dif = abs(y_cummax - yhat_pd)
        
        # Find elements that are not satisfying cummax(Yk) == Yk
        col_count = y_dif[y_dif > 1e-15].count()
        total_count = col_count.to_numpy().sum()
        
        return total_count
    
    
    def forward(self,yhat,y):
        abs_tensor = torch.abs(yhat - y)
        L1 = torch.mean(abs_tensor)
        L2 = self.MonotonicityLoss(yhat)
        
        totLoss = L1 + self.alpha*L2
        return totLoss
    
    
class R2Loss(torch.nn.Module):
    
    def __init__(self):
        super(R2Loss,self).__init__()
       
    
    def forward(self,yhat,y):
        
        denom = torch.var(y,1, unbiased=False)
        MSE_each =   F.mse_loss(yhat, y, reduction="none") 
        MSE_sample = torch.sum(MSE_each,1)
        # Normalize error per element
        NormError = torch.div(MSE_sample,denom)
        # Sum errors
        return torch.sum(NormError)

class HolzScaler:
    
    def __init__(self):
        
        a = 1e3 #convert to kPa
        b = 1 #leave exponent as is
        
        self.scale_mat = np.diag([a*10,b,a,b,a,b,a*10,b]) # af and afs where sampled one order lower than the others
        self.inv_mat = np.linalg.inv(self.scale_mat)
        
    def transform(self, Y_inp):
        
        Y_scaled = np.matmul(Y_inp,self.scale_mat)
        
        return Y_scaled
    
    def inverse_transform(self, Y_scaled):
        
        Y_inp = np.matmul(Y_scaled, self.inv_mat)
        
        return Y_inp
        


class OgdenScaler:
    
    def __init__(self):
        
        a = 1e3 #convert to kPa
        b = 1 #leave exponent as is
        
        self.scale_mat = np.diag([a,b])
        self.inv_mat = np.linalg.inv(self.scale_mat)
        
    def transform(self, Y_inp):
        
        Y_scaled = np.matmul(Y_inp,self.scale_mat)
        
        return Y_scaled
    
    def inverse_transform(self, Y_scaled):
        
        Y_inp = np.matmul(Y_scaled, self.inv_mat)
        
        return Y_inp
        
#########################################################
# Active learning GPR

def GP_regression_std(regressor, X, n_max=1):
    
    _, std = regressor.predict(X, return_std=True)
    
    query_idx = (-std).argsort()[:n_max]
    
    return query_idx, X[query_idx,:]

def NN_regression_R2(regressor, X, Y_true, n_max):
    # NN prediction
    Y_pred = regressor.predict(X)
    # Calculate R2 error
    R2_Raws = r2_score(Y_true.T, Y_pred.T, multioutput= 'raw_values')
    
    query_idx = (R2_Raws).argsort()[:n_max]
    
    return query_idx, X[query_idx,:]