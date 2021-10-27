# -*- coding: utf-8 -*-
"""
Least square fit using trained stress-strain NN metamodel
for the Ogden model

"""

import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import pandas as pd
import pickle
import csv
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
import imageio
from time import time
from random import seed, randint
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import io
from lmfit import Parameters, minimize, report_fit, fit_report
from scipy import interpolate

import Metamodels
import PreProcess

def MetaModelExport(a, b, x, model, scaler):
            
    # NN device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Transform to acceptable model input
    X_temp = np.atleast_2d([a,b])
    X_scaled = scaler.transform(X_temp)
    X_tensor = torch.from_numpy(X_scaled).float().to(device)
    
    fX_star  = model(X_tensor)
    
    FxFz = fX_star.data.numpy()
    
    Fx =  FxFz[0,0:50]
    Fz = FxFz[0,50:100]
    
    #Interpolate model export
    FEBioStrain = np.linspace(-0.5,0.5,50)
    Fx_interp = interpolate.interp1d(FEBioStrain, Fx, kind='cubic')
    Fz_interp = interpolate.interp1d(FEBioStrain, Fz, kind='cubic')
    
    Fx_new = Fx_interp(x) 
    Fz_new = Fz_interp(x)
    
    MDL_data = np.column_stack((Fx_new, Fz_new))
    
    return MDL_data

# Objective function
def objective(params, x, exp_data, model, scaler):
    """Calculate total residual for fits of Gaussians to several data sets."""
    # Export parameters
    a = params['a'] #apply scaling to have meaningful results
    b = params['b']*1e4
    
    # Intialize
    resid = 0.0*data[:]
        
    mdl_dat = MetaModelExport(a, b, x, model, scaler)
  
    # make residual per data set
    for i in range(2):
        resid[:,i] = exp_data[:,i] - mdl_dat[:,i]

    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()


if __name__ == "__main__":
    
    # Input (experimental data)
    exp_data_points = 100
    conc_list = [10,20,40]
    coagtime_list = [60,90,120]
    replica_list = ['x', 'y', 'z']
    
    # Input (trained model data)
    Num_train = 8000
    DirName = './ExportedData/FWD_OgdenNN_Final/Output_OgdenNN_Final'

    Xscaler_dir = DirName + str(Num_train) + '/Torch_NN_' + str(Num_train) + 'Xscaler.sav'
    Settings_dir = DirName + str(Num_train) + '/Torch_NN_' + str(Num_train) + 'Settings.sav'
    Model_path = DirName + str(Num_train) + '/Torch_NN_' + str(Num_train) + '.pt'
    
    NN_settings = pickle.load(open(Settings_dir, 'rb'))
    scaler = pickle.load(open(Xscaler_dir, 'rb'))
    NUM_FEATURES = NN_settings['Feat']
    HIDDEN_DIM = NN_settings['Dim']
    NUM_OUTPUT = NN_settings['Out']
    
    model = Metamodels.NN(feature_dim=NUM_FEATURES, hidden_dim=HIDDEN_DIM, output_dim=NUM_OUTPUT)
    model.load_state_dict(torch.load(Model_path))
    model.eval()
    
    # Read Experimental Data
    for i,conc in enumerate(conc_list):
        for ii,coagtime in enumerate(coagtime_list):
            for iii, replica in enumerate(replica_list):
                
                strain, Fx, Fz = PreProcess.load_Ogden_ExpData(exp_data_points,conc,coagtime,replica)
                data = np.asarray([Fx, Fz]).T
                x = strain
                    
                # Initialize parameters to be fitted
                fit_params = Parameters()
                fit_params.add('a', value=0.001, min=0.0, max=100)
                fit_params.add('b', value=0.0010, min=0.0, max=10)
                
                # Run matlab lsqnonlin equivalent    
                out = minimize(objective, fit_params, method = 'least_squares', args=(x, data, model, scaler))
                
                # Inverse scale final outputs
                af = out.params['a']
                bf = out.params['b']*1e4
                final_params = np.array([af,bf])
                best_fit = MetaModelExport(af, bf, x, model, scaler)
                # Figure
                plt.subplots(2,1,figsize=(12,16))
                plt.subplot(2, 1, 1)
                plt.plot(x, data[:,0], 'o', label='Experimental Data' ,color = "orange", lw=3)
                plt.plot(x, best_fit[:,0], 'g-', label='Best fit', lw=3)
                plt.legend(loc = 'best', fontsize = 16)
                
                plt.subplot(2, 1, 2)
                plt.plot(x, data[:,1], 'o',  color = "orange", lw=3)
                plt.plot(x, best_fit[:,1], 'g-', lw=3)
                                
                print(fit_report(out))
                
                if not os.path.exists('./ExportedData/OgdenParams'):
                    os.makedirs('./ExportedData/OgdenParams')
                param_path = './ExportedData/OgdenParams/NN_C' + str(conc) +'_T'+str(coagtime)+ '_'+ replica + '.csv'
                img_path = './ExportedData/OgdenParams/NN_C' + str(conc) +'_T'+str(coagtime)+ '_'+ replica + '.png'
                
                np.savetxt(param_path, final_params, delimiter=",")
                plt.savefig(img_path, bbox_inches='tight')