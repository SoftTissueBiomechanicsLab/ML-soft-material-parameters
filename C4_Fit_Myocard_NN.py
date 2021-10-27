# -*- coding: utf-8 -*-
"""
Least square fit using trained stress-strain NN metamodel
for the Holzapfel model

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
import PostProcess

def MetaModelExport(scaled_params, strain, fiber_param, WDH_param, models, scalers):
            
    # NN device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize output
    n_data_points = len(strain[1])
    MDL_data = np.array([]).reshape(n_data_points,0)
    
    # Loop through testing modes
    for i in range(1,10):
        
        IsShear = (i%3) != 0
        
        # Export cube dimensions for current position
        current_pos = ((i-1)//3)*3
        WDH_dim = WDH_param[current_pos:current_pos+3]
        
        # Assemble X_input
        X_input = np.atleast_2d(np.concatenate((scaled_params,WDH_dim,fiber_param)))
        
        
        # Scale input
        X_scaled = scalers[i].transform(X_input)
        
        # Convert to tensor
        X_tensor = torch.from_numpy(X_scaled).float().to(device)
        
        # Run model
        fX  = models[i](X_tensor)
        fX_np = fX.data.numpy()
        
        Tdir = fX_np[0,0:50]
        Tnor = fX_np[0,50:100] #if applicable
        
        # Interpolate model export
        if IsShear:
            FEBioStrain = np.linspace(-0.4,0.4,50)
            Fx_interp = interpolate.interp1d(FEBioStrain, Tdir, kind='cubic')
            Fz_interp = interpolate.interp1d(FEBioStrain, Tnor, kind='cubic')
            
            # Interpolate the strains of interest
            Fx_new = Fx_interp(strain[i]) 
            Fz_new = Fz_interp(strain[i])
            
            # Assemble the global vector
            MDL_data = np.column_stack((MDL_data,Fx_new))
            MDL_data = np.column_stack((MDL_data,Fz_new))
            
        else:
            FEBioStrain = np.linspace(-0.15,0.15,50)
            Fx_interp = interpolate.interp1d(FEBioStrain, Tdir, kind='cubic')
            
            Fx_new = Fx_interp(strain[i])
            
            # Assemble the global vector
            MDL_data = np.column_stack((MDL_data,Fx_new))
    
    return MDL_data

# Objective function
def objective(params, strain, exp_data, fiber_param, WDH_param, models, scalers):
    """Calculate total residual for fits of Gaussians to several data sets."""
    # Export and scale parameters
    scaled_params = [0]*8
    scaled_params[0] = params['a'].value
    scaled_params[1] = params['b']*1e4
    scaled_params[2] = params['af'].value
    scaled_params[3] = params['bf']*1e4
    scaled_params[4] = params['ash'].value
    scaled_params[5] = params['bsh']*1e4
    scaled_params[6] = params['afs'].value
    scaled_params[7] = params['bfs']*1e4
    
    # Intialize
    resid = 0.0*exp_data[:]
        
    mdl_dat = MetaModelExport(scaled_params, strain, fiber_param, WDH_param, models, scalers)
  
    # make residual per data set
    for i in range(15):
        resid[:,i] = exp_data[:,i] - mdl_dat[:,i]

    # now flatten this to a 1D array, as minimize() needs
    return resid.flatten()


if __name__ == "__main__":
    
    # Input (experimental data)
    TSLs = [1,2,3,4,5,7,8,9,10,11,12]
    n_data_points = 100 # points per stress strain curve
    
    
    # Load trained NNs
    Num_train = 9600
    DirName = './ExportedData/FWD_HolzapfelNN_Final/Output_HolzapfelFinal2_'
    
    # initialize scalers and models
    scalers = {x: [] for x in range(1,10)}
    models = {x: [] for x in range(1,10)}
    
    for i in range(1,10):
            
        Xscaler_dir = DirName + str(Num_train)+'Mode_' + str(i) + '/Torch_NN_' + str(Num_train) + 'Xscaler.sav'
        Settings_dir = DirName + str(Num_train)+'Mode_' + str(i) + '/Torch_NN_' + str(Num_train) + 'Settings.sav'
        Model_path = DirName + str(Num_train)+'Mode_' + str(i) + '/Torch_NN_' + str(Num_train) + '.pt'
        
        NN_settings = pickle.load(open(Settings_dir, 'rb'))
        NUM_FEATURES = NN_settings['Feat']
        HIDDEN_DIM = NN_settings['Dim']
        NUM_OUTPUT = NN_settings['Out']
        DEPTH = 4
        # import scalers and models
        scalers[i] = pickle.load(open(Xscaler_dir, 'rb'))
        models[i] = Metamodels.NN(feature_dim=NUM_FEATURES, hidden_dim=HIDDEN_DIM, output_dim=NUM_OUTPUT, depth = DEPTH)
        models[i].load_state_dict(torch.load(Model_path))
        models[i].eval()
    
    for TSLn in TSLs:
        print('\n Subject TSL: %i' %(TSLn))
        
        # Read experimental data and parameters        
        strain, exp_data, fiber_param, WDH_param  = PreProcess.load_RV_ExpData(n_data_points,TSLn)
           
        # Initialize parameters to be fitted
        fit_params = Parameters()
        fit_params.add('a', value=0.001, min=0.0, max=1)
        fit_params.add('b', value=0.0010, min=0.0, max=0.1)
        fit_params.add('af', value=0.001, min=0.0, max=1)
        fit_params.add('bf', value=0.0010, min=0.0, max=0.1)
        fit_params.add('ash', value=0.001, min=0.0, max=1)
        fit_params.add('bsh', value=0.0010, min=0.0, max=0.1)
        fit_params.add('afs', value=0.001, min=0.0, max=1)
        fit_params.add('bfs', value=0.0010, min=0.0, max=0.1)
    
        # Run matlab lsqnonlin equivalent    
        out = minimize(objective, fit_params, method = 'least_squares', args=(strain, exp_data, fiber_param, WDH_param, models, scalers))
    
        # Inverse scale final outputs
        final_params = [0]*8
        final_params[0] = out.params['a'].value
        final_params[1] = out.params['b']*1e4
        final_params[2] = out.params['af'].value
        final_params[3] = out.params['bf']*1e4
        final_params[4] = out.params['ash'].value
        final_params[5] = out.params['bsh']*1e4
        final_params[6] = out.params['afs'].value
        final_params[7] = out.params['bfs']*1e4
         
        best_fit = MetaModelExport(final_params, strain, fiber_param, WDH_param, models, scalers)
        
        # Plot fits from ML metamodels
        export = PostProcess.ExportData('RV_Holz', 'FWD2_BestFits')
        export.RV_StressStrain(strain, exp_data, best_fit, TSLn)
        
        
        if not os.path.exists('./ExportedData/HolzapfelParams'):
                    os.makedirs('./ExportedData/HolzapfelParams')
        
        param_path = './ExportedData/HolzapfelParams/FWD2_TSL' + str(TSLn) + '.csv'             
        np.savetxt(param_path, final_params, delimiter=",")
    
      
    