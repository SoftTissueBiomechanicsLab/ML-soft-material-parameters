# -*- coding: utf-8 -*-
"""
Material parameter identification 
of the Holzapfel model, using trained NNR

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




if __name__ == "__main__":
    
    # Input (experimental data)
    TSLs = [1]
    t = time()
    
    # Input (trained model data)
    Num_train = 9600
    DirName = './ExportedData/BWD_Holzapfel_Final/Output_BWD_HolzapfelNN_Final_'

    Xscaler_dir = DirName + str(Num_train)  + '/Torch_NN_' + str(Num_train) + 'Xscaler.sav'
    Yscaler_dir = DirName  + str(Num_train) + '/Torch_NN_' + str(Num_train) + 'Yscaler.sav'
    Settings_dir = DirName + str(Num_train)  + '/Torch_NN_' + str(Num_train) + 'Settings.sav'
    Model_path = DirName  + str(Num_train)  + '/Torch_NN_' + str(Num_train) + '.pt'
    
    NN_settings = pickle.load(open(Settings_dir, 'rb'))
    Xscaler = pickle.load(open(Xscaler_dir, 'rb'))
    Yscaler = pickle.load(open(Yscaler_dir, 'rb'))
    NUM_FEATURES = NN_settings['Feat']
    HIDDEN_DIM = NN_settings['Dim']
    NUM_OUTPUT = NN_settings['Out']
    DEPTH = 4#NN_settings['depth']
    N_STRAIN = 10
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = Metamodels.NN(feature_dim=NUM_FEATURES, hidden_dim=HIDDEN_DIM, output_dim=NUM_OUTPUT, depth = DEPTH, Ogden_BWD = False, Holz_BWD = True)
    model.load_state_dict(torch.load(Model_path))
    model.eval()
    
    for TSLn in TSLs:
        print('\n Subject TSL: %i' %(TSLn))
        
        # Read experimental data and parameters        
        strain, exp_data, fiber_param, WDH_param  = PreProcess.load_RV_ExpData(N_STRAIN,TSLn)
        
        # Construct input vector for the inverse NN            
        Strain_amps =  [strain[i][-1] for i in range(1,10)]
        StressStrain_crvs = exp_data.flatten('F')
        
        X_temp = np.concatenate([Strain_amps,StressStrain_crvs, WDH_param, fiber_param])
        
        
        X_temp = np.atleast_2d(X_temp)
        X_scaled = Xscaler.transform(X_temp)
        X_tensor = torch.from_numpy(X_scaled).float().to(DEVICE)

        fX  = model(X_tensor)
        fX_np = fX.data.numpy()
        final_params = Yscaler.inverse_transform(fX_np)
        
        if not os.path.exists('./ExportedData/HolzapfelParams'):
            os.makedirs('./ExportedData/HolzapfelParams')
        param_path = './ExportedData/HolzapfelParams/BWD_TSL_TIME' + str(TSLn) + '.csv'
                
        np.savetxt(param_path, final_params, delimiter=",")
        elapsed = time() - t