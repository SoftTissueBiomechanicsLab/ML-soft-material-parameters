# -*- coding: utf-8 -*-
"""
Material parameter identification 
of the Ogden model, using trained NNR

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
    conc_list = [10,20,40]
    coagtime_list = [60,90,120]
    replica_list = ['x', 'y', 'z']
    
    # Input (trained model data)
    Num_train = 8000
    DirName = './ExportedData/BWD_Ogden_Final/Output_BWD_Ogden_FINAL_8000'

    Xscaler_dir = DirName  + '/Torch_NN_' + str(Num_train) + 'Xscaler.sav'
    Yscaler_dir = DirName  + '/Torch_NN_' + str(Num_train) + 'Yscaler.sav'
    Settings_dir = DirName  + '/Torch_NN_' + str(Num_train) + 'Settings.sav'
    Model_path = DirName  + '/Torch_NN_' + str(Num_train) + '.pt'
    
    NN_settings = pickle.load(open(Settings_dir, 'rb'))
    Xscaler = pickle.load(open(Xscaler_dir, 'rb'))
    Yscaler = pickle.load(open(Yscaler_dir, 'rb'))
    NUM_FEATURES = NN_settings['Feat']
    HIDDEN_DIM = NN_settings['Dim']
    NUM_OUTPUT = NN_settings['Out']
    DEPTH = NN_settings['depth']
    N_STRAIN = 10 #for the version with Strain
    STRAIN_FEATURE = True
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = Metamodels.NN(feature_dim=NUM_FEATURES, hidden_dim=HIDDEN_DIM, output_dim=NUM_OUTPUT, depth = DEPTH, Ogden_BWD = True)
    model.load_state_dict(torch.load(Model_path))
    model.eval()
    
    # Read Experimental Data
    for i,conc in enumerate(conc_list):
        for ii,coagtime in enumerate(coagtime_list):
            for iii, replica in enumerate(replica_list):
                
                strain, Fx, Fz = PreProcess.load_Ogden_ExpData(N_STRAIN,conc,coagtime,replica,STRAIN_FEATURE)
                data = np.asarray([Fx, Fz]).T
                if STRAIN_FEATURE:
                    x = max(strain)
                    print(x)
                    X_temp = np.hstack([x,data.flatten('F')])
                else:
                    X_temp = data.flatten('F')
                    
                X_temp = np.atleast_2d(X_temp)
                X_scaled = Xscaler.transform(X_temp)
                X_tensor = torch.from_numpy(X_scaled).float().to(DEVICE)
    
                fX  = model(X_tensor)
                fX_np = fX.data.numpy()
                final_params = Yscaler.inverse_transform(fX_np)
                
                if not os.path.exists('./ExportedData/OgdenParams'):
                    os.makedirs('./ExportedData/OgdenParams')
                param_path = './ExportedData/OgdenParams/BWD_C' + str(conc) +'_T'+str(coagtime)+ '_'+ replica + '.csv'
                img_path = './ExportedData/OgdenParams/BWD_C' + str(conc) +'_T'+str(coagtime)+ '_'+ replica + '.png'
                
                np.savetxt(param_path, final_params, delimiter=",")
                plt.savefig(img_path, bbox_inches='tight')