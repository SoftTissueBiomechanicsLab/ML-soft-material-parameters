# -*- coding: utf-8 -*-
"""
Main script to train and export GPR Forward models for the Ogden Material

"""

import numpy as np
from random import seed
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from time import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import PreProcess
import PostProcess


# Metamodel settings
NUM_FEATURES = 2
NUM_OUTPUT = 100
L_ScaleBound = 1e-13
U_ScaleBound = 1e+5


# Training and testing data (number of FEBio simulations)
num_train_list = [50,75,100,250,500, 750,1000, 1500, 2000, 2500, 3000]
valid_sets = list(range(10001,11001))
n_val = len(valid_sets)
star_set = 249 # The ID of one validation test to visualize stress-strain

# Reproduce
np.random.seed(1234)
seed(1234)

# Initialize learning curve dictionary
lc_stats = {'num_train':[], 'train_time':[],'MAE_train': [], 'MAPE_train': [], 'R2_train': [],'MAE_val': [], 'MAPE_val': [], 'R2_val': [] }   
    
# Load data
TestData = PreProcess.OgdenData() 
StrainObservs = TestData.FEBio_strain # As is, use all available points   

# Separate Validation data
X_val, Y_val = TestData.FE_in_out(valid_sets,  strain_vals = StrainObservs )

# Loop Training sets 
for kk in range(0, len(num_train_list)):
      
    # Separate Training set
    train_num = num_train_list[kk]
    print(f'TRAIN_n...{train_num+0:03}')
    train_sets = list(range(1, train_num + 1))
    X_train, Y_train = TestData.FE_in_out(train_sets, strain_vals = StrainObservs)
    
      
    # Scale Training Set
    scaler = StandardScaler()
    scaler.fit(X_train)      
        
    # Scale Input Sets
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
                           
    ####### GPR Model ##################
    length_scales = np.ones(2)
    lscale_bounds = (L_ScaleBound, U_ScaleBound)
    kernel =  RBF(length_scales,length_scale_bounds = lscale_bounds) #Matern(length_scales)
    
    model = GaussianProcessRegressor(kernel=kernel, alpha=1e-8, random_state=None,n_restarts_optimizer=10)
    
    # Train, fit model
    start = time()
    model.fit(X_train_scaled, Y_train)
        
    # Find predictions
    fX_train = model.predict(X_train_scaled)
    fX_val = model.predict(X_val_scaled)
    
    # Calcutate train and evaltime         
    end = time()
    run_time = (end - start)/60
    print('\t run time in mins.....%.2f' %(run_time))  
    
    # Initialize utils for post processing
    export = PostProcess.ExportData('Ogden', 'GPR_Tuned'+ str(train_num))
           
    # Export total errors after training
    lc_stats['num_train'].append(train_num)
    lc_stats['train_time'].append(run_time)
    lc_stats = export.compute_error(Y_train, fX_train, lc_stats, 'train')
    lc_stats = export.compute_error(Y_val, fX_val, lc_stats, 'val')
    export.dict_to_csv(lc_stats,'LC')
        
    # Plot star set
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(5,8))
    export.stress_strain(Y_val, fX_val, None, lc_stats['MAE_val'][-1] , StrainObservs, star_set, ax1, ax2)
    
    # Plot stress scattered data
    R2 = 1-lc_stats['R2_val'][-1]
    export.stress_scatter(Y_val,fX_val, R2)
      
    # Export trained models
    export.trained_GPRs(scaler, model, train_num, lscale_bounds)
    plt.close('all')
    
# Plot Learning curve
export.learning_curve(num_train_list,lc_stats['MAE_train'],lc_stats['MAE_val'],'MAE')
export.learning_curve(num_train_list,lc_stats['MAPE_train'],lc_stats['MAPE_val'],'MAPE')
export.learning_curve(num_train_list,lc_stats['R2_train'],lc_stats['R2_val'],'R2')


    
    
    
    
    
    
    
    
    
    
    
    
    
    