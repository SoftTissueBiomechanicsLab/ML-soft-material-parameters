# -*- coding: utf-8 -*-
"""
Main script to train and export NN Forward models for the Holzapfel Material

"""
import numpy as np
from random import seed
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from time import time

import PreProcess
import Metamodels
import PostProcess

# Select testing mode (1-9)
TEST_MODE_list = [1,2,3,4,5,6,7,8,9]


# Optimizer settings
RUN_NAME = 'MyocardiumNN_'
EPOCHS = 500
LEARNING_RATE = 0.001
HIDDEN_DIM = 100
DEPTH = 4
NUM_FEATURES = 17
PRINT_INT = 5 # Print gif every other PRINT_INT epoch
NOISE_STD = 1e-2
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training and testing data (number of FEBio simulations)
num_train_list = [1000, 1500, 2000, 2500, 3000,4000, 5000, 6000, 7000, 8000, 9000, 9600]
valid_sets = list(range(9601,10801))
test_sets = list(range(10801,12001))
n_val = len(valid_sets)
star_set = 250 # The ID of one validation test to visualize stress-strain

# Loop over modes
for ii in range(0,len(TEST_MODE_list)):
            
    # Initialize learning curve dictionary and Export data class
    TEST_MODE =  TEST_MODE_list[ii]
    print('\n MODE: %i' %(TEST_MODE))
    lc_stats = {'num_train':[], 'train_time':[],'MAE_train': [], 'MAPE_train': [], 'R2_train': [],'MAE_val': [], 'MAPE_val': [], 'R2_val': [] }   
    
    # Load data
    TestData = PreProcess.HolzapfelData(TEST_MODE) 
    StrainObservs = TestData.FEBio_strain # As is, use all available points   
    # Check if current mode is shear or not
    if TestData.isshear:
        NUM_OUTPUT = 100
            
    else:
        NUM_OUTPUT = 50
         
            
    # Separate Validation data
    X_val, Y_val = TestData.FE_in_out(valid_sets,  strain_vals = StrainObservs)
        
    # Loop Training sets 
    for kk in range(0, len(num_train_list)):
    
        # Reproduce
        np.random.seed(1234)
        seed(1234)
        torch.manual_seed(42)
        
        # Separate Training set
        train_num = num_train_list[kk]
        print(f'TRAIN_n...{train_num+0:03}')
        BATCH_SIZE = min(train_num,100)
        train_sets = list(range(1, train_num + 1))
        X_train, Y_train = TestData.FE_in_out(train_sets, strain_vals = StrainObservs)
                      
        # Scale Training Set
        scaler = StandardScaler()
        scaler.fit(X_train)
                
        # Prepare data to Torch compatible
        X_train_tensor, train_data, train_loader = Metamodels.scaled_to_tensor(DEVICE, scaler, X_train, Y_train, BATCH_SIZE)
        X_val_tensor, val_data, val_loader = Metamodels.scaled_to_tensor(DEVICE, scaler, X_val, Y_val, n_val)
                        
        # Initialize utils for post processing
        export = PostProcess.ExportData('Holzapfel', RUN_NAME + str(train_num) + 'Mode_' + str(TEST_MODE))
        loss_stats = {'train': [], "val": [] }
        epochFitImg = []
        if TestData.isshear:
            fig, (ax1, ax2) = plt.subplots(2,1,figsize=(5,8))
        else:
            fig, ax1 = plt.subplots(figsize=(5,4))
            ax2 = None
                
        # Set up neural network metamodel
        model = Metamodels.NN(feature_dim=NUM_FEATURES, hidden_dim=HIDDEN_DIM, output_dim=NUM_OUTPUT, depth=DEPTH)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_func = torch.nn.L1Loss() 
        #loss_func = Metamodels.CustomLoss(1e-1)
        #loss_func = Metamodels.R2Loss()
        train_step = Metamodels.make_train_step(model, loss_func, optimizer)
        
        # BEGIN TRAINNING
        start = time()
        for epoch in range(1,EPOCHS+1):
            
            # Initialize train and validation error
            train_epoch_loss = 0
            val_epoch_loss = 0
            
            # Batch training data
            for x_batch, y_batch in train_loader:
                
                # Send mini-batches to the device 
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                
                # Add noise to input
                noise = np.random.normal(0, NOISE_STD, x_batch.shape)
                x_batch += torch.from_numpy(noise)
                    
                train_loss = train_step(x_batch, y_batch)
                train_epoch_loss += train_loss
            
            # Stop training and batch validation data    
            with torch.no_grad():
                for x_val_batch, y_val_batch in val_loader:
                    x_val_batch = x_val_batch.to(DEVICE)
                    y_val_batch = y_val_batch.to(DEVICE)
                    
                    model.eval()
        
                    yhat = model(x_val_batch)
                    val_loss = loss_func(y_val_batch, yhat)
                                            
                    val_epoch_loss += val_loss.item()
                    
            print(f'Epoch {epoch+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f}')
            loss_stats['train'].append(train_epoch_loss/len(train_loader))
            loss_stats['val'].append(val_epoch_loss/len(val_loader))
                                   
            with torch.no_grad():
                model.eval()
                fX_val_tensor = model(X_val_tensor)
                fX_train_tensor = model(X_train_tensor)
                
                # Convert to numpy arrays
                fX_val = fX_val_tensor.data.numpy()
                fX_train = fX_train_tensor.data.numpy()
                            
            # Generate gif for Training of star set
            export.stress_strain(Y_val, fX_val, epoch, val_epoch_loss/len(val_loader), StrainObservs, star_set, ax1, ax2)
            fig.canvas.draw()       # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
            if (epoch==1) or (epoch % PRINT_INT == 0):
                epochFitImg.append(image)

        # Calculate Run Time
        end = time()
        run_time = (end - start)/60
        
        # Export total errors after training
        lc_stats['num_train'].append(train_num)
        lc_stats['train_time'].append(run_time)
        lc_stats = export.compute_error(Y_train, fX_train, lc_stats, 'train')
        lc_stats = export.compute_error(Y_val, fX_val, lc_stats, 'val')
        export.dict_to_csv(lc_stats,'LC')
        export.dict_to_csv(loss_stats,'EpochLoss')
        export.stress_strain(Y_val, fX_val, epoch, val_epoch_loss/len(val_loader), StrainObservs, star_set, ax1, ax2, final = True)
        export.epoch_curve(loss_stats) 
         
        # Plot stress scattered data
        R2 = 1-lc_stats['R2_val'][-1]
        export.stress_scatter(Y_val,fX_val, R2)
        
        # Save training gif
        export.save_gif(train_num, epochFitImg)
        
        Worst_ID, WorstScore, Best_ID, BestScore, MAEs = export.find_Best_Worst(Y_val,fX_val)

        #Export Best, Worst
        if TestData.isshear:
            fig, (ax1, ax2) = plt.subplots(2,1,figsize=(5,8))
        else:
            fig, ax1 = plt.subplots(figsize=(5,4))
            ax2 = None
            
        export.stress_strain(Y_val, fX_val, None, WorstScore, StrainObservs, Worst_ID, ax1, ax2, True, 'Worst')
        export.stress_strain(Y_val, fX_val, None, BestScore, StrainObservs, Best_ID, ax1, ax2, True, 'Best')
        
        # Export trained models
        export.trained_NNs(scaler, model, train_num, HIDDEN_DIM, NUM_FEATURES, NUM_OUTPUT)
        plt.close('all')
        
    # Plot Learning curve
    export.learning_curve(num_train_list,lc_stats['MAE_train'],lc_stats['MAE_val'],'MAE')
    export.learning_curve(num_train_list,lc_stats['MAPE_train'],lc_stats['MAPE_val'],'MAPE')
    export.learning_curve(num_train_list,lc_stats['R2_train'],lc_stats['R2_val'],'R2')

 
    
    
    
    
    
    
    
    
    
    
    
    
    
    