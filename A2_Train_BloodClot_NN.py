# -*- coding: utf-8 -*-
"""
Main script to train and export NN Forward models for the Ogden Material

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


# Optimizer settings
RUN_NAME = 'BloodClotNN_'
EPOCHS = 500
LEARNING_RATE = 0.001
HIDDEN_DIM = 50
DEPTH = 3
NUM_FEATURES = 2
NUM_OUTPUT = 100
PRINT_INT = 1 # Print gif every other PRINT_INT epoch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training and testing data (number of FEBio simulations)
num_train_list = [100, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000]
valid_sets = list(range(10001,11001))
n_val = len(valid_sets)
star_set = 250 # The ID of one validation test to visualize stress-strain

# Reproduce
np.random.seed(1234)
seed(1234)
torch.manual_seed(42)

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
    BATCH_SIZE = min(train_num,100)
    train_sets = list(range(1, train_num + 1))
    X_train, Y_train = TestData.FE_in_out(train_sets, strain_vals = StrainObservs)
    
    # Scale Training Set
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Prepare data to Torch compatible
    X_train_tensor, train_data, train_loader = Metamodels.scaled_to_tensor(DEVICE, scaler, X_train, Y_train, BATCH_SIZE)
    X_val_tensor, val_data, val_loader = Metamodels.scaled_to_tensor(DEVICE, scaler, X_val, Y_val, n_val)
    
    # Set up neural network metamodel
    model = Metamodels.NN(feature_dim=NUM_FEATURES, hidden_dim=HIDDEN_DIM, output_dim=NUM_OUTPUT, , depth=DEPTH)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_func = torch.nn.L1Loss()  
    train_step = Metamodels.make_train_step(model, loss_func, optimizer)
    
    # Initialize utils for post processing
    export = PostProcess.ExportData('Ogden', RUN_NAME + str(train_num))
    loss_stats = {'train': [], "val": [] }
    epochFitImg = []
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(5,8))
    
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
    export.stress_scatter(Y_val, fX_val, R2)
    
    # Save training gif
    export.save_gif(train_num, epochFitImg)
    
    # Export trained models
    export.trained_NNs(scaler, model, train_num, HIDDEN_DIM, NUM_FEATURES, NUM_OUTPUT)
    plt.close('all')
        
# Plot Learning curve
export.learning_curve(num_train_list,lc_stats['MAE_train'],lc_stats['MAE_val'],'MAE')
export.learning_curve(num_train_list,lc_stats['MAPE_train'],lc_stats['MAPE_val'],'MAPE')
export.learning_curve(num_train_list,lc_stats['R2_train'],lc_stats['R2_val'],'R2')


    
    
    
    
    
    
    
    
    
    
    
    
    
    