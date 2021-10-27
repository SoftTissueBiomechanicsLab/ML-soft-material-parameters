# -*- coding: utf-8 -*-
"""
Main script to train and export NN Inverse models for the Holzapfel Material.
Input: Stress (kPa) - Strain curves + cube dimensions + fiber orientation
Output: Material parameters

"""
import numpy as np
from random import seed
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from time import time
from pyDOE import lhs

import PreProcess
import Metamodels
import PostProcess


# Optimizer settings
RUN_NAME = 'MyocardiumNNR_'
EPOCHS = 500
LEARNING_RATE = 0.001
HIDDEN_DIM = 50
NUM_OUTPUT = 2
N_STRAIN = 30 # Originally 30 for the model w/ Strain
DEPTH = 3
STRAIN_FEATURE = True  
PRINT_INT = 5 # Print gif every other PRINT_INT epoch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training and testing data (number of FEBio simulations)
num_train_list = [8000, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000]
valid_sets = list(range(10001,11001))
n_val = len(valid_sets)

# Reproduce
np.random.seed(1234)
seed(1234)
torch.manual_seed(1234)

# Initialize val data, test data and learning curve dictionary
if STRAIN_FEATURE:
    NUM_FEATURES = 1+N_STRAIN*2
else:
    NUM_FEATURES = N_STRAIN*2
TestData ={'1':[],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[]}
lc_stats = {'num_train':[],'train_time':[], 'MAE_train': [], 'MAPE_train': [], 'R2_train': [],'MAE_val': [], 'MAPE_val': [], 'R2_val': [] }
       
# Initialize data class
TestData = PreProcess.OgdenData()

# Load Validation data
lhd_val  = lhs(1, samples=n_val) # Sample strain amplitude for validation
Fx_val, Fz_val, matl_val = PreProcess.sort_OgdenData(TestData, valid_sets)
Strain_val = PreProcess.sample_OgdenStrain(lhd_val,STRAIN_FEATURE)
Fx_val, Fz_val = PreProcess.interpolate_Ogden_ss(TestData, n_val, N_STRAIN ,Fx_val, Fz_val , Strain_val)
 
# Loop Training sets 
for kk in range(0, len(num_train_list)):

    # Load Training set
    train_sets = list(range(1,num_train_list[kk]+1))
    n_train = len(train_sets)
    print(f'TRAIN_n...{n_train+0:03}')
    BATCH_SIZE = min(n_train,100)
    Fx_train, Fz_train, matl_train = PreProcess.sort_OgdenData(TestData,train_sets)
    
    
    # Sample the strain Amplitudes, Shear:0.3-0.4 and Uniaxial: 0.10-0.15
    lhd_train = lhs(1, samples=n_train)
    Strain_train = PreProcess.sample_OgdenStrain(lhd_train, STRAIN_FEATURE)
        
    # Interpolate stress strain curve
    Fx_train, Fz_train = PreProcess.interpolate_Ogden_ss(TestData, n_train, N_STRAIN, Fx_train, Fz_train , Strain_train)
       
    # Assemble Stress Vectors
    X_train = PreProcess.assemble_OgdenX(Strain_train, Fx_train, Fz_train, STRAIN_FEATURE)
    X_val = PreProcess.assemble_OgdenX(Strain_val, Fx_val, Fz_val, STRAIN_FEATURE)  
   
    Y_train = matl_train
    Y_val = matl_val
          
    # Scale both input and output
    Xscaler, Yscaler = StandardScaler(), Metamodels.OgdenScaler()
    Xscaler.fit(X_train)
    

    Y_train_scaled = Yscaler.transform(Y_train)          
    Y_val_scaled = Yscaler.transform(Y_val)          
      
    # Add noise to training data
    # noise = np.random.normal(0,.001, X_train.shape)
    noise = None
    # Prepare data to Torch compatible
    X_train_tensor, train_data, train_loader = Metamodels.scaled_to_tensor(DEVICE, Xscaler, X_train, Y_train, BATCH_SIZE, Yscaler, noise)
    X_val_tensor, val_data, val_loader = Metamodels.scaled_to_tensor(DEVICE, Xscaler, X_val, Y_val, n_val ,Yscaler)
    
    # Initialize utils for post processing
    export = PostProcess.ExportData('BWD_Ogden', RUN_NAME + str(n_train) )
    loss_stats = {'train': [], "val": [] }
    epochFitImg = []
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(10,14))
            
    # Set up neural network metamodel
    model = Metamodels.NN(feature_dim=NUM_FEATURES, hidden_dim=HIDDEN_DIM, output_dim=NUM_OUTPUT, depth = DEPTH, Ogden_BWD = True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_func = torch.nn.L1Loss()  
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
            fX_val_scaled = fX_val_tensor.data.numpy()
            fX_val = Yscaler.inverse_transform(fX_val_scaled)
        
            fX_train_tensor = model(X_train_tensor)
            fX_train_scaled = fX_train_tensor.data.numpy()
            fX_train = Yscaler.inverse_transform(fX_train_scaled)
                        
        # Generate gif for Training of star set
        export.OgdenParam_scatter(Y_val,fX_val, epoch, val_epoch_loss/len(val_loader), ax1, ax2)
        fig.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        if (epoch==1) or (epoch % PRINT_INT == 0):
            epochFitImg.append(image)

    # Calculate Run Time
    end = time()
    run_time = (end - start)/60
    
    # Export total errors after training
    lc_stats['num_train'].append(n_train)
    lc_stats['train_time'].append(run_time)
    lc_stats = export.compute_error(Y_train_scaled, fX_train_scaled, lc_stats, 'train')
    lc_stats = export.compute_error(Y_val_scaled, fX_val_scaled, lc_stats, 'val')
    export.dict_to_csv(lc_stats,'LC')
    export.dict_to_csv(loss_stats,'EpochLoss')
    export.OgdenParam_scatter(Y_val,fX_val, epoch, val_epoch_loss/len(val_loader), ax1, ax2, final=True)
    export.epoch_curve(loss_stats)        
        
    # Save training gif
    export.save_gif(n_train, epochFitImg)
    
    # Export Trained NN
    export.trained_NNs(Xscaler, model, n_train, HIDDEN_DIM, NUM_FEATURES, NUM_OUTPUT, Yscaler, DEPTH)

# Save learning curve    
export.learning_curve(num_train_list,lc_stats['MAE_train'],lc_stats['MAE_val'],'MAE')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    












            

    
    
    
    
    
    
    
    
    
    