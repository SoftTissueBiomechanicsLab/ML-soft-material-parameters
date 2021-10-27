# -*- coding: utf-8 -*-
"""
Read and preprocess data from Finite Element simulations.
-Reads force values and material parameters from csv files
-Converts force to stress
-Sorts and Exports data

@author: Sotiris Kakaletsis
"""
import os
import numpy as np
import pandas as pd
import pickle
import csv
from scipy import interpolate

##############################################################################
# Blood clot data
##############################################################################

class OgdenData:
    # initialize the class
    def __init__(self):
        
        self.FE_must_points = 50 # number of available strain points
        self.strain_peak = 0.5       
        self.FEBio_strain = np.linspace(-self.strain_peak, self.strain_peak, self.FE_must_points)
        
        # load data lists
        self.max_sets, self.strain, self.Fx, self.Fz, self.params = self.load_sav_data()
        
        
    def load_sav_data(self):
       
        # initalize force and params list
        Fx =  pickle.load(open('./SyntheticBloodClotData/FxOgden.sav', 'rb'))
        Fz = pickle.load(open('./SyntheticBloodClotData/FzOgden.sav', 'rb'))
        params = pickle.load(open('./SyntheticBloodClotData/paramsOgden.sav', 'rb'))
        max_num_sets = len(Fz)
        strain = [self.FEBio_strain]*max_num_sets
               
        return max_num_sets, strain, Fx, Fz, params
        
        
    # find the indices closest to the selected strain points
    def find_strain_IDs(self,strain):
        # available strain point data
        x = self.FEBio_strain
        
        # initialize IDs array
        IDs = np.array([])
        
        if isinstance(strain, list):
            for x_d in strain:
                
                idx = (np.abs(x - x_d)).argmin()
                IDs = np.append(IDs,int(idx))
        else:
            idx = (np.abs(x - strain)).argmin()
            IDs = int(idx)
            
        return IDs
    
    # export param values given ID of the set
    def param_vals(self, set_ID):                      
        return self.params[set_ID-1]
 
    # return values of shear and normal force
    def force_strain_vals(self, set_ID, strain):    
        
        strain_ID = self.find_strain_IDs(strain)
        strain_value = self.strain[set_ID-1][strain_ID]
        Fx_value = self.Fx[set_ID-1][strain_ID]
        Fz_value = self.Fz[set_ID-1][strain_ID]

        s_Fx_Fz = np.array([strain_value,Fx_value,Fz_value])
        return s_Fx_Fz    
    
        
    # assembly FE simulation input/output, given IDs of data sets
    def FE_in_out(self,set_IDs, strain_vals):
        
        # initialize data
        X_data =  np.empty((0,2), int)
        Y_data = np.empty((0, 2*len(strain_vals)), int)
        
        # specific strain values provided            
        for i in set_IDs: # loop through train data set
            
            # Matl parameters for i-th set
            x_param = np.atleast_2d(self.param_vals(i))
            ForceVec = np.empty((0,2), int) #initialize
                
            for x in strain_vals:
                
                raw_data = np.array(self.force_strain_vals(i, x))
                # export values
                #gamma = raw_data[0]
                Fx = raw_data[1]
                Fz = raw_data[2]
                ForceVec  = np.vstack((ForceVec,[Fx,Fz])) 
                    
            X_data = np.vstack((X_data,x_param))
            Y_data = np.vstack((Y_data, ForceVec.flatten('F')))
      
        return X_data, Y_data

# Load experimental data
def load_Ogden_ExpData(num_data_points,conc,coagtime,replica, StrainFeature = True):
    # initalize force and params list
    strain = [0]*num_data_points
    Fx = [0]*num_data_points
    Fz = [0]*num_data_points
        
    # read *.csv data to lists
    force_file = './ExperimentalBloodClotData/C' + str(conc) +'_T'+str(coagtime)+ '_'+ replica + '.csv'
            
    with open(force_file, newline='') as ff:
        reader = csv.reader(ff)
        force_data_temp = list(reader)
        
           
    strain_temp = [float(force_data_temp[j][0]) for j in range(1,101)] # starts from 1 to skip column label
    Fx_temp = [float(force_data_temp[j][1]) for j in range(1,101)] 
    Fz_temp = [float(force_data_temp[j][2]) for j in range(1,101)]
    
    # Find epsilon max
    strain_lim = min((abs(strain_temp[0]),abs(strain_temp[-1])))
    if strain_lim > 0.5:
        strain_lim = 0.5
    
    # copy to the "global" list
    #FEBioStrain = np.linspace(-0.5,0.5,50)
    strain =  np.asarray(strain_temp)
    Fx = np.asarray(Fx_temp)
    Fz = np.asarray(Fz_temp)
    
    # Define interpolants
    Fx_interp = interpolate.interp1d(strain, Fx, kind='cubic')
    Fz_interp = interpolate.interp1d(strain, Fz, kind='cubic')                
    
    #Interpolate
    if StrainFeature:
        strain_new  = np.linspace(-strain_lim,strain_lim,num_data_points)
    else:
        strain_new  = np.linspace(strain[0],strain[-1],num_data_points)
        
    Fx_new = Fx_interp(strain_new)
    Fz_new = Fz_interp(strain_new)
    


    return strain_new, Fx_new, Fz_new    

##############################################################################
# Holzapfel myocardium data
##############################################################################

class HolzapfelData:
    # initialize the class
    def __init__(self, mode):
        
        self.FE_must_points = 50 # number of available strain points
        self.mode = mode
        self.isshear = True # check what kind of mode 
        self.strain_peak = 0.4
        if (self.mode % 3)==0:
            self.isshear = False
            self.strain_peak = 0.15 # strain amplitude
        
        self.FEBio_strain = np.linspace(-self.strain_peak, self.strain_peak, self.FE_must_points)
        
        # load data lists
        self.max_sets, self.strain, self.Fx, self.Fz, self.params = self.load_sav_data()
        
        
    def load_sav_data(self):
        # initalize force and params list
        Fz = pickle.load(open('./SyntheticMyocardiumData/Fz_M'+str(self.mode)+'.sav', 'rb'))
        params = pickle.load(open('./SyntheticMyocardiumData/params_M'+str(self.mode)+'.sav', 'rb'))
        max_num_sets = len(Fz)
        strain = [self.FEBio_strain]*max_num_sets
        Fx = [0]*max_num_sets
        if self.isshear:
            Fx =  pickle.load(open('./SyntheticMyocardiumData/Fx_M'+str(self.mode)+'.sav', 'rb'))
        
                 
        return max_num_sets, strain, Fx, Fz, params
        
        
    # find the indices closest to the selected strain points
    def find_strain_IDs(self,strain):
        # available strain point data
        x = self.FEBio_strain
        
        # initialize IDs array
        IDs = np.array([])
        
        if isinstance(strain, list):
            for x_d in strain:
                
                idx = (np.abs(x - x_d)).argmin()
                IDs = np.append(IDs,int(idx))
        else:
            idx = (np.abs(x - strain)).argmin()
            IDs = int(idx)
            
        return IDs
    
    # export param values given ID of the set
    def param_vals(self, set_ID):                      
        return self.params[set_ID-1]
 
    # return values of shear and normal force
    def force_strain_vals(self, set_ID, strain):    
        
        strain_ID = self.find_strain_IDs(strain)
        strain_value = self.strain[set_ID-1][strain_ID]
        if self.isshear:
            Fx_value = self.Fx[set_ID-1][strain_ID]
        else:
            Fx_value = 0.0 # for uniaxial modes, there's no Fx
            
        Fz_value = self.Fz[set_ID-1][strain_ID]

        s_Fx_Fz = np.array([strain_value,Fx_value,Fz_value])
        return s_Fx_Fz    
    
        
    # assembly FE simulation input/output, given IDs of data sets
    def FE_in_out(self,set_IDs, strain_vals):
        
        # initialize data
        X_data =  np.empty((0,17), int)
        
        if self.isshear:
            
            Y_data = np.empty((0, 2*len(strain_vals)), int)
            
            # specific strain values provided            
            for i in set_IDs: # loop through train data set
                
                # Matl parameters for i-th set
                x_param = np.atleast_2d(self.param_vals(i))
                W = x_param[0,8]
                D = x_param[0,9]
                A_area = W*D
                #Delete cube dimensions
                # x_param = np.atleast_2d(np.delete(x_param, [8,9,10]))
                ForceVec = np.empty((0,2), int) #initialize
                    
                for x in strain_vals:
                    
                    
                    
                    raw_data = np.array(self.force_strain_vals(i, x))
                    # export values
                    #gamma = raw_data[0]
                    Fx = raw_data[1]/A_area*1e3 # Convert to kPa
                    Fz = raw_data[2]/A_area*1e3 # Convert to kPa
                    ForceVec  = np.vstack((ForceVec,[Fx,Fz])) 
                        
                X_data = np.vstack((X_data,x_param))
                Y_data = np.vstack((Y_data, ForceVec.flatten('F')))
            
        else:
            
            Y_data = np.empty((0, len(strain_vals)), int)
            
            # specific strain values provided            
            for i in set_IDs: # loop through train data set
                
                # Matl parameters for i-th set
                x_param = np.atleast_2d(self.param_vals(i))
                W = x_param[0,8]
                D = x_param[0,9]
                A_area = W*D
                #Delete cube dimensions
                # x_param = np.atleast_2d(np.delete(x_param, [8,9,10]))
                ForceVec = np.empty((0,1), int) #initialize
                    
                for x in strain_vals:
                    
                    raw_data = np.array(self.force_strain_vals(i, x))
                    # export values
                    #gamma = raw_data[0]
                    Fz = raw_data[2]/A_area *1e3 # Convert to kPa
                    ForceVec  = np.vstack((ForceVec,[Fz])) 
                        
                X_data = np.vstack((X_data,x_param))
                Y_data = np.vstack((Y_data, ForceVec.flatten('F')))
        
      
        return X_data, Y_data 
    
# Load experimental data
def load_RV_ExpData(num_data_points, TSLn):
    # initalize force and params list
    strain = {x: [0]*num_data_points for x in range(1,10)}
    Tdir = {x: [0]*num_data_points for x in range(1,10)}
    Tnor = {x: [0]*num_data_points for x in range(1,10)}
    exp_data = np.array([]).reshape(num_data_points,0)
    
    # Loop through modes
    for i in range(1,10):
        # Check is shear
        IsShear = (i%3) != 0
                
        # read *.csv data to lists
        force_file = './ExperimentalMyocardiumData/TSL' + str(TSLn) +'/StressStrain_M'+ str(i) + '.csv'
        fiber_file = './ExperimentalMyocardiumData/TSL' + str(TSLn) +'/FiberParams.csv'
        WDH_file = './ExperimentalMyocardiumData/TSL' + str(TSLn) +'/WDH_Params.csv'
        
        with open(force_file, newline='') as ff:
            reader = csv.reader(ff)
            force_data_temp = list(reader)
            
        with open(fiber_file, newline='') as ff:
            reader = csv.reader(ff)
            fiber_data_temp = list(reader)
            
        with open(WDH_file, newline='') as ff:
            reader = csv.reader(ff)
            WDH_data_temp = list(reader)
        
        # Save Fiber Data
        fiber_param = [float(fiber_data_temp[1][j]) for j in range(0,6)]
        WDH_param = [float(WDH_data_temp[0][j]) for j in range(0,9)]
        
                
        # Organize Stress Strain Curves
        strain_temp = [float(force_data_temp[j][0]) for j in range(1,101)] # starts from 1 to skip column label
        
        Tx_temp = [float(force_data_temp[j][1]) for j in range(1,101)] 
                
        if IsShear:
            Tz_temp = [float(force_data_temp[j][2]) for j in range(1,101)]
                                    
        else:
            Tz_temp = Tx_temp #both Tx and Tz are the same for uniaxial modes
    
        # Find epsilon max
        strain_lim = min((abs(strain_temp[0]),abs(strain_temp[-1])))
        
        if IsShear:
            if strain_lim > 0.4:
                strain_lim = 0.4
        else:
            if strain_lim > 0.15:
                strain_lim = 0.15
    
        # copy to the "global" list
        strain_temp =  np.asarray(strain_temp)
        Tx = np.asarray(Tx_temp)
        Tz = np.asarray(Tz_temp)
    
        # Define interpolants
        Tx_interp = interpolate.interp1d(strain_temp, Tx, kind='cubic')
        Tz_interp = interpolate.interp1d(strain_temp, Tz, kind='cubic')                
    
        # Save to dictionary
        strain[i]  = np.linspace(-strain_lim,strain_lim,num_data_points)
        Tdir[i] = Tx_interp(strain[i])
        Tnor[i] = Tz_interp(strain[i])
    
        exp_data = np.column_stack((exp_data,Tdir[i]))
        if IsShear:
            exp_data = np.column_stack((exp_data,Tnor[i]))

    return strain, exp_data, fiber_param, WDH_param  

#################################################################################    
    
def assembly_interplated(TestData, features_num, target_num, X_ref, Y_ref, sampling_strain = None):
       
    
    if  sampling_strain is None:
        
        n_sams = X_ref.shape[0]
        X_strain =  TestData.FEBio_strain
        X_strain = np.tile(X_strain,n_sams)[:,np.newaxis]
        X_temp = np.repeat(X_ref,50,axis=0)
        
        X_new = np.hstack((X_strain,X_temp))
        
        if TestData.isshear:
            Fx = Y_ref[:,0:50].flatten()[:,np.newaxis]
            Fz = Y_ref[:,50:100].flatten()[:,np.newaxis]
            
            Y_new = np.hstack((Fx,Fz))
            
        else:
            Fz = Y_ref[:,0:50].flatten()[:,np.newaxis]
                        
            Y_new = Fz
    
    else:
        
        dims = sampling_strain.shape
        n_samples = dims[0]
        m_strains = dims[1]
        
        X_new = np.array([]).reshape(0, features_num)
        Y_new = np.array([]).reshape(0, target_num)
        
        for i in range(0,n_samples):
            
            e_strain = sampling_strain[i,:]
            X_strain = sampling_strain[i,:].reshape(m_strains,1)
            X_temp = X_ref[i,:].reshape(1,features_num-1)
            X_temp = np.tile(np.array(X_temp), (m_strains, 1))
            X_temp = np.hstack((X_strain,X_temp))
            X_new = np.vstack((X_new,X_temp))
            
            if TestData.isshear:
                Fx = Y_ref[i,0:50]
                Fz = Y_ref[i,50:100]
                
                Fx_interp = interpolate.interp1d(TestData.FEBio_strain, Fx, kind='cubic')
                Fz_interp = interpolate.interp1d(TestData.FEBio_strain, Fz, kind='cubic')
                
                Fx_new = Fx_interp(e_strain.T)
                Fz_new = Fz_interp(e_strain.T)
                
                temp = np.hstack((Fx_new[:,np.newaxis],Fz_new[:,np.newaxis]))
                
                Y_new = np.vstack((Y_new,temp))
                
            else:
                Fz = Y_ref[i,0:50]
                Fz_interp = interpolate.interp1d(TestData.FEBio_strain, Fz, kind='cubic')
                Fz_new = Fz_interp(e_strain.T)
                Y_new = np.vstack((Y_new,Fz_new[:,np.newaxis]))
                
    return X_new, Y_new


# Assemle X,Y arrays for all modes for the inverse NN
def sort_data(TestData, n_range_list):
    
    Fx = {'1':[],'2':[],'4':[],'5':[],'7':[],'8':[]}
    Fz = {'1':[],'2':[],'4':[],'5':[],'7':[],'8':[]}
    Fzz = {'3':[],'6':[],'9':[]}
       
    for n_mode in range(1,10):
        target_strain = TestData[str(n_mode)].FEBio_strain # Include all modes
        X,Y = TestData[str(n_mode)].FE_in_out(n_range_list,  strain_vals = target_strain)
        
        if TestData[str(n_mode)].isshear:
            Fx[str(n_mode)].append(Y[:,0:50])
            Fz[str(n_mode)].append(Y[:,50:100])
            
        else:
            Fzz[str(n_mode)].append(Y)
        
        if n_mode == 1:
            matl = X[:,0:8]
            fiber_param = X[:,11:17]
            WDH_cube = X[:,8:11]
        
        elif n_mode==4:
            WDH_cube = np.hstack((WDH_cube, X[:,8:11]))
        
        elif n_mode==7:
            WDH_cube = np.hstack((WDH_cube, X[:,8:11]))
            
    param = np.hstack((WDH_cube,fiber_param))
   
    return Fx,Fz,Fzz, param, matl


# Sample strain amplitude for all modes
def sample_strain(lhd):
    
    Strain_amp = [0.3, 0.3, 0.10, 0.3, 0.3, 0.10, 0.3, 0.3, 0.10]+lhd*[0.1, 0.1, 0.05, 0.1, 0.1, 0.05, 0.1, 0.1, 0.05]
    
    return Strain_amp

def sample_OgdenStrain(lhd, StrainFeature = True):
    
    if StrainFeature:
        Strain_amp = [0.4]+lhd*[0.1]
        
    else:
        Strain_amp = 0.5*np.ones(np.shape(lhd))
    
    return Strain_amp


# Interpolate force-strain curves based on new strain amplitude
def interpolate_ss(TestData, n_samples, n_strain, Fx_in, Fz_in ,Fzz_in, strain_amp=None):
    
     
    Fx_out= {'1':[],'2':[],'4':[],'5':[],'7':[],'8':[]}
    Fz_out = {'1':[],'2':[],'4':[],'5':[],'7':[],'8':[]}
    Fzz_out = {'3':[],'6':[],'9':[]}
    
    for n_mode in range(1,10):
        
        target_strain = TestData[str(n_mode)].FEBio_strain # Include all modes
        
        for n_sam in range (0, n_samples):
            
            if strain_amp is not None:
                max_strain = strain_amp[n_sam,n_mode-1]
                interp_strain = np.linspace(-max_strain, max_strain, n_strain)
            
            if TestData[str(n_mode)].isshear:
                
                Fx_temp = Fx_in[str(n_mode)][0][n_sam,:]
                Fz_temp = Fz_in[str(n_mode)][0][n_sam,:]
                
                if strain_amp is not None:
                
                    Fx_interpolator = interpolate.interp1d(target_strain, Fx_temp, kind='cubic')
                    Fz_interpolator = interpolate.interp1d(target_strain, Fz_temp, kind='cubic')
                    
                    Fx_new = Fx_interpolator(interp_strain)
                    Fz_new = Fz_interpolator(interp_strain)
                    
                    Fx_out[str(n_mode)].append(Fx_new)
                    Fz_out[str(n_mode)].append(Fz_new)
                    
                else:
                    
                    Fx_out[str(n_mode)].append(Fx_temp)
                    Fz_out[str(n_mode)].append(Fz_temp)
                
            else:
                
                Fzz_temp = Fzz_in[str(n_mode)][0][n_sam,:]
                
                if strain_amp is not None:
                
                    Fzz_interpolator = interpolate.interp1d(target_strain, Fzz_temp, kind='cubic')
                    
                    Fzz_new = Fzz_interpolator(interp_strain)
                    
                    Fzz_out[str(n_mode)].append(Fzz_new)
                    
                else:
                    
                    Fzz_out[str(n_mode)].append(Fzz_temp)
            
    return Fx_out, Fz_out, Fzz_out


# Assemble stress vector
def assemble_StressVec( TestData, Fx_train, Fz_train, Fzz_train):
    
    StressVec = []
    for n_mode in range(1,10):
                            
        if TestData[str(n_mode)].isshear:
                        
            StressVec.append(np.asarray(Fx_train[str(n_mode)]))
            StressVec.append(np.asarray(Fz_train[str(n_mode)]))
            
        else:
            
            StressVec.append(np.asarray(Fzz_train[str(n_mode)]))
            
    StressVec = np.concatenate(StressVec, axis=1)
    
        
    return StressVec

# Assemble Inverse Neural Network input
def assemble_X(StressVec, param, strain_amp):
    
    # Strain Magnitudes
    X = strain_amp
    
    # Stack Stress components
    X = np.hstack(( X , StressVec))        
    X = np.hstack((X, param))
    
    return X

def sort_OgdenData(TestData, n_range_list):
    
    target_strain = TestData.FEBio_strain # Include all modes
    X,Y = TestData.FE_in_out(n_range_list,  strain_vals = target_strain)
    Fx = Y[:,0:50]
    Fz = Y[:,50:100]
    matl = X[:,0:2]   
    
    return Fx,Fz,matl

# Interpolate force-strain curves based on new strain amplitude Ogden
def interpolate_Ogden_ss(TestData, n_samples, n_strain, Fx_in, Fz_in , strain_amp):
    
     
    Fx_out = []
    Fz_out = []
    
    target_strain = TestData.FEBio_strain # Include all modes
       
    for n_sam in range (0, n_samples):
        
        if strain_amp is not None:
            max_strain = strain_amp[n_sam,0]
            interp_strain = np.linspace(-max_strain, max_strain, n_strain)
                            
            Fx_temp = Fx_in[n_sam,:]
            Fz_temp = Fz_in[n_sam,:]
            
            if strain_amp is not None:
            
                Fx_interpolator = interpolate.interp1d(target_strain, Fx_temp, kind='cubic')
                Fz_interpolator = interpolate.interp1d(target_strain, Fz_temp, kind='cubic')
                
                Fx_new = Fx_interpolator(interp_strain)
                Fz_new = Fz_interpolator(interp_strain)
                
                Fx_out.append(Fx_new)
                Fz_out.append(Fz_new)
    
    return Fx_out, Fz_out


# Assemble Inverse Neural Network input
def assemble_OgdenX(Strain, Fx, Fz, StrainFeature =True):
    
    # Stack Stress components
    X = np.hstack(( Fx , Fz))
    
    # Strain Magnitudes
    if StrainFeature:
        X = np.column_stack(( Strain , X))
       
    return X


##############################################################################
# Holzapfel a neural network for everything
##############################################################################

def all_modes_FE_in_out(setIDs, TestData):
    
    n_sams = len(setIDs)
        
    # Initialize
    X = np.zeros((n_sams,0))
    Y = np.zeros((n_sams,0))
    # Loop all modes
    for mode_i in range(1,10):
        
        target_strain = TestData[mode_i].FEBio_strain
        X_temp, Y_temp = TestData[mode_i].FE_in_out(setIDs,  strain_vals = target_strain)
        
        Y = np.hstack((Y,Y_temp))
        
        if mode_i == 1:
            X = np.hstack((X,X_temp[:,0:11]))
            
        elif mode_i == 4:
            X = np.hstack((X,X_temp[:,8:11]))
            
        elif mode_i == 7:
            X = np.hstack((X,X_temp[:,8:]))
    
    return X, Y



























