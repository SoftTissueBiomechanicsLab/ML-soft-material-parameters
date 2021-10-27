# -*- coding: utf-8 -*-
"""
Exports plots and data in trained models

@author: Sotiris
"""
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import pickle
import imageio
import torch
from random import seed
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

class ExportData:
    # initialize the class
    def __init__(self, material, ML_method):
        
        self.matl = material
        self.ML = ML_method
        self.OutDir = './ExportedData/Output_' + material + ML_method
        self.PlotDir = './ExportedData/Plots_' + material + ML_method
        self.MetaMdlDir = self.OutDir #+ '/Trained_models'
        
        # Create directories to save files
        if not os.path.exists(self.OutDir):
            os.makedirs(self.OutDir)
            
        if not os.path.exists(self.PlotDir):
            os.makedirs(self.PlotDir)
            
        if not os.path.exists(self.MetaMdlDir):
            os.makedirs(self.MetaMdlDir)

    def list_to_csv(self, data_list, Var2save):
                                
        #Save Any Variable data
        DF = pd.DataFrame(data_list)
        DF.to_csv(self.OutDir + Var2save +'.csv')
    
    def dict_to_csv(self, data_dict, data_name):
                  
        #Export data dictionaly to csv
        filename = self.OutDir + '/'+ data_name +'_NN.csv'  
        pd.DataFrame.from_dict(data=data_dict).to_csv(filename)
   
    def save_gif(self,train_num, epochFitImg):
        
        filename = self.PlotDir + '/' + self.matl + f'_{train_num+0:03}.gif'
        # Save images as a gif    
        imageio.mimsave(filename, epochFitImg, fps=10)
        
    
    def learning_curve(self, num_train_list,train_error,test_error,error_type):
        
        plt.figure()
        plt.plot(num_train_list, train_error,'g-',label='Train')
        plt.plot(num_train_list, test_error,'r-',label='Validation')    
        plt.legend()
        plt.title(self.ML + ' learning curve')
        plt.ylabel(error_type)
        plt.xlabel('Number of training samples')
        #plt.ylim((0,0.01))
        plt.savefig(self.PlotDir + '/FWD_' + error_type +'_' + self.ML +'_LC.pdf', bbox_inches='tight')
        
    def width_curve(self, num_train_list,train_error,test_error,error_type):
        
        plt.figure()
        plt.plot(num_train_list, train_error,'g-',label='Train')
        plt.plot(num_train_list, test_error,'r-',label='Validation')    
        plt.legend()
        plt.title(self.ML + ' width curve')
        plt.ylabel(error_type)
        plt.xlabel('Number of nodes per hidden layer')
        #plt.ylim((0,0.01))
        plt.savefig(self.PlotDir + '/FWD_' + error_type +'_' + self.ML +'_WidthCurve.pdf', bbox_inches='tight')
        
    def depth_curve(self, num_train_list,train_error,test_error,error_type):
        
        plt.figure()
        plt.plot(num_train_list, train_error,'g-',label='Train')
        plt.plot(num_train_list, test_error,'r-',label='Validation')    
        plt.legend()
        plt.title(self.ML + ' depth curve')
        plt.ylabel(error_type)
        plt.xlabel('Number of hidden layers')
        #plt.ylim((0,0.01))
        plt.savefig(self.PlotDir + '/FWD_' + error_type +'_' + self.ML +'_DepthCurve.pdf', bbox_inches='tight')
        
        
    def noise_curve(self, num_train_list,train_error,test_error,error_type):
        
        plt.figure()
        plt.semilogx(num_train_list, train_error,'g-',label='Train')
        plt.semilogx(num_train_list, test_error,'r-',label='Validation')    
        plt.legend()
        plt.title(self.ML + ' LR curve')
        plt.ylabel(error_type)
        plt.xlabel('Learning Rate')
        #plt.ylim((0,0.01))
        plt.savefig(self.PlotDir + '/FWD_' + error_type +'_' + self.ML +'_LRCurve.pdf', bbox_inches='tight')
        
    def epoch_curve(self, loss_stats):
        
        plt.figure(figsize=(5,4))
        plt.plot(loss_stats['train'],'g-',label='Train')
        plt.plot(loss_stats['val'],'r-',label='Validation')   
        plt.legend(fontsize=10)
        plt.xlabel('Number of training epochs', fontsize=10)
        plt.ylabel('MAE', fontsize=10)
        plt.savefig(self.PlotDir+'/TrainingCurveEpochs.pdf',bbox_inches='tight')
         
        
    def stress_strain(self, Y_true,Y_pred, step, loss,strain_plot, IDn, ax1 = None, ax2 = None, final = None,  SpecialLabel = None):
        
                
        if ax2:
            # Print
            ax1.cla()
            ax1.set_ylabel('Shear Stress [kPa]', fontsize=10)
            ax1.scatter(strain_plot, Y_true[IDn,0:50], s=4, color = "black")
            shear_lim = ax1.get_ylim()
            #ax1.plot(strain_plot, Y_pred[IDn,0:50], 'r-', lw=1.5)
            ax1.plot(strain_plot, Y_pred[IDn,0:50], '-',color = [204/255,85/255,0], lw=1.5)
            if step:
                ax1.set_ylim(shear_lim)
                ax1.text(0.1, 0.6*shear_lim[0], 'Epoch = %d' % step, fontdict={'size': 10, 'color':  'tab:blue'})
            ax1.text(0.1, 0.8*shear_lim[0], 'Val. Loss = %.4f' % loss, fontdict={'size': 10, 'color':  'tab:blue'})
            
            ax2.cla()
            ax2.set_xlabel('Strain', fontsize=10)
            ax2.set_ylabel('Normal Stress [kPa]', fontsize=10)
            ax2.scatter(strain_plot, Y_true[IDn,50:101], s=4, color = "black")
            normal_lim = ax2.get_ylim()
            if step:
                ax2.set_ylim(normal_lim)
            #ax2.plot(strain_plot,Y_pred[IDn,50:101], 'r-', lw=1.5)
            ax2.plot(strain_plot,Y_pred[IDn,50:101], '-', color = [204/255,85/255,0], lw=1.5)
                
        else:
            ax1.cla()
            ax1.set_ylabel('Normal Stress [kPa]', fontsize=10)
            ax1.set_xlabel('Strain', fontsize=10)
            ax1.scatter(strain_plot, Y_true[IDn,0:50], s=4, color = "black")
            uniax_lim = ax1.get_ylim()
            ax1.plot(strain_plot, Y_pred[IDn,0:50], 'r-', lw=1.5)
            if step:
                ax1.set_ylim(uniax_lim)
                ax1.text(0.05, 0.6*uniax_lim[0], 'Epoch = %d' % step, fontdict={'size': 10, 'color':  'red'})
            ax1.text(0.05, 0.8*uniax_lim[0], 'Val. Loss = %.4f' % loss, fontdict={'size': 10, 'color':  'red'})
            
       
        fig_name_str = self.PlotDir + '/StarSet_'+ self.ML
        
        if SpecialLabel is not None:
            fig_name_str = self.PlotDir + '/StressStrain_'+ self.ML + SpecialLabel
        
        plt.savefig(fig_name_str+'.png', bbox_inches='tight')
        if final:
            plt.savefig(fig_name_str+'.pdf', bbox_inches='tight')
        
            
    def stress_scatter(self, Ytrue ,Ypred, R2, ext_label = None):
        #  ##### ERASE THAT 
        # Ytrue = 10*Ytrue
        # Ypred = 10*Ypred
        # ########################
        fig_name_str = self.PlotDir + '/Scatter_'+ self.ML 
        if ext_label is not None:
            fig_name_str = fig_name_str+ ext_label
        
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams.update({'font.size': 8})               
        fig, (ax1,ax2) = plt.subplots(2,1,figsize=(1.625,6)) 
    
        
        ax1.set_title('R2 = %.5f'%(R2))
        ax1.scatter(Ytrue[:,0:50], Ypred[:,0:50], s=2 ,color = "black")
        y_limits = ax1.get_ylim()
        x_limits = ax1.get_xlim()
        l_bound =  min(min(y_limits),min(x_limits))
        u_bound = max(max(y_limits),max(x_limits))
        #####################################
        # l_bound =  -80 
        # u_bound = 80 
        # ax1.set_xticks((0,15,30))
        # ax1.set_yticks((0,15,30))
        #######################################
        ax1.plot([l_bound,u_bound], [l_bound,u_bound],color='#E63946', linestyle='--', linewidth=1 )
        ax1.set_xlabel('Target', fontsize=8)
        ax1.set_ylabel('Predicted', fontsize=8)
        # ax1.set_title('Shear force')
        ax1.set_xlim([l_bound,u_bound])
        ax1.set_ylim([l_bound,u_bound])
        # ax1.set_aspect('equal', adjustable='box')
        ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label]+ax1.get_xticklabels() + ax1.get_yticklabels()):
            item.set_fontsize(8)
      
        
        #---------------------- NORMAL FORCE ------------------------------  
          
        ax2.set_title('R2 = dmm')
        ax2.scatter(Ytrue[:,50:100],Ypred[:,50:100], s=2 ,color = "black")
        y_limits = ax2.get_ylim()
        x_limits = ax2.get_xlim()
        l_bound = min(min(y_limits),min(x_limits))
        u_bound = max(max(y_limits),max(x_limits))
        #####################################
        # l_bound =  0 
        # u_bound = 4 
        # ax2.set_xticks((0,2,4))
        # ax2.set_yticks((0.00,.25,0.50))
        #######################################
        ax2.plot([l_bound,u_bound], [l_bound,u_bound],color='#E63946', linestyle='--', linewidth=1 )
        ax2.set_xlabel('True [kPa]', fontsize=8)
        ax2.set_ylabel('Predicted ', fontsize=8)
        ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable='box')
        #ax2.set_title('Normal force')
        ax2.set_xlim([l_bound,u_bound])
        ax2.set_ylim([l_bound,u_bound])
        # ax2.set_aspect('equal', adjustable='box')
        
       
        for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label]+ax2.get_xticklabels() + ax2.get_yticklabels()):
            item.set_fontsize(8)
                
        plt.savefig(fig_name_str +'.pdf', bbox_inches='tight')
        plt.savefig(fig_name_str +'.png', bbox_inches='tight')
        
    def Trained_stress_scatter(self, Ytrue ,Ypred, R2, ext_label = None):
        #  ##### ERASE THAT 
        # Ytrue = 10*Ytrue
        # Ypred = 10*Ypred
        # ########################
        fig_name_str = self.PlotDir + '/Scatter_'+ self.ML 
        if ext_label is not None:
            fig_name_str = fig_name_str+ ext_label
        
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams.update({'font.size': 8})               
        fig1, ax1= plt.subplots(figsize=(1.625,3)) 
    
        
        # ax1.set_title('R2 = %.5f'%(R2))
        ax1.scatter(Ytrue[:,0:50], Ypred[:,0:50], s=2 ,color = "black")
        y_limits = ax1.get_ylim()
        x_limits = ax1.get_xlim()
        l_bound =  min(min(y_limits),min(x_limits))
        u_bound = max(max(y_limits),max(x_limits))
        #####################################
        # l_bound =  -80 
        # u_bound = 80 
        #
        #######################################
        ax1.set_xlabel('Target', fontsize=8)
        ax1.set_ylabel('Predicted', fontsize=8)
        # ax1.set_title('Shear force')
        ax1.set_xlim([l_bound,u_bound])
        ax1.set_ylim([l_bound,u_bound])
        # ax1.set_aspect('equal', adjustable='box')
        ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')
        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label]+ax1.get_xticklabels() + ax1.get_yticklabels()):
            item.set_fontsize(8)
       
        plt.axis('off')
        plt.savefig(fig_name_str +'DIR.png', bbox_inches='tight',  pad_inches=0,dpi=600)
        # plt.axis('off')
        # plt.savefig(fig_name_str +'DIR.png', bbox_inches='tight')
        
        plt.axis('on')
        ax1.cla()
        ax1.plot([l_bound,u_bound], [l_bound,u_bound],color='#E63946', linestyle='--', linewidth=1 )
        ax1.set_xlim([l_bound,u_bound])
        ax1.set_ylim([l_bound,u_bound])
        ################################################3
        Fx_ticks = (-60,0,60)
        ax1.set_xticks(Fx_ticks)
        ax1.set_yticks(Fx_ticks)
        ############################################################
        plt.savefig(fig_name_str +'DIR_axes.pdf', bbox_inches='tight') 
        
        
        #---------------------- NORMAL FORCE ------------------------------  
        fig2, ax2= plt.subplots(figsize=(1.625,3))    
        ax2.scatter(Ytrue[:,50:100],Ypred[:,50:100], s=2 ,color = "black")
        y_limits = ax2.get_ylim()
        x_limits = ax2.get_xlim()
        l_bound = min(min(y_limits),min(x_limits))
        u_bound = max(max(y_limits),max(x_limits))
        #####################################
        # l_bound =  0 
        # u_bound = 4 
        
        #######################################
        ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable='box')
        #ax2.set_title('Normal force')
        ax2.set_xlim([l_bound,u_bound])
        ax2.set_ylim([l_bound,u_bound])
        # ax2.set_aspect('equal', adjustable='box')
        
       
        for item in ([ax2.title, ax2.xaxis.label, ax2.yaxis.label]+ax2.get_xticklabels() + ax2.get_yticklabels()):
            item.set_fontsize(8)
            
        plt.axis('off')
        plt.savefig(fig_name_str +'NOR.png', bbox_inches='tight',  pad_inches=0, dpi=600)
        
        
        plt.axis('on')
        ax2.cla()
        ax2.plot([l_bound,u_bound], [l_bound,u_bound],color='#E63946', linestyle='--', linewidth=1 )
        ax2.set_xlim([l_bound,u_bound])
        ax2.set_ylim([l_bound,u_bound])
        #################################################3
        Fz_ticks = (0,30,60)
        ax2.set_xticks(Fz_ticks)
        ax2.set_yticks(Fz_ticks)
        ############################################################
        plt.savefig(fig_name_str +'NOR_axes.pdf', bbox_inches='tight') 
                
        
        
        
        
    
    def HolzParam_scatter(self, Ytrue ,Ypred, step, loss, ax, final = None):
        
        a_strings = ['a','af','as','afs']
        b_strings = ['b','bf','bs','bfs']
        for i in range(0,4):
            
            a = ax[0,i]
            b = ax[1,i]
        
            # Print
            param_index = 2*i
            a_limits = [min(1e3*Ytrue[:,param_index]), max(1e3*Ytrue[:,param_index])]
            
            a.cla()
            a.scatter(1e3*Ytrue[:,param_index], 1e3*Ypred[:,param_index], s=2 ,color = "orange")
            a.plot([a_limits[0],a_limits[1]], [a_limits[0],a_limits[1]],color='g', linestyle='-', linewidth=1 )
            a.set_xlabel('True [kPa]', fontsize=12)
            a.set_ylabel('Prediction [kPa]', fontsize=12)
            a.set_title(a_strings[i])
            # a.set_xlim(a_limits)
            # a.set_ylim(a_limits)
           
            
            if i == 0:
                a.text(0.55, 0.3, 'Epoch = %d' % step, fontdict={'size': 12, 'color':  'red'})
                a.text(0.55, 0.2, 'Val. Loss = %.4f' % loss, fontdict={'size': 12, 'color':  'red'})
            
            b_limits = [min(Ytrue[:,param_index+1]), max(Ytrue[:,param_index+1])]
            b.cla()
            b.set_xlabel('True [-]', fontsize=12)
            b.set_ylabel('Prediction [-]', fontsize=12)
            b.scatter(Ytrue[:,param_index+1], Ypred[:,param_index+1], s=2 ,color = "orange")
            b.plot([b_limits[0],b_limits[1]], [b_limits[0],b_limits[1]],color='g', linestyle='-', linewidth=1 )
            b.set_title(b_strings[i])
            # b.set_xlim(b_limits)
            # b.set_ylim(b_limits)
            
        fig_name_str = self.PlotDir + '/ScatterHolzParam_'+ self.ML
        plt.savefig(fig_name_str+'.png', bbox_inches='tight')
        
        if final:
            plt.savefig(fig_name_str+'.pdf', bbox_inches='tight')
            
    def TrainedHolzParam_scatter(self, Ytrue ,Ypred):
        
        fig, ax = plt.subplots(2,4,figsize=(7.5,3.2))
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams.update({'font.size': 8})
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = "Arial"
        a_strings = ['a','af','as','afs']
        b_strings = ['b','bf','bs','bfs']
        
        for i in range(0,4):
            
            a = ax[0,i]
            b = ax[1,i]
        
            # Print
            param_index = 2*i
            a_limits_1 = min([min(1e3*Ytrue[:,param_index]), min(1e3*Ypred[:,param_index])])
            a_limits_2 = max([max(1e3*Ytrue[:,param_index]), max(1e3*Ypred[:,param_index])])
            a_limits = [a_limits_1,a_limits_2]
            
            
            a.cla()
            a.scatter(1e3*Ytrue[:,param_index], 1e3*Ypred[:,param_index], s=1 ,color = "black")
            a.plot([0,a_limits[1]], [0,a_limits[1]],color='#E63946', linestyle='--', linewidth=1 )
            # a.set_xlabel('True [kPa]', fontsize=12)
            # a.set_ylabel('Prediction [kPa]', fontsize=12)
            # a.set_title(a_strings[i])
            # a.set_aspect(1.0/a.get_data_ratio(), adjustable='box')
            a.set_xlim(a_limits)
            a.set_ylim(a_limits)
            a.set_aspect('equal', adjustable='box')
            if i==0 or i==3:
                a.set_xticks((0,0.5,1))
                a.set_yticks((0,0.5,1))
            else:
                a.set_xticks((0,5,10))
                a.set_yticks((0,5,10))
                
           
        
            b_limits = [min(Ytrue[:,param_index+1]), max(Ytrue[:,param_index+1])]
            b_limits_1 = min([min(Ytrue[:,param_index+1]), min(Ypred[:,param_index+1])])
            b_limits_2 = max([max(Ytrue[:,param_index+1]), max(Ypred[:,param_index+1])])
            b_limits = [b_limits_1,b_limits_2]
            
            
            
            b.cla()
            # b.set_xlabel('True [-]', fontsize=12)
            # b.set_ylabel('Prediction [-]', fontsize=12)
            b.scatter(Ytrue[:,param_index+1], Ypred[:,param_index+1], s=1 ,color = "black")
            b.plot([b_limits[0],b_limits[1]], [b_limits[0],b_limits[1]],color='#E63946', linestyle='--', linewidth=1 )
            # b.set_title(b_strings[i])
            # b.set_aspect(1.0/a.get_data_ratio(), adjustable='box')
            b.set_xlim(b_limits)
            b.set_ylim(b_limits)
            b.set_xticks((0,5,10,15))
            b.set_yticks((0,5,10,15))
            b.set_aspect('equal', adjustable='box')
        
        fig_name_str = self.PlotDir + '/ScatterHolzParam_'+ self.ML
        plt.savefig(fig_name_str+'.png', bbox_inches='tight', dpi=900)
        
        #plt.savefig(fig_name_str+'.pdf', bbox_inches='tight')
            
            
    def OgdenParam_scatter(self, Ytrue ,Ypred, step, loss, ax1, ax2, final = None):
        
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams.update({'font.size': 8})               
        
        
        a_strings = ['a']
        b_strings = ['b']
        for i in range(0,1):
            
            a = ax1
            b = ax2
        
            # Print
            param_index = 2*i
            a_limits = [min(1e3*Ypred[:,param_index]), max(1e3*Ypred[:,param_index])]
            
            a.cla()
            a.scatter(1e3*Ytrue[:,param_index], 1e3*Ypred[:,param_index], s=2, color = "black")
            a.plot([a_limits[0],a_limits[1]], [a_limits[0],a_limits[1]],color='#E63946', linestyle='--', linewidth=1 )
            a.set_xlabel('True [kPa]', fontsize=12)
            a.set_ylabel('Prediction [kPa]', fontsize=12)
            a.set_title(a_strings[i])
            # a.set_xlim(a_limits)
            # a.set_ylim(a_limits)
            
            # ####################################
            # a.set_aspect('equal', adjustable='box')
            # a.set_xlim((0,2))
            # a.set_ylim((0,2))
            # a.set_xticks((0,1,2))
            # a.set_yticks((0,1,2))
            # ######################################
            # if i == 0:
            #     a.text(0.55, 0.3, 'Epoch = %d' % step, fontdict={'size': 12, 'color':  'red'})
            #     a.text(0.55, 0.2, 'Val. Loss = %.4f' % loss, fontdict={'size': 12, 'color':  'red'})
            
            b_limits = [min(Ypred[:,param_index+1]), max(Ypred[:,param_index+1])]
            b.cla()
            b.set_xlabel('True [-]', fontsize=12)
            b.set_ylabel('Prediction [-]', fontsize=12)
            b.scatter(Ytrue[:,param_index+1], Ypred[:,param_index+1], s=2 ,color = "black")
            b.plot([b_limits[0],b_limits[1]], [b_limits[0],b_limits[1]],color='#E63946', linestyle='--', linewidth=1 )
            b.set_title(b_strings[i])
            # b.set_xlim(b_limits)
            # b.set_ylim(b_limits)
            # ####################################
            # b.set_aspect('equal', adjustable='box')
            # b.set_xlim((0,30))
            # b.set_ylim((0,30))
            # b.set_xticks((0,15,30))
            # b.set_yticks((0,15,30))
            # ######################################
            
            
        fig_name_str = self.PlotDir + '/ScatterOgdenParam_'+ self.ML
        plt.savefig(fig_name_str+'.png', bbox_inches='tight')
        
        if final:
            plt.savefig(fig_name_str+'.pdf', bbox_inches='tight')
    
    
    
    def compute_error(self, Y_true, Y_pred, lc_stats, set_lbl):
        # Y is a 2-column array with [Fx,Fz] -> convert to column vector
        Y_true_vec = Y_true.flatten('F')
        Y_pred_vec = Y_pred.flatten('F')
                
        # calculate errors
        MAE = mean_absolute_error(Y_true_vec, Y_pred_vec)
        MAPE = mean_absolute_percentage_error(Y_true_vec, Y_pred_vec)
        R2 = 1.0 - r2_score(Y_true_vec, Y_pred_vec)
        
        # Append learning curve dictionary    
        lc_stats['MAE_' + set_lbl].append(MAE)
        lc_stats['MAPE_'+ set_lbl].append(MAPE)
        lc_stats['R2_'+ set_lbl].append(R2)
            
        return lc_stats
    
    def trained_NNs(self, Xscaler, model, num_train, HIDDEN_DIM, NUM_FEATURES, NUM_OUTPUT, Yscaler = None, DEPTH = None):
        
        NN_Settings =  {'Dim': HIDDEN_DIM, "Feat": NUM_FEATURES, "Out": NUM_OUTPUT, 'n_train': num_train, 'depth': DEPTH}
        
        Settings_name = self.MetaMdlDir + '/Torch_NN_' + str(num_train) + 'Settings.sav'
        Xscalername = self.MetaMdlDir + '/Torch_NN_' + str(num_train) + 'Xscaler.sav'
        Yscalername = self.MetaMdlDir + '/Torch_NN_' + str(num_train) + 'Yscaler.sav'
        NN_filename = self.MetaMdlDir + '/Torch_NN_' + str(num_train) +'.pt'
        
        # Export scale Settings,Scaler and Trained NN
        pickle.dump(NN_Settings, open(Settings_name, 'wb'))
        pickle.dump(Xscaler, open(Xscalername, 'wb'))
        torch.save(model.state_dict(), NN_filename)
        
        if Yscaler is not None:
            pickle.dump(Yscaler, open(Yscalername, 'wb'))
        
                
    def trained_GPRs(self, scaler, model, num_train, lscale_bounds):
        
        GPR_Settings =  {'LScaleBounds': lscale_bounds, 'n_train': num_train}
        
        Settings_name = self.MetaMdlDir + '/GPR_' + str(num_train) + 'Settings.sav'
        scalername = self.MetaMdlDir + '/GPR_' + str(num_train) + 'scaler.sav'
        GPR_filename = self.MetaMdlDir + '/GPR_' + str(num_train) +'.pt'
        
        # Export scale Settings,Scaler and Trained GPR
        pickle.dump(GPR_Settings, open(Settings_name, 'wb'))
        pickle.dump(scaler, open(scalername, 'wb'))
        pickle.dump(model, open(GPR_filename, 'wb'))
    
    def find_Best_Worst(self, Y_true ,Y_pred):
        
        R2_Raws = r2_score(Y_true.T, Y_pred.T, multioutput= 'raw_values')
        MAE_Raws = mean_absolute_error(Y_true.T, Y_pred.T, multioutput= 'raw_values')
        Worst_ID = np.argmin(R2_Raws)
        WorstScore = R2_Raws[Worst_ID]
        Best_ID = np.argmax(R2_Raws)
        BestScore = R2_Raws[Best_ID]
        
        return Worst_ID, WorstScore, Best_ID, BestScore, MAE_Raws

    def LossHist(self, Errors, n_bins):
        fig, axs = plt.subplots(tight_layout=True)

        # N is the count in each bin, bins is the lower-limit of the bin
        N, bins, patches = axs.hist(Errors, bins=n_bins)
        
        # We'll color code by height, but you could use any scalar
        fracs = N / N.max()
        
        # we need to normalize the data to 0..1 for the full range of the colormap
        norm = colors.Normalize(fracs.min(), fracs.max())
        
        # # Now, we'll loop through our objects and set the color of each accordingly
        # for thisfrac, thispatch in zip(fracs, patches):
        #     color = plt.cm.viridis(norm(thisfrac))
        #     thispatch.set_facecolor(color)
        
        axs.set_xlabel('MAE')
        axs.set_ylabel('Counts')
        
        fig_name_str = self.PlotDir + '/Hist_'+ self.ML 
            
        
        plt.savefig(fig_name_str+'.png', bbox_inches='tight')
        plt.savefig(fig_name_str+'.pdf', bbox_inches='tight')
        
        
        
        
    def RV_StressStrain(self, strain, exp_data, best_fit, TSLi):
        
        fig, ax_all = plt.subplots(5,3,figsize=(10,16))
       
        modeIDs = []
        modeIDs[0:1] = [1]*2
        modeIDs[2:3] = [2]*2
        modeIDs.append(3)    
        modeIDs[5:6] = [4]*2
        modeIDs[7:8] = [5]*2
        modeIDs.append(6)
        modeIDs[10:11] = [7]*2
        modeIDs[12:13] = [8]*2
        modeIDs.append(9)
        
        #plot each mode
        for i in range(0,15):
            col_ID = i//5
            row_ID = i - col_ID*5
            ax = ax_all[row_ID,col_ID]
            
            x_axis = strain[modeIDs[i]] 
            y_exp = exp_data[:,i]
            y_mdl = best_fit[:,i]
            
            ax.scatter(x_axis, y_exp)
            ax.plot(x_axis, y_mdl, 'r-')
            
        fig_name_str = self.PlotDir + '/Fit_TSL'+ str(TSLi) 
        
        plt.savefig(fig_name_str+'.png', bbox_inches='tight')
        plt.savefig(fig_name_str+'.pdf', bbox_inches='tight') 
        
        
    def Holz_StressStrain(self, Y_true, Y_pred, epoch, loss, star_set, ax_all, Final = False, ext_label = None):
        
        Y_star = Y_true[star_set,:]
        fX_star = Y_pred[star_set,:]
        
        true_data = np.reshape(Y_star,(50,15), order='F')
        NN_data = np.reshape(fX_star,(50,15), order='F')
               
        
        modeIDs = []
        modeIDs[0:1] = [1]*2
        modeIDs[2:3] = [2]*2
        modeIDs.append(3)    
        modeIDs[5:6] = [4]*2
        modeIDs[7:8] = [5]*2
        modeIDs.append(6)
        modeIDs[10:11] = [7]*2
        modeIDs[12:13] = [8]*2
        modeIDs.append(9)
        
        
        #plot each mode
        for i in range(0,15):
            col_ID = i//5
            row_ID = i - col_ID*5
            ax = ax_all[row_ID,col_ID]
            ax.cla()
            
            x = np.linspace(1,50,50)
            
            y_exp = true_data[:,i]
            y_mdl = NN_data[:,i]
            
            ax.scatter(x,y_exp)
            y_limits = ax.get_ylim()
            if not Final:
                ax.set_ylim(y_limits)
            
            ax.plot(x, y_mdl, 'r-')
            
        if ext_label is None:
            ext_label = ''
                  
            
        fig_name_str = self.PlotDir + '/Fit_TSL'+ str(star_set) + ext_label
        
        plt.savefig(fig_name_str+'.png', bbox_inches='tight')
        
        if Final:
            plt.savefig(fig_name_str+'.pdf', bbox_inches='tight') 
            
            
        
        
    def OgdenCustomcompute_error(self, Y_true, Y_pred, lc_stats, set_lbl):
        # Y is a 2-column array with [Fx,Fz] -> convert to column vector
        Y_true_vec_a = Y_true[:,0].flatten('F')
        Y_pred_vec_a = Y_pred[:,0].flatten('F')
        
        Y_true_vec_b = Y_true[:,1].flatten('F')
        Y_pred_vec_b = Y_pred[:,1].flatten('F')
                
        # calculate errors
        MAE = mean_absolute_error(Y_true_vec_a, Y_pred_vec_a)
        MAPE = mean_absolute_percentage_error(Y_true_vec_a, Y_pred_vec_a)
        R2_a = r2_score(Y_true_vec_a, Y_pred_vec_a)
        R2_b = r2_score(Y_true_vec_b, Y_pred_vec_b)
        
        # Append learning curve dictionary    
        lc_stats['MAE_' + set_lbl].append(MAE)
        lc_stats['MAPE_'+ set_lbl].append(R2_b)
        lc_stats['R2_'+ set_lbl].append(R2_a)
            
        return lc_stats
    
    
    def HolzCustomcompute_error(self, Y_true, Y_pred, lc_stats, set_lbl):
        # Y is a 2-column array with [Fx,Fz] -> convert to column vector
        Y_true_vec_a = Y_true[:,0].flatten('F')
        Y_pred_vec_a = Y_pred[:,0].flatten('F')
        
        Y_true_vec_b = Y_true[:,1].flatten('F')
        Y_pred_vec_b = Y_pred[:,1].flatten('F')
        
        Y_true_vec_af = Y_true[:,2].flatten('F')
        Y_pred_vec_af = Y_pred[:,2].flatten('F')
        
        Y_true_vec_bf = Y_true[:,3].flatten('F')
        Y_pred_vec_bf = Y_pred[:,3].flatten('F')
        
        Y_true_vec_as = Y_true[:,4].flatten('F')
        Y_pred_vec_as = Y_pred[:,4].flatten('F')
        
        Y_true_vec_bs = Y_true[:,5].flatten('F')
        Y_pred_vec_bs = Y_pred[:,5].flatten('F')
        
        Y_true_vec_afs = Y_true[:,6].flatten('F')
        Y_pred_vec_afs = Y_pred[:,6].flatten('F')
        
        Y_true_vec_bfs = Y_true[:,7].flatten('F')
        Y_pred_vec_bfs = Y_pred[:,7].flatten('F')
                
        # calculate errors
        MAE = mean_absolute_error(Y_true_vec_a, Y_pred_vec_a)
        MAPE = mean_absolute_percentage_error(Y_true_vec_a, Y_pred_vec_a)
        R2_a = r2_score(Y_true_vec_a, Y_pred_vec_a)
        R2_b = r2_score(Y_true_vec_b, Y_pred_vec_b)
        
        R2_af = r2_score(Y_true_vec_af, Y_pred_vec_af)
        R2_bf= r2_score(Y_true_vec_bf, Y_pred_vec_bf)
        
        R2_as = r2_score(Y_true_vec_as, Y_pred_vec_as)
        R2_bs = r2_score(Y_true_vec_bs, Y_pred_vec_bs)
        
        R2_afs = r2_score(Y_true_vec_afs, Y_pred_vec_afs)
        R2_bfs = r2_score(Y_true_vec_bfs, Y_pred_vec_bfs)
        
        # Append learning curve dictionary    
        lc_stats['MAE_' + set_lbl].append(MAE)
        lc_stats['MAPE_'+ set_lbl].append(R2_b)
        lc_stats['R2_'+ set_lbl].append(R2_a)
            
        return lc_stats
        
        