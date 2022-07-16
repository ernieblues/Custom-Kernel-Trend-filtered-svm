# Ernest P. Guagliata
# COT 6970 Master's Thesis
# Advisor: Michael DeGiorgio Ph.D.

import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns

from IPython.display import display
from joblib import dump, load
from math import log, pi
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from os.path import exists
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Test Support Vector Machine
class TestSVM:

    # constants
    C2 = [] # penalty coefficients
    for i in range(-4, 11):
        C2.append(pow(10, i))
        
    palette_kernels = sns.color_palette("Set1", 4) # plot colors for kernels
    palette_ntrain = sns.color_palette("Set2", 4) # plot colors for n_train
    palette_ntrain2 = np.repeat(np.array(palette_ntrain), 2, axis=0) # plot colors for n_train duplicated for dashed lines
    palette_c2 = np.repeat(np.array(sns.color_palette("Dark2", 5)), 3, axis=0) # plot colors for C2 triplicated for dashed lines
    palette_reg_targets = sns.color_palette("Set2", 7)[4:] # plot colors for 3 regression targets
    palette_prob = sns.color_palette("Set3", 5)[3:] # plot colors for probabilities

    # constructor
    def __init__(self, **kwargs):
        
        # variables
        dataset_temp = kwargs.pop('dataset') # dataset temp
        self.dataset = np.array(dataset_temp) # dataset
        self.dataset_name = dataset_temp.name # dataset name
        self.n_train_list = [int(i) for i in kwargs.pop('n_train_list', '0,0,0,0').split(",")] # list of number of training observations
        self.n = self.dataset.shape[0] # number of observations
        
        if self.dataset_name[12:23] == 'Regression' and self.dataset_name[23:] == '':
            self.p = self.dataset.shape[1] - 3 # number of features
        else:
            self.p = self.dataset.shape[1] - 1 # number of features
        
        self.Ip = np.identity(self.p) # identity matrix
        
        self.debug = kwargs.pop('debug', 'false') # debug mode
        self.file_in = kwargs.pop('file_in', '') # path and file name of input file to process
        self.file_out = kwargs.pop('file_out', '') # path and file name of output file
        self.results = kwargs.pop('results', '') # directory for output, program data, and models
        self.input_path = kwargs.pop('input_path', '') # input path
        self.model = kwargs.pop('model', '') # name of trained model
        self.models = kwargs.pop('models', []) # list of trained models
        
        # random_state number
        if self.results != '':
            if not exists('{}program_data/random_state.csv'.format(self.results)):
                pd.DataFrame(data=[random.randrange(1, 100)]).to_csv('{}program_data/random_state.csv'.format(self.results), index=False)
            
            self.random_state = pd.read_csv('{}program_data/random_state.csv'.format(self.results)).iloc[0,0]

    ##############################################
    ############## Helper Functions ##############
    ##############################################
    
    # Beta Plots Y Axis Limits
    def beta_ylim(self):
        if self.dataset_name[23:] == '':
            return [-0.3, 0.4] 
        if self.dataset_name[23:] == '-a':
            return [0, 0.2]
        if self.dataset_name[23:] == '-f':
            return [-0.16, -0.03]
        if self.dataset_name[23:] == '-ws':
            return [-0.2, 0.25]
            
    # Calibration Curve
    def calibration_curve(self, y_true, y_prob, n_bins):
    
        # variables
        prob_pred = np.array([]) # mean probability in each bin
        prob_true = np.array([]) # fraction of positives in each bin
        
        for j in [k/1000 for k in range(951)]:
            indices = [] # indices of observations in the bucket
            
            if j > 0.8:
                break
            
            # get indicies of observations in the probability bucket
            for i in range(len(y_prob)):
                if j <= y_prob[i] and y_prob[i] <= j + 0.2:
                    indices.append(i)
            
            # fraction of class 1 predictions in the bucket
            count = 0
            
            for i in range(len(indices)):
                if y_true[indices[i]] == 1:
                    count += 1
                    
            if len(indices) >= 10:
                prob_pred = np.append(prob_pred, np.mean(y_prob[indices])) # mean probability in the bin
                prob_true = np.append(prob_true, count/len(indices)) # fraction of positives in the bin
                
        return prob_true, prob_pred
    
    # Difference Matrix
    def diff_matrix(self, p, d):
    
        # identity matrix
        D = np.identity(p)
        
        # finite difference matrix
        L = np.eye(p-1, p, dtype=int, k=1) - np.eye(p-1, p, dtype=int)
        
        for k in range(1, d+1):
            D = np.dot(L[0:p-k, 0:p-k+1], D)
        
        return D
    
    # Gram Matrix
    def gram_matrix(self, U, V, C2, d):
        # C2 penalty coefficient
        # d degree of the difference matrix
            
        # variables
        p = U.shape[1] # number of features
        Ip = np.identity(p) # identity matrix
        D = self.diff_matrix(p, d).astype(np.float64)
        
        # G = X(Ip + C2*D^T*D)^-1*X^T
        G = np.dot(U, np.linalg.inv(Ip + C2*np.dot(D.T, D)))
        
        return np.dot(G, V.T)

    # Grayscale cmap
    def grayscale_cmap(self, cmap):
        """Return a grayscale version of the given colormap"""
        cmap = plt.cm.get_cmap(cmap)
        colors = cmap(np.arange(cmap.N))
    
        # convert RGBA to perceived grayscale luminance
        # cf. http://alienryderflex.com/hsp.html
        RGB_weight = [0.299, 0.587, 0.114]
        luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
        colors[:, :3] = luminance[:, np.newaxis]
            
        return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)
    
    # Hinge Loss Error
    def hinge_loss(self, X, y, beta_0, beta):
        
        # f(x_i) = beta_0 + x_i^T * beta
        f_x = beta_0 + np.dot(X, beta)
        
        # error_k = (sum(i = 1 to n_k) of max{0, 1 - y_i * f(x_i)})/n
        hinge_loss_error = np.sum(np.maximum(0, 1-y*f_x))/X.shape[0]
        
        return hinge_loss_error

    # Kernel Dictionary
    def kernel_dic(self, kernel, **kwargs):
    
        # variables
        c = kwargs.pop('C', 1) # regularization coefficient

        # kernel dictionary
        kernel_dic = {
        'svc_custom': svm.SVC(kernel='precomputed', probability=True, random_state=self.random_state),
        'svr_custom': svm.SVR(kernel='precomputed', epsilon=1e-1, tol=1e-3),
        'svc_linear': svm.SVC(kernel='linear', probability=True, random_state=self.random_state),
        'svr_linear': svm.SVR(kernel='linear'),
        'svc_rbf': svm.SVC(kernel='rbf', probability=True, random_state=self.random_state),
        'svr_rbf': svm.SVR(kernel='rbf')
        }

        return kernel_dic[kernel]
    
    # Convert Number to ASCII
    def to_ascii(self, num):

        # exponent dictionary
        exp_dic = {
            -4: '\u207B\u2074',
            -3: '\u207B\u00b3',
            -2: '\u207B\u00b2',
            -1: '\u207B\u00b9',
            0: '\u2070',
            1: '\u00b9',
            2: '\u00b2',
            3: '\u00b3',
            4: '\u2074',
            5: '\u2075',
            6: '\u2076',
            7: '\u2077',
            8: '\u2078',
            9: '\u2079',
            10: '\u00b9\u2070'
        }
        
        return '10{}'.format(exp_dic[round(log(num, 10))])
    
    # Trend Filtered Support Vector Machine
    def TrendFilteredSVM(self, C2, d):
        # C2 penalty coefficient
        # d degree of the difference matrix
        
        def nest_TrendFilteredSVM(U, V):
            
            # variables
            p = U.shape[1] # number of features
            Ip = np.identity(p) # identity matrix
            D = self.diff_matrix(p, d)

            # G = X(Ip + C2*D^T*D)^-1*X^T
            G = np.dot(U, np.linalg.inv(Ip + C2*np.dot(D.T, D)))

            return np.dot(G, V.T)
        
        return nest_TrendFilteredSVM
    
    ####################################################
    ##################### Beta #########################
    ####################################################
    def beta(self, **kwargs):

        # variables
        kernel = kwargs.pop('kernel', 'linear') # kernel
        num_deg = kwargs.pop('num_deg', 2) # number of difference matrices
        degrees = range(1, num_deg + 1) # degrees of the difference matrix
        
        # for each dataset with n training observations
        for n_train in self.n_train_list:
        
            # double n_train for classifiers
            if self.dataset_name[-14:] == 'Classification':
                n_train = 2*n_train
            
            # dataset
            X_train = self.dataset[list(range(0, int(n_train/2))) + list(range(int(self.n/2), int((self.n/2)+(n_train/2)))), :-1]
            y_train = self.dataset[list(range(0, int(n_train/2))) + list(range(int(self.n/2), int((self.n/2)+(n_train/2)))), -1]
            
            # standardize data
            ss = StandardScaler()
            X_train = ss.fit_transform(X_train)
            
            if self.dataset_name[12:23] == 'Regression':
                y_train = ss.fit_transform(y_train.reshape(-1, 1)).squeeze()
                
            # for each difference matrix (custom kernel only)
            for d in degrees:
                c2_x_beta = pd.DataFrame() # C2 x β̂  matrix
                
                # for each penalty coefficient C2
                for c2 in self.C2:

                    # scikit learn Support Vector Machines
                    slm = self.kernel_dic(kernel)
                    
                    if kernel[4:] == 'custom':
                        slm.fit(self.gram_matrix(X_train, X_train, c2, d), y_train)
                    else:
                        slm.fit(X_train, y_train)

                    # support vectors
                    X_support = X_train[slm.support_, :]
                    y_support = y_train[slm.support_]

                    # beta
                    alpha = np.abs(slm.dual_coef_.squeeze()) # α (dual coefficient)
                    if kernel[4:] == 'custom':
                        # β̂  = (Ip + C2 D^T D)^-1 * X^T(α * y)
                        D = self.diff_matrix(self.p, d) # difference matrix
                        beta = np.linalg.inv(self.Ip + c2*D.T@D)@X_support.T@(alpha*y_support)
                    else:
                        # β̂  = X^T(α * y)
                        beta = X_support.T@(alpha*y_support)

                    # l2 norm of beta
                    beta = beta/np.linalg.norm(beta, 2)

                    # C2 x β̂  matrix
                    c2_x_beta = c2_x_beta.append(pd.Series(beta), ignore_index=True)

                    # if not using c2
                    if kernel[4:] != 'custom':
                        break

                # label c2_x_beta rows
                row_labels = []
                if kernel[4:] == 'custom':
                    for i in range(len(self.C2)):
                        row_labels.append(self.to_ascii(self.C2[i]))
                        
                    c2_x_beta.index = row_labels
                else:
                    c2_x_beta = c2_x_beta.rename(index={0: 'C=1'})

                # label c2_x_beta columns
                col_labels = []
                for i in range(1, c2_x_beta.shape[1] + 1):
                    col_labels.append('β{}'.format(i))
                c2_x_beta.columns = col_labels

                # if not using d
                if kernel[4:] != 'custom':
                    break
                    
                # beta plot
                rcparams = {"lines.linewidth": 1, 'axes.labelsize': 10, 'legend.title_fontsize': 9, 'legend.fontsize': 9,
                            'xtick.labelsize': 12, 'ytick.labelsize': 12}
                sns.set(style='white', rc=rcparams)
                
                if kernel[:3] == 'svc':
                    gs_kw = {'left': 0.12, 'right': 0.97, 'top': 0.97, 'bottom': 0.06}
                elif kernel[:3] == 'svr':
                    gs_kw = {'left': 0.14, 'right': 0.97, 'top': 0.97, 'bottom': 0.06}
                    
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), gridspec_kw=gs_kw)
                
                x_tick_labels = np.arange(start=1, stop=len(col_labels)+1, step=int((len(col_labels)-1)/10)) # set number of x ticks to 11
                
                ax.set(xlim=(1,self.p), xticks=range(1,self.p+1,int((self.p-1)/10)), xticklabels=x_tick_labels)
                
                dashes = ['', (2,2), (4,4), '', (2,2), (4,4), '', (2,2), (4,4), '', (2,2), (4,4), '', (2,2), (4,4)]
                ax = sns.lineplot(data=c2_x_beta.T, palette=self.palette_c2, dashes=dashes, legend=False)
                
                #plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor") # rotate tick labels
                #plt.legend(title='C\u2082', bbox_to_anchor=(1.012, 0.85), loc='upper left', borderaxespad=0) # legend title and location
                
                if kernel[:3] == 'svc':
                    plt.savefig('{}output/beta_{}_d={}_{}.eps'.format(self.results, kernel[:3], d, int(n_train/2)), format='eps', transparent=False)
                elif kernel[:3] == 'svr':
                    plt.savefig('{}output/beta_{}_{}_d={}_{}.eps'.format(self.results, kernel[:3], self.dataset_name[23:], d, n_train), format='eps', transparent=False)
                    
                plt.close()
    
    #######################################################################
    ############## Beta (optimal regularization parameter) ################
    #######################################################################
    def beta_opt(self, **kwargs):

        # variables
        param = kwargs.pop('param') # kernel and d (for custom kernel)
        
        # for each dataset with n training observations
        for n_train in self.n_train_list:
        
            # double n_train for classifiers
            if self.dataset_name[-14:] == 'Classification':
                n_train = 2*n_train
                
            param_x_beta = pd.DataFrame() # param x β̂  matrix
            
            # dataset
            X_train = self.dataset[list(range(0, int(n_train/2))) + list(range(int(self.n/2), int((self.n/2)+(n_train/2)))), :-1]
            y_train = self.dataset[list(range(0, int(n_train/2))) + list(range(int(self.n/2), int((self.n/2)+(n_train/2)))), -1]

            # standardize data
            ss = StandardScaler()
            X_train = ss.fit_transform(X_train)
                
            if self.dataset_name[12:23] == 'Regression':
                y_train = ss.fit_transform(y_train.reshape(-1, 1)).squeeze()

            # for each kernel, regularization parameters, and d (for custom kernel)
            for kernel, d in param:
                
                # get C2 from the cross validation data
                if kernel[4:] == 'custom':
                    if kernel[:3] == 'svc':
                        cv_c2_x_kernel = pd.read_csv('{}program_data/cv_{}.csv'.format(self.results, kernel)) # import cross validation data
                        idx = cv_c2_x_kernel['n_train={} d={}'.format(int(n_train/2), d)].idxmin()
                    elif kernel[:3] == 'svr':
                        cv_c2_x_kernel = pd.read_csv('{}program_data/cv_{}_{}.csv'.format(self.results, kernel, self.dataset_name[23:])) # import cross validation data
                        idx = cv_c2_x_kernel['n_train={} d={}'.format(n_train, d)].idxmin()
                        
                    c2 = self.C2[idx]

                # scikit learn Support Vector Machines
                slm = self.kernel_dic(kernel)
                    
                if kernel[4:] == 'custom':
                    slm.fit(self.gram_matrix(X_train, X_train, c2, d), y_train)
                else:
                    slm.fit(X_train, y_train)

                # support vectors
                X_support = X_train[slm.support_, :]
                y_support = y_train[slm.support_]

                # beta
                alpha = np.abs(slm.dual_coef_.squeeze()) # α (dual coefficient)
                if kernel[4:] == 'custom':
                    # β̂  = (Ip + C2 D^T D)^-1 * X^T(α * y)
                    D = self.diff_matrix(self.p, d) # difference matrix
                    beta = np.linalg.inv(self.Ip + c2*D.T@D)@X_support.T@(alpha*y_support)
                else:
                    # β̂  = X^T(α * y)
                    beta = X_support.T@(alpha*y_support)

                # l2 norm of beta
                beta = beta/np.linalg.norm(beta, 2)
                
                # param x β̂  matrix
                if kernel[4:] == 'custom':
                    param_x_beta = param_x_beta.append(pd.Series(beta, name='{} C\u2082={} d={}'.format(kernel[4:], self.to_ascii(c2), d)))
                else:
                    param_x_beta = param_x_beta.append(pd.Series(beta, name='{}'.format(kernel[4:])))

            # label param_x_beta columns
            col_labels = []
            for i in range(1, param_x_beta.shape[1] + 1):
                col_labels.append('β{}'.format(i))
            param_x_beta.columns = col_labels
                    
            # beta optimal plot
            rcparams = {"lines.linewidth": 1, 'axes.labelsize': 10, 'legend.fontsize': 9, 'xtick.labelsize': 12, 'ytick.labelsize': 12}
            sns.set(style='white', rc=rcparams)
            
            if kernel[:3] == 'svc':
                gs_kw = {'left': 0.12, 'right': 0.97, 'top': 0.97, 'bottom': 0.06}
            elif kernel[:3] == 'svr':
                gs_kw = {'left': 0.14, 'right': 0.97, 'top': 0.97, 'bottom': 0.06}
                
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), gridspec_kw=gs_kw)
            
            x_tick_labels = np.arange(start=1, stop=len(col_labels)+1, step=int((len(col_labels)-1)/10)) # set number of x ticks to 11
            ax.set(xlim=(1,self.p), xticks=range(1,self.p+1,int((self.p-1)/10)), xticklabels=x_tick_labels)
            ax = sns.lineplot(data=param_x_beta.T, palette=self.palette_kernels, dashes=False, legend=False)
            
            #plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor") # rotate tick labels
            
            if kernel[:3] == 'svc':
                plt.savefig('{}output/beta_opt_{}_{}.eps'.format(self.results, kernel[:3], int(n_train/2)), format='eps', transparent=False)
            elif kernel[:3] == 'svr':
                plt.savefig('{}output/beta_opt_{}_{}_{}.eps'.format(self.results, kernel[:3], self.dataset_name[23:], n_train), format='eps', transparent=False)
            
            plt.close()
    
    ######################################
    ############ Chromosome 2 ############
    ######################################
    def chrom2(self, **kwargs):
        
        #variables
        chrom2_results = pd.DataFrame(columns=['Chromosome', 'Position']) # chromosome 2 results
        chrom2_mean_and_median = pd.DataFrame({'Value': ['mean_all', 'median_all', 'mean_lct', 'median_lct']}) # chromosome 2 mean and median
        n_train = 10000 # number of training observations
        X_test = np.empty((0, 101))
        
        # dataset
        X_train = self.dataset[list(range(0, int(n_train/2))) + list(range(int(self.n/2), int((self.n/2)+(n_train/2)))), :-3]
        y_train_ws = self.dataset[list(range(0, int(n_train/2))) + list(range(int(self.n/2), int((self.n/2)+(n_train/2)))), -3:-2]
        y_train_a = self.dataset[list(range(0, int(n_train/2))) + list(range(int(self.n/2), int((self.n/2)+(n_train/2)))), -2:-1]
        y_train_f = self.dataset[list(range(0, int(n_train/2))) + list(range(int(self.n/2), int((self.n/2)+(n_train/2)))), -1]
        
        # import parsed chromosome 2 data: Position, Computation
        chrome_data = pd.read_csv('{}ceu2.csv'.format(self.input_path))
            
        # debug mode
        if self.debug == 'true':
            end = 55
        else:
            end = chrome_data.shape[0]-50
        
        for i in range(50, end):
            position = chrome_data.loc[i, 'POS']
            index = 0
            start = True
            
            # if chromosome 2 is between positions 134,000,000 and 138,000,000
            if 134e6 <= position and position <= 138e6:
                # chromosome 2 results: Chromosome, Position
                chrom2_results = chrom2_results.append({'Chromosome': 2, 'Position': position}, ignore_index=True)
                
                # enter: 101 Features
                X_test = np.append(X_test, [chrome_data.iloc[i-50:i+51, 1]], axis=0)
                
                # start = 136,545,415 and stop = 136,594,750 
                if start == True and 136545415 <= position:
                    start_index = index
                    start = False
                elif position <= 136594750:
                    stop_index = index
                    
                index += 1
            
        # standardize data
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.transform(X_test)
        
        for model in self.models:
            
            # load regressor model
            slm = load('{}models/{}.joblib'.format(self.results, model)) # load model
            models = pd.read_csv('{}models/models.csv'.format(self.results)) # read models parameters table
            idx = models.index[models['Model']==model] # get index of model
            c2 = int(models.loc[idx, 'C\u2082'])
            d = int(models.loc[idx, 'd'])
        
            # predict target
            y_pred = slm.predict(self.gram_matrix(X_test, X_train, c2, d)) # y_pred
            
            # histogram margins
            rcparams = {'axes.labelsize': 13, 'xtick.labelsize': 13, 'ytick.labelsize': 13, 'xtick.bottom': True, 'ytick.left': True}
            sns.set(style='white', rc=rcparams)
            gs_kw = {'left': 0.15, 'right': 0.98, 'top': 0.98, 'bottom': 0.11}
        
            if model[4:5] == 'a': # Predicted Selection Coefficient
                
                # reverse process targets
                y_pred = y_pred*np.std(y_train_a) + np.mean(y_train_a) # unstandardize
                y_pred = np.power(10, y_pred) # antilog10
                y_pred = y_pred / (2*10000) # population-scale parameter
                chrom2_results['PredictedSelectionCoefficient'] = y_pred
                
                # mean for all data and the LCT region
                mean_all = round(np.mean(chrom2_results['PredictedSelectionCoefficient']), 3)
                median_all = round(np.median(chrom2_results['PredictedSelectionCoefficient']), 3)
                mean_lct = round(np.mean(chrom2_results.loc[start_index:stop_index, 'PredictedSelectionCoefficient']), 3)
                median_lct = round(np.median(chrom2_results.loc[start_index:stop_index, 'PredictedSelectionCoefficient']), 3)
                chrom2_mean_and_median['PredictedSelectionCoefficient'] = [mean_all, median_all, mean_lct, median_lct]
                
                # histogram
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), gridspec_kw=gs_kw)
                ax.axvline(x=mean_lct, linewidth=2, dashes=(8,4), color='red')
                sns.histplot(data=y_pred, color=self.palette_reg_targets[0])
                ax.set(xlabel='Selection Coefficient (s)', xticks=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
                plt.savefig('{}output/chrom2_a.eps'.format(self.results))
                plt.close()
                
            elif model[4:5] == 'f': # Predicted Frequency of Beneficial Mutation
                
                # reverse process targets
                y_pred = y_pred*np.std(y_train_f) + np.mean(y_train_f) # unstandardize
                y_pred = np.power(10, y_pred) # antilog10
                chrom2_results['PredictedFrequency'] = y_pred
                
                # mean and median for all data and the LCT region
                mean_all = round(np.mean(chrom2_results['PredictedFrequency']), 3)
                median_all = round(np.median(chrom2_results['PredictedFrequency']), 3)
                mean_lct = round(np.mean(chrom2_results.loc[start_index:stop_index, 'PredictedFrequency']), 3)
                median_lct = round(np.median(chrom2_results.loc[start_index:stop_index, 'PredictedFrequency']), 3)
                chrom2_mean_and_median['PredictedFrequency'] = [mean_all, median_all, mean_lct, median_lct]
                
                # histogram
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), gridspec_kw=gs_kw)
                ax.axvline(x=mean_lct, linewidth=2, dashes=(8,4), color='red')
                sns.histplot(data=y_pred, color=self.palette_reg_targets[1])
                ax.set(xlabel='Frequency of Mutation (f)', xticks=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
                plt.savefig('{}output/chrom2_f.eps'.format(self.results))
                plt.close()
                
            elif model[4:6] == 'ws': # Predicted Time Sweep Completed
                # reverse process targets
                y_pred = y_pred*np.std(y_train_ws) + np.mean(y_train_ws) # unstandardize
                y_pred = np.power(10, y_pred) # antilog10
                y_pred = y_pred * 4*10000 * 25 # coalescent time unit and time in generations
                chrom2_results['PredictedTime'] = y_pred
                
                # mean for all data and the LCT region
                mean_all = round(np.mean(chrom2_results['PredictedTime']))
                median_all = round(np.median(chrom2_results['PredictedTime']))
                mean_lct = round(np.mean(chrom2_results.loc[start_index:stop_index, 'PredictedTime']))
                median_lct = round(np.median(chrom2_results.loc[start_index:stop_index, 'PredictedTime']))
                chrom2_mean_and_median['PredictedTime'] = [mean_all, median_all, mean_lct, median_lct]
                
                # histogram
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), gridspec_kw=gs_kw)
                ax.axvline(x=mean_lct, linewidth=2, dashes=(8,4), color='red')
                sns.histplot(data=y_pred, color=self.palette_reg_targets[2])
                ax.set(xlabel='Time Sweep Completed (t)', xlim=(13500, 22400))
                plt.savefig('{}output/chrom2_ws.eps'.format(self.results))
                plt.close()
                
        # save chromosome 2 data
        chrom2_results.to_csv('{}/output/chrom2_results.csv'.format(self.results), index=False)
        chrom2_mean_and_median.to_csv('{}/output/chrom2_mean_and_median.csv'.format(self.results), index=False)
    
    ######################################
    ########## Cross Validation ##########
    ######################################
    def cv(self, **kwargs):

        # variables
        kernel = kwargs.pop('kernel', 'linear') # kernel
        num_deg = kwargs.pop('num_deg', 2) # number of difference matrices
        degrees = range(1, num_deg + 1) # degrees of the difference matrix
        #cv_c2_x_kernel = pd.DataFrame(index=[self.to_ascii(self.C2[i]) for i in range(len(self.C2))])
        cv_c2_x_kernel = pd.DataFrame(index=self.C2)
        cv_c2_x_kernel.index.name='C\u2082'
        
        # for each dataset with n training observations
        for n_train in self.n_train_list:
        
            # double n_train for classifiers
            if self.dataset_name[-14:] == 'Classification':
                n_train = 2*n_train
        
            # dataset (train and validation)
            train_val = self.dataset[list(range(0, int(n_train/2))) + list(range(int(self.n/2), int((self.n/2)+(n_train/2)))), :]

            # scikit learn KFold
            kf = KFold(n_splits=10, shuffle=True, random_state=self.random_state)

            # for each degrees of the difference matrix
            for d in degrees:
                cv5_error_x_c2 = []
                
                # for each penalty coefficient C2
                for c2 in self.C2:
                    kfold_error = np.array([])

                    if kernel[4:] != 'custom':
                        c = c2

                    # for each K-fold
                    for train_indices, validate_indices in kf.split(train_val):
                        X_train = train_val[train_indices, :-1]
                        y_train = train_val[train_indices, -1]
                        X_validate = train_val[validate_indices, :-1]
                        y_validate = train_val[validate_indices, -1]
                        
                        # standardize data
                        ss = StandardScaler()
                        X_train = ss.fit_transform(X_train)
                        X_validate = ss.fit_transform(X_validate)
                        
                        if self.dataset_name[12:23] == 'Regression':
                            y_train = ss.fit_transform(y_train.reshape(-1, 1)).squeeze()
                            y_validate = ss.transform(y_validate.reshape(-1, 1)).squeeze()
                        
                        # scikit learn Support Vector Machines
                        if kernel[4:] == 'custom':
                            slm = self.kernel_dic(kernel)
                            slm.fit(self.gram_matrix(X_train, X_train, c2, d), y_train)
                            y_hat = slm.predict(self.gram_matrix(X_validate, X_train, c2, d))
                        else:
                            slm = self.kernel_dic(kernel)
                            slm.fit(X_train, y_train)
                            y_hat = slm.predict(X_validate)

                        # support vectors
                        X_support = X_train[slm.support_, :]
                        y_support = y_train[slm.support_]

                        # beta
                        beta_0 = slm.intercept_
                        alpha = np.abs(slm.dual_coef_.squeeze()) # α (dual coefficient)
                        if kernel[4:] == 'custom':
                            # β̂  = (Ip + C2 D^T D)^-1 * X^T(α * y)
                            D = self.diff_matrix(self.p, d) # difference matrix
                            beta = np.linalg.inv(self.Ip + c2*D.T@D)@X_support.T@(alpha*y_support)
                        else:
                            # β̂  = X^T(α * y)
                            beta = X_support.T@(alpha*y_support)

                        # error
                        if kernel[:3] == 'svc':
                            # hinge loss error
                            hle = self.hinge_loss(X_validate, y_validate, beta_0, beta)
                            kfold_error = np.append(kfold_error, hle)
                        elif kernel[:3] == 'svr':
                            # mean square error
                            mse = mean_squared_error(y_validate, y_hat)
                            kfold_error = np.append(kfold_error, mse)
                            
                    # CV(5) Error
                    cv5_error = np.sum(kfold_error) / kfold_error.shape[0]
                    
                    # CV(5) Error x C2 matrix
                    cv5_error_x_c2.append(cv5_error)
                
                # C2 x kernel Error matrix
                if kernel == 'svc_custom':
                    cv_c2_x_kernel['n_train={} d={}'.format(int(n_train/2), d)] = pd.Series(cv5_error_x_c2).values
                elif kernel == 'svr_custom':
                    cv_c2_x_kernel['n_train={} d={}'.format(n_train, d)] = pd.Series(cv5_error_x_c2).values
                else:
                    cv_c2_x_kernel['n_train={}'.format(n_train)] = pd.Series(cv5_error_x_c2).values
                
                # if not using d
                if kernel[4:] != 'custom':
                    break
                    
        # write cross validation error to csv
        if kernel[:3] == 'svc':
            cv_c2_x_kernel.to_csv('{}program_data/cv_{}.csv'.format(self.results, kernel))
        elif kernel[:3] == 'svr':
            cv_c2_x_kernel.to_csv('{}program_data/cv_{}_{}.csv'.format(self.results, kernel, self.dataset_name[23:]))
        
        # cross validation plot
        rcparams = {"lines.linewidth": 2, 'axes.labelsize': 14, 'legend.fontsize': 11, 'xtick.labelsize': 12, 'ytick.labelsize': 12}
        sns.set(style='white', rc=rcparams)
        
        if kernel[:3] == 'svc':
            gs_kw = dict(left=0.16, right=0.96, top=0.97, bottom=0.12)
        elif kernel[:3] == 'svr':
            gs_kw = dict(left=0.14, right=0.96, top=0.97, bottom=0.12)
            
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), gridspec_kw=gs_kw)
        
        dashes = ['', (2,2), '', (2,2), '', (2,2), '', (2,2)]
        
        if kernel[:3] == 'svc':
            ax.set(xlabel='Regularization Parameter (C\u2082)', ylabel='Hinge Loss', xlim=(self.C2[0], self.C2[-1]))
        elif kernel[:3] == 'svr':
            ax.set(xlabel='Regularization Parameter (C\u2082)', ylabel='Mean Squared Error', xlim=(self.C2[0], self.C2[-1]))
            
        cv_c2_x_kernel = (cv_c2_x_kernel - cv_c2_x_kernel.min())/(cv_c2_x_kernel.max() - cv_c2_x_kernel.min()) # ylim=(0,1)
        
        ax = sns.lineplot(data=cv_c2_x_kernel, palette=self.palette_ntrain2, dashes=dashes, legend=False)
        ax.set_xscale("log")
        
        if kernel[:3] == 'svc':
            plt.savefig('{}output/cv_{}.eps'.format(self.results, kernel), format='eps', transparent=False)
        elif kernel[:3] == 'svr':    
            plt.savefig('{}output/cv_{}{}.eps'.format(self.results, kernel, self.dataset_name[23:]), format='eps', transparent=False)
            
        plt.close()
            
    ######################################################################
    ######################## Mean Squared Error ##########################
    ######################################################################
    def mse(self, param):
    
        # variables
        mse_table = pd.DataFrame(index=self.n_train_list) # mean squared error table
        mse_table.index.name='n_train'
        n_test = 1000
        
        # for each kernel and d
        for kernel, d in param:
            mse = [] # mean squared error for each n_train
        
            # for each n_train
            for n_train in self.n_train_list:
    
                # dataset
                X_train = self.dataset[list(range(0, int(n_train/2))) + list(range(int(self.n/2), int((self.n/2)+(n_train/2)))), :-1]
                y_train = self.dataset[list(range(0, int(n_train/2))) + list(range(int(self.n/2), int((self.n/2)+(n_train/2)))), -1]
                X_test = self.dataset[list(range(int((self.n/2)-(n_test/2)), int(self.n/2))) + list(range(int(self.n-(n_test/2)), self.n)), :-1]
                y_test = self.dataset[list(range(int((self.n/2)-(n_test/2)), int(self.n/2))) + list(range(int(self.n-(n_test/2)), self.n)), -1]
                
                # standardize data
                ss = StandardScaler()
                X_train = ss.fit_transform(X_train)
                X_test = ss.fit_transform(X_test)
                y_train = ss.fit_transform(y_train.reshape(-1, 1)).squeeze()
                y_true = ss.transform(y_test.reshape(-1, 1)).squeeze()
                
                # get C2 from the cross validation data
                if kernel[4:] == 'custom':
                    cv_c2_x_kernel = pd.read_csv('{}program_data/cv_{}_{}.csv'.format(self.results, kernel, self.dataset_name[23:])) # import cross validation data
                    idx = cv_c2_x_kernel['n_train={} d={}'.format(n_train, d)].idxmin()
                    c2 = self.C2[idx]
                    
                # scikit learn Support Vector Machines
                if kernel[4:] == 'custom':
                    slm = self.kernel_dic(kernel)
                    slm.fit(self.gram_matrix(X_train, X_train, c2, d), y_train)
                    y_pred = slm.predict(self.gram_matrix(X_test, X_train, c2, d))
                else:
                    slm = self.kernel_dic(kernel)
                    slm.fit(X_train, y_train)
                    y_pred = slm.predict(X_test)

                # support vectors
                X_support = X_train[slm.support_, :]
                y_support = y_train[slm.support_]

                # beta
                beta_0 = slm.intercept_
                alpha = np.abs(slm.dual_coef_.squeeze()) # α (dual coefficient)
                if kernel[4:] == 'custom':
                    # β̂  = (Ip + C2 D^T D)^-1 * X^T(α * y)
                    D = self.diff_matrix(self.p, d) # difference matrix
                    beta = np.linalg.inv(self.Ip + c2*D.T@D)@X_support.T@(alpha*y_support)
                else:
                    # β̂  = X^T(α * y)
                    beta = X_support.T@(alpha*y_support)

                # mean square error
                mse.append(mean_squared_error(y_true, y_pred))
            
            if kernel[4:] == 'custom':
                mse_table['{} d={}'.format(kernel, d)] = mse
            else:
                mse_table[kernel] = mse
                
        # round mse table to 3 decimal places
        mse_table = round(mse_table, 3)
                
        # write mse table to csv
        mse_table.to_csv('{}output/mse_table_{}.csv'.format(self.results, self.dataset_name[23:]))
    
    ######################################################################
    ############# Models (optimal regularization parameter) ##############
    ######################################################################
    def create_models(self, **kwargs):

        # variables
        param = kwargs.pop('param') # kernel and d (for custom kernel)
        
        # for each dataset with n training observations
        for n_train in self.n_train_list:
        
            if self.dataset_name[-14:] == 'Classification':
                # double n_train for classifiers
                n_train = 2*n_train
                
                # calibration data for classifiers
                n_calibrate = 2*1000 # number of calibration observations
                X_calibrate = self.dataset[list(range(int((self.n/2)-(n_calibrate/2)-1000), int(self.n/2)-1000)) + list(range(int(self.n-(n_calibrate/2)-1000), self.n-1000)), :-1]
                y_calibrate = self.dataset[list(range(int((self.n/2)-(n_calibrate/2)-1000), int(self.n/2)-1000)) + list(range(int(self.n-(n_calibrate/2)-1000), self.n-1000)), -1]
                
            # dataset
            X_train = self.dataset[list(range(0, int(n_train/2))) + list(range(int(self.n/2), int((self.n/2)+(n_train/2)))), :-1]
            y_train = self.dataset[list(range(0, int(n_train/2))) + list(range(int(self.n/2), int((self.n/2)+(n_train/2)))), -1]
            
            # standardize data
            ss = StandardScaler()
            X_train = ss.fit_transform(X_train)
            
            if self.dataset_name[12:22] == 'Regression':
                y_train = ss.fit_transform(y_train.reshape(-1, 1)).squeeze()
            
            # for each kernel and d
            for kernel, d in param:
                
                # get C2 from the cross validation data
                if kernel[4:] == 'custom':
                    if kernel[:3] == 'svc':
                        cv_c2_x_kernel = pd.read_csv('{}program_data/cv_{}.csv'.format(self.results, kernel)) # import cross validation data
                        idx = cv_c2_x_kernel['n_train={} d={}'.format(int(n_train/2), d)].idxmin()
                    elif kernel[:3] == 'svr':
                        cv_c2_x_kernel = pd.read_csv('{}program_data/cv_{}_{}.csv'.format(self.results, kernel, self.dataset_name[23:])) # import cross validation data
                        idx = cv_c2_x_kernel['n_train={} d={}'.format(n_train, d)].idxmin()
                        
                    c2 = self.C2[idx]
                    
                # scikit learn Support Vector Machines
                slm = self.kernel_dic(kernel)
                    
                if kernel[4:] == 'custom':
                    slm.fit(self.gram_matrix(X_train, X_train, c2, d), y_train)
                else:
                    slm.fit(X_train, y_train)
                
                # create model
                models = pd.read_csv('{}models/models.csv'.format(self.results)) # import models c2
                
                if kernel[:3] == 'svc':
                    # uncalibrated model
                    dump(slm, '{}models/{}_ntrain={}_d={}.joblib'.format(self.results, kernel[:3], int(n_train/2), d)) # save model
                    idx = models.index[models['Model']=='{}_ntrain={}_d={}'.format(kernel[:3], int(n_train/2), d)] # get index of model
                    models.at[idx, 'C\u2082 Sci'] = self.to_ascii(c2) # enter model c2 in scientific notation
                    models.at[idx, 'C\u2082'] = c2 # enter model c2
                    models.at[idx, 'd'] = d # enter model d
                    
                    # calibrated model
                    clf_cal = CalibratedClassifierCV(base_estimator=slm, method='isotonic', cv='prefit')
                    clf_cal.fit(self.gram_matrix(X_calibrate, X_train, c2, d), y_calibrate)
                    dump(clf_cal, '{}models/{}_ntrain={}_d={}_cal.joblib'.format(self.results, kernel[:3], int(n_train/2), d)) # save model
                    idx = models.index[models['Model']=='{}_ntrain={}_d={}_cal'.format(kernel[:3], int(n_train/2), d)] # get index of model
                    models.at[idx, 'C\u2082 Sci'] = self.to_ascii(c2) # enter model c2 in scientific notation
                    models.at[idx, 'C\u2082'] = c2 # enter model c2
                    models.at[idx, 'd'] = d # enter model d
                    
                    # save models parameters
                    models.to_csv('{}models/models.csv'.format(self.results), index=False) # save models
                    
                elif kernel[:3] == 'svr':
                    dump(slm, '{}models/{}_{}_ntrain={}_d={}.joblib'.format(self.results, kernel[:3], self.dataset_name[24:], n_train, d)) # save model
                    idx = models.index[models['Model']=='{}_{}_ntrain={}_d={}'.format(kernel[:3], self.dataset_name[24:], n_train, d)] # get index of model
                    models.at[idx, 'C\u2082 Sci'] = self.to_ascii(c2) # enter model c2 in scientific notation
                    models.at[idx, 'C\u2082'] =  c2 # enter model c2
                    models.at[idx, 'd'] = d # enter model d
                    models.to_csv('{}models/models.csv'.format(self.results), index=False) # save models

    ##################################################
    ############# Performance Evaluation #############
    ##################################################
    def perf(self, **kwargs):
        n_test = kwargs.pop('n_test', 100)
        num_deg = kwargs.pop('num_deg', 1)
        param = kwargs.pop('param')
        accuracy = pd.DataFrame(index=[n for n in self.n_train_list])
        accuracy.index.name='n_train'
        
        # for each dataset with n training observations
        for n_train in self.n_train_list:
            
            # double n_train for classifiers
            if self.dataset_name[-14:] == 'Classification':
                n_train = 2*n_train
                n_test = 2*n_test
            
            # dataset
            X_train = self.dataset[list(range(0, int(n_train/2))) + list(range(int(self.n/2), int((self.n/2)+(n_train/2)))), :-1]
            y_train = self.dataset[list(range(0, int(n_train/2))) + list(range(int(self.n/2), int((self.n/2)+(n_train/2)))), -1]
            X_test = self.dataset[list(range(int((self.n/2)-(n_test/2)), int(self.n/2))) + list(range(int(self.n-(n_test/2)), self.n)), :-1]
            y_test = self.dataset[list(range(int((self.n/2)-(n_test/2)), int(self.n/2))) + list(range(int(self.n-(n_test/2)), self.n)), -1]

            # standardize data
            ss = StandardScaler()
            X_train = ss.fit_transform(X_train)
            X_test = ss.transform(X_test)
            
            if self.dataset_name[12:23] == 'Regression':
                y_train = ss.fit_transform(y_train.reshape(-1, 1)).squeeze()
                y_true = ss.transform(y_test.reshape(-1, 1)).squeeze()
            else:
                y_true = y_test
            
            # variables
            fpr_tpr = [] # false positive rates and true positive rates
            legend = [] # legend for ROC curves
            residuals_x_kernel = pd.DataFrame() # residuals

            # for each kernel and d
            for kernel, d in param:
                    
                # get C2 from the cross validation data
                if kernel[4:] == 'custom':
                    if kernel[:3] == 'svc':
                        #cv_c2_x_kernel = pd.read_csv('{}program_data/cv_{}.csv'.format(self.results, kernel)) # import cross validation data
                        #idx = cv_c2_x_kernel['n_train={} d={}'.format(int(n_train/2), d)].idxmin()
                        
                        # load calibrated classifier model
                        model = 'svc_ntrain={}_d={}_cal'.format(int(n_train/2), d)
                        slm = load('{}models/{}.joblib'.format(self.results, model)) # load model
                        models = pd.read_csv('{}models/models.csv'.format(self.results)) # read models parameters table
                        idx = models.index[models['Model']==model] # get index of model
                        c2 = int(models.loc[idx, 'C\u2082']) # C2
                        d = int(models.loc[idx, 'd']) # d
                    
                    elif kernel[:3] == 'svr':
                        cv_c2_x_kernel = pd.read_csv('{}program_data/cv_{}_{}.csv'.format(self.results, kernel, self.dataset_name[23:])) # import cross validation data
                        idx = cv_c2_x_kernel['n_train={} d={}'.format(n_train, d)].idxmin()
                        c2 = self.C2[idx]
                    
                # scikit learn Support Vector Machines
                if kernel[4:] == 'custom':
                    slm = self.kernel_dic(kernel)
                    slm.fit(self.gram_matrix(X_train, X_train, c2, d), y_train)
                    y_pred = slm.predict(self.gram_matrix(X_test, X_train, c2, d)) # predict y_true
                else:
                    slm = self.kernel_dic(kernel)
                    slm.fit(X_train, y_train)
                    y_pred = slm.predict(X_test) # predict y_true

                # dataframe column name
                if kernel[4:] == 'custom':
                    col_name = '{}\nC\u2082={} d={}'.format(kernel[4:].capitalize(), self.to_ascii(c2), d)
                else:
                    col_name = '{}'.format(kernel[4:].upper() if kernel[4:] == 'rbf' else kernel[4:].capitalize())
                    
                # classifier
                if kernel[:3] == 'svc':
                    # ROC curves
                    if kernel[4:] == 'custom':
                        probs = slm.predict_proba(self.gram_matrix(X_test, X_train, c2, d)) # predict probabilities for classes (-1, 1)
                    else:
                        probs = slm.predict_proba(X_test) # predict probabilities for classes (-1, 1)
                        
                    probs_pos = probs[:, 1] # probabilities for the positive outcome only
                    fpr, tpr, threshold = metrics.roc_curve(y_true, probs_pos)
                    roc_auc = metrics.auc(fpr, tpr)

                    fpr_tpr.append(fpr)
                    fpr_tpr.append(tpr)
                    
                    if kernel[4:] == 'custom':
                        legend.append('{}, C\u2082={}, d={}'.format(kernel[4:].capitalize(), self.to_ascii(c2), d))
                    else:
                        legend.append('{}'.format(kernel[4:].upper() if kernel[4:] == 'rbf' else kernel[4:].capitalize()))

                    # confusion matrix
                    if kernel[4:] == 'custom':
                        y_pred = slm.predict(self.gram_matrix(X_test, X_train, c2, d))
                    else:
                        y_pred = slm.predict(X_test)
                        
                    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
                    
                    cm = [[(tn/(tn+fp))*100, (fp/(tn+fp))*100], [(fn/(fn+tp))*100, (tp/(fn+tp))*100]]

                    df_cm = pd.DataFrame(cm, index = ["Neutral", "Sweep"],
                        columns = [i for i in ["Neutral", "Sweep"]])

                    cmap = sns.cubehelix_palette(start=(5/6)*pi, rot=0.1, light=1, dark=0.25, n_colors=10)

                    gs_kw = dict(left=0, right=1, top=1, bottom=0)

                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 2.5), gridspec_kw=gs_kw)

                    ax = sns.heatmap(data=df_cm, vmin=0.0, vmax=100.0, annot=True, annot_kws={"size": 14}, fmt='.1f', ax=ax, square=True, cbar=False,
                                    cmap=self.grayscale_cmap(ListedColormap(cmap)), xticklabels=False, yticklabels=False)
                    
                    for _, spine in ax.spines.items():
                        spine.set_visible(True)
                        spine.set_linewidth(2)
                        spine.set_color('black')
                    
                    if kernel[4:] == 'custom':
                        plt.savefig('{}output/confusion_matrix_{}_d={}_{}.eps'.format(self.results, kernel[4:], d, int(n_train/2)), format='eps')
                    else:
                        plt.savefig('{}output/confusion_matrix_{}_{}.eps'.format(self.results, kernel[4:], int(n_train/2)), format='eps')

                    # accuracy
                    if kernel[4:] == 'custom':
                        accuracy.loc[int(n_train/2), '{} d={}'.format(kernel[4:], d)] = round(100*metrics.accuracy_score(y_true, y_pred), 1)
                    else:
                        accuracy.loc[int(n_train/2), kernel[4:]] = round(100*metrics.accuracy_score(y_true, y_pred), 1)
                
                # regressor
                elif kernel[:3] == 'svr':
                    residuals = y_true - y_pred # residuals
                    residuals_x_kernel[col_name] = pd.Series(residuals)
                    
            # ROC curves
            if kernel[:3] == 'svc':
                rcparams = {"lines.linewidth": 1.5, 'axes.labelsize': 16, 'legend.fontsize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14}
                sns.set(style='white', rc=rcparams)
                gs_kw = {'left': 0.11, 'right': 0.98, 'top': 0.98, 'bottom': 0.07}
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), gridspec_kw=gs_kw)
                
                for i, c in zip(range(0, len(fpr_tpr), 2), self.palette_kernels):
                    sns.lineplot(x=fpr_tpr[i], y=fpr_tpr[i+1], color=c, ci=None)
                    
                ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
                #ax.legend(legend)
                plt.savefig('{}output/roc_curves_{}.eps'.format(self.results, int(n_train/2)), format='eps')
                plt.close()
                
                # zoomed in
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), gridspec_kw=gs_kw)
                
                for i, c in zip(range(0, len(fpr_tpr), 2), self.palette_kernels):
                    sns.lineplot(x=fpr_tpr[i], y=fpr_tpr[i+1], color=c, ci=None)
                    
                ax.set(xlim=(-0.01, 0.21), ylim=(0.79, 1.01), xticks=[0, 0.1, 0.2], yticks=[0.8, 0.9, 1])
                #ax.legend(legend)
                plt.savefig('{}output/roc_curves_{}_zoom.eps'.format(self.results, int(n_train/2)), format='eps')
                plt.close()
                
            # violin plots
            elif kernel[:3] == 'svr':
                rcparams = {'axes.labelsize': 14, 'xtick.labelsize': 10, 'ytick.labelsize': 12, 'ytick.left': True}
                sns.set(style='white', rc=rcparams)
                gs_kw = {'left': 0.10, 'right': 0.98, 'top': 0.96, 'bottom': 0.02}
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), gridspec_kw=gs_kw)
                ax = sns.violinplot(data=residuals_x_kernel, scale='width', palette=self.palette_kernels)
                ax.set(xticklabels=[], ylim=(-6, 8))
                plt.savefig('{}output/violin_plots_{}_{}.eps'.format(self.results, self.dataset_name[23:], n_train), format='eps')
                plt.close()
                
        # accuracy
        if self.dataset_name[-14:] == 'Classification':
            accuracy.to_csv('{}output/svc_accuarcy.csv'.format(self.results))
    
    
    ##################################################
    ########## Performance on Empirical Data #########
    ##################################################
    def perf_emp(self):
    
        #variables
        clf_results = pd.DataFrame(columns=['Chromosome', 'Position', 'Probability']) # classification results
        n_train = 10000
        X_test = np.empty((0, 101))
        
        # double n_train for classifiers
        if self.dataset_name[-14:] == 'Classification':
            n_train = 2*n_train

        # X_train
        X_train = self.dataset[list(range(0, int(n_train/2))) + list(range(int(self.n/2), int((self.n/2)+(n_train/2)))), :-1]
        y_train = self.dataset[list(range(0, int(n_train/2))) + list(range(int(self.n/2), int((self.n/2)+(n_train/2)))), -1]

        # load classifier model
        model = self.model # name of model
        clf = load('{}models/{}.joblib'.format(self.results, self.model)) # load model
        models = pd.read_csv('{}models/models.csv'.format(self.results)) # read models parameters table
        idx = models.index[models['Model']==model] # get index of model
        c2 = int(models.loc[idx, 'C\u2082'])
        d = int(models.loc[idx, 'd'])
        
        # import parsed chromosome data: Position, Computation
        chrome_data = pd.read_csv(self.file_in)
        
        # get chromosome number from name of file_in
        if self.file_in[-6].isdigit():
            chrome_num = self.file_in[-6:-4]
        else:
            chrome_num = self.file_in[-5:-4]
        
        # debug mode
        if self.debug == 'true':
            end = 55
        else:
            end = chrome_data.shape[0]-50
        
        for i in range(50, end):
            # enter: Chromosome, Position
            clf_results = clf_results.append({'Chromosome': chrome_num, 'Position': chrome_data.loc[i, 'POS']}, ignore_index=True)
            
            # enter: 101 Features
            X_test = np.append(X_test, [chrome_data.iloc[i-50:i+51, 1]], axis=0)
            
        # standardize data
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.transform(X_test)
        
        # predict probabilities
        probs_pos = clf.predict_proba(self.gram_matrix(X_test, X_train, c2, d))[:, 1] # predict probabilities for positive outcome
        clf_results['Probability'] = probs_pos
        
        # predict class
        y_pred = clf.predict(self.gram_matrix(X_test, X_train, c2, d))
        clf_results['Class'] = y_pred
        
        # predict calibrated probabilities
        clf_cal = CalibratedClassifierCV(base_estimator=clf, method='isotonic', cv='prefit')
        clf_cal.fit(self.gram_matrix(X_train, X_train, c2, d), y_train)
        probs_pos_cal = clf_cal.predict_proba(self.gram_matrix(X_test, X_train, c2, d))[:, 1] # predict probabilities for positive class (1)
        pd.DataFrame(data=probs_pos_cal).to_csv('{}output/probs_pos_cal.csv'.format(self.results))
        clf_results['ProbabilityCalibrated'] = probs_pos_cal
        
        # predict calibrated class
        y_pred = clf_cal.predict(self.gram_matrix(X_test, X_train, c2, d))
        clf_results['ClassCalibrated'] = y_pred
        
        # save classifier results
        clf_results.to_csv(self.file_out, index=False)
        
    ##################################################
    ################ Reliability Plot ################
    ##################################################
    def reliability(self, kernel):
        
        #variables
        n_train = 2*10000 # number of training observations
        n_calibrate = 2*1000 # number of calibration observations
        n_test = 2*1000 # number of test observations
        
        # dataset
        X_train = self.dataset[list(range(0, int(n_train/2))) + list(range(int(self.n/2), int((self.n/2)+(n_train/2)))), :-1]
        y_train = self.dataset[list(range(0, int(n_train/2))) + list(range(int(self.n/2), int((self.n/2)+(n_train/2)))), -1]
        X_calibrate = self.dataset[list(range(int((self.n/2)-(n_calibrate/2)-1000), int(self.n/2)-1000)) + list(range(int(self.n-(n_calibrate/2)-1000), self.n-1000)), :-1]
        y_calibrate = self.dataset[list(range(int((self.n/2)-(n_calibrate/2)-1000), int(self.n/2)-1000)) + list(range(int(self.n-(n_calibrate/2)-1000), self.n-1000)), -1]
        X_test = self.dataset[list(range(int((self.n/2)-(n_test/2)), int(self.n/2))) + list(range(int(self.n-(n_test/2)), self.n)), :-1]
        y_true = self.dataset[list(range(int((self.n/2)-(n_test/2)), int(self.n/2))) + list(range(int(self.n-(n_test/2)), self.n)), -1]
        
        # standardize data
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_calibrate = ss.transform(X_calibrate)
        X_test = ss.transform(X_test)
        
        # load classifier model
        clf = load('{}models/{}.joblib'.format(self.results, self.model)) # load model
        models = pd.read_csv('{}models/models.csv'.format(self.results)) # read models parameters table
        idx = models.index[models['Model']==self.model] # get index of model
        c2 = int(models.loc[idx, 'C\u2082']) # C2
        d = int(models.loc[idx, 'd']) # d
        
        # predict uncalibrated probabilities
        probs_pos = clf.predict_proba(self.gram_matrix(X_test, X_train, c2, d))[:, 1] # predict probabilities for the positive class
        prob_true, prob_pred = self.calibration_curve(y_true, probs_pos, n_bins=950)
        rel_data = pd.DataFrame({'prob_pred': prob_pred, 'prob_true': prob_true})
        rel_data.to_csv('{}output/rel_data.csv'.format(self.results), index=False)
        
        # predict calibrated probabilities
        clf_cal = CalibratedClassifierCV(base_estimator=clf, method='isotonic', cv='prefit')
        clf_cal.fit(self.gram_matrix(X_calibrate, X_train, c2, d), y_calibrate)
        probs_pos_cal = clf_cal.predict_proba(self.gram_matrix(X_test, X_train, c2, d))[:, 1] # predict probabilities for the positive class
        prob_true, prob_pred = self.calibration_curve(y_true, probs_pos_cal, n_bins=950)
        rel_data_cal = pd.DataFrame({'prob_pred': prob_pred, 'prob_true': prob_true})
        rel_data_cal.to_csv('{}output/rel_data_cal.csv'.format(self.results), index=False)
        
        # save model
        models = pd.read_csv('{}models/models.csv'.format(self.results)) # import models c2
        dump(clf_cal, '{}models/{}_ntrain={}_d={}_cal.joblib'.format(self.results, kernel[:3], int(n_train/2), d)) # save model
        idx = models.index[models['Model']=='{}_ntrain={}_d={}_cal'.format(kernel[:3], int(n_train/2), d)] # get index of model
        models.at[idx, 'C\u2082 Sci'] = self.to_ascii(c2) # enter model c2 in scientific notation
        models.at[idx, 'C\u2082'] = c2 # enter model c2
        models.at[idx, 'd'] = d # enter model d
        models.to_csv('{}models/models.csv'.format(self.results), index=False) # save models
                
        # probabilities uncalibrated and calibrated
        probabilities = pd.DataFrame({'prob_uncal': probs_pos, 'prob_cal': probs_pos_cal})
        probabilities.to_csv('{}output/probabilities.csv'.format(self.results), index=False)
        
        # histogram uncalibrated probabilities
        gs_kw = {'left': 0.1, 'right': 0.98, 'top': 0.98, 'bottom': 0.08}
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6), gridspec_kw=gs_kw)
        ax.set(xlabel='Probability of Sweep Class', ylabel='Number of Observations')
        sns.histplot(data=probabilities.loc[0:1000, 'prob_uncal'], color=self.palette_prob[1], bins=30, edgecolor="none", legend=False) # neutral
        sns.histplot(data=probabilities.loc[1000:2000, 'prob_uncal'], color=self.palette_prob[0], bins=30, edgecolor="none", legend=False) # sweep
        sns.rugplot(data=probabilities.loc[0:1000, 'prob_uncal'], color=self.palette_prob[1], clip_on=False, height=0.5, legend=False) # neutral
        sns.rugplot(data=probabilities.loc[1000:2000, 'prob_uncal'], color=self.palette_prob[0], clip_on=False, height=0.5, legend=False) # sweep
        plt.savefig('{}output/probabilities_uncal.eps'.format(self.results))
        plt.close()
        
        # histogram calibrated probabilities
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6), gridspec_kw=gs_kw)
        ax.set(xlabel='Probability of Sweep Class', ylabel='Number of Observations')
        sns.histplot(data=probabilities.loc[0:1000, 'prob_cal'], color=self.palette_prob[1], bins=30, edgecolor="none", legend=False) # neutral
        sns.histplot(data=probabilities.loc[1000:2000, 'prob_cal'], color=self.palette_prob[0], bins=30, edgecolor="none", legend=False) # sweep
        sns.rugplot(data=probabilities.loc[0:1000, 'prob_cal'], color=self.palette_prob[1], legend=False) # neutral
        sns.rugplot(data=probabilities.loc[1000:2000, 'prob_cal'], color=self.palette_prob[0], legend=False) # sweep
        plt.savefig('{}output/probabilities_cal.eps'.format(self.results))
        plt.close()
        
        # reliability plot
        rcparams = {'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10}
        sns.set(style='white', rc=rcparams)
        gs_kw = {'left': 0.13, 'right': 0.98, 'top': 0.98, 'bottom': 0.1}
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), gridspec_kw=gs_kw)
        ax.set(xlabel='Mean Prediction', ylabel='Observed Fraction')
        
        sns.lineplot(data=[0, 1], linestyle='--', color='black') # perfectly calibrated reliability curve
        sns.lineplot(data=rel_data, x='prob_pred', y='prob_true', legend='brief', label='uncalibrated',
                     linewidth=1, marker='o', markersize=5, markeredgewidth=0) # uncalibrated reliability curve
        sns.lineplot(data=rel_data_cal, x='prob_pred', y='prob_true', legend='brief', label='calibrated',
                     linewidth=1, marker='o', markersize=5, markeredgewidth=0) # calibrated reliability curve
        
        plt.savefig('{}output/reliability.eps'.format(self.results))
    