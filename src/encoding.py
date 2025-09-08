# I/O packages
import os
import argparse
import glob
from pathlib import Path
import csv
import nibabel
import nibabel.processing
import h5py

# data manipulation
import numpy as np
import pandas as pd
import nilearn
from nilearn import surface
import nilearn.datasets
import nilearn.masking
import nilearn.signal
from scipy import stats
from joblib import Parallel, delayed

# plotting packages
from matplotlib import colors
import matplotlib.pyplot as plt

# for encoding model
from sklearn.model_selection import KFold,GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from voxelwise_tutorials.delayer import Delayer
from himalaya.backend import set_backend
from himalaya.kernel_ridge import MultipleKernelRidgeCV
from himalaya.kernel_ridge import Kernelizer
from himalaya.kernel_ridge import ColumnKernelizer
from himalaya.scoring import r2_score_split
from himalaya.viz import plot_alphas_diagnostic
from statsmodels.stats.multitest import fdrcorrection
import torch

# custom python modules
from src import helpers, plotting_helpers

class EncodingModel:
    def __init__(self, args):
        self.testing = args.testing #specify if you are in testing mode (True) or not (False). Testing mode makes the encoding go faster (fewer cv folds and hyperparameters to choose from)
        self.process = 'EncodingModel'
        self.dir = args.dir #highest level directory of the BIDS style dataset
        self.data_dir = args.data_dir #data directory of the BIDS style dataset
        self.out_dir = args.out_dir + "/" + self.process #output directory of the BIDS style directory, probably 'analysis'
        self.figure_dir = args.figure_dir +'/'+self.process #figure output directory of the BIDS style directory
        self.sid = 'sub-'+str(int(args.s_num)).zfill(2) #creating the BIDS style subject id from the given subject number
        self.task = args.task #the task name of the fmri data that will be used for the encoding model
        self.space = args.space #the brain space the encoding model will be run in
        self.mask_name = args.mask #the name of the brain mask to use
        self.mask = None #the variable for the mask data to be loaded into (see: load_mask())
        self.mask_affine = None #the variable for the mask affine fata to be loaded into (see: load_mask())
        self.smoothing_fwhm = args.smoothing_fwhm #the smoothing fwhm (full-width at half maximum) of the gaussian used for smoothing brain data
        self.chunklen = args.chunklen #the number of TRs to group the fmri data into (used during encoding model to take care of temporal autocorrelation in the data)
        self.model = args.model #the name of the encoding model to run
        self.fMRI_data = [] #variable to store the fmri data in
        self.brain_shape = None #the brain shape that all masks and data will be resampled to. this will be the output size of nii.gz data
        self.affine = [] #variable to store the affine of the fmri data

        # Sherlock specific timing parameters
        self.wait_TR = 2 #number of TRs from the beginning with no data (pause before the movie)
        self.stim_start = 26 #TR of the start of the stimulus we are interested in
        self.intro_start = self.wait_TR # TR of the start of the introductory movie
        self.intro_end = self.stim_start-1 # TR of the end of the introductory movie
        self.included_data_fMRI = [(self.stim_start+self.wait_TR,946+self.wait_TR),(975+self.wait_TR,1976+self.wait_TR)] #TR indices of the fmri data to use in encoding model, there should be one entry in this list for every run that will be included #Haemy had 27:946, and 973:1976, have to subtract one for python 0-indexing and add one to end bc exclusive slicing in python
        self.save_weights = bool(args.save_weights) #whether to save the fitted encoding model weights or not
        print('args.save_weights',args.save_weights)
        self.save_hemodynamic_fits = bool(args.save_weights) #if saving weights, also saving hemodynamic fits
        self.save_individual_feature_performance=True
    

        #variables to store results in:
        self.performance = []
        self.ind_feature_performance = []
        self.ind_product_measure = []
        self.beta_weights = []
        self.perf_p_unc = []
        self.perf_fdr_reject = []
        self.perf_p_fdr = []
        self.ind_perf_p_unc = []
        self.ind_perf_fdr_reject = []
        self.ind_perf_p_fdr = []
        self.ind_prod_p_unc = []
        self.ind_prod_fdr_reject = []
        self.ind_prod_p_fdr = []
        self.performance_null =  []
        self.ind_feature_performance_null = []
        self.ind_product_measure_null = []
        self.preference1_map = []
        self.preference2_map = []
        self.features_preferred_delay = []

        
        #creation of necessary output folders:
        Path(f'{self.out_dir}/').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"performance"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"perf_p_unc"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"perf_p_fdr"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"ind_feature_performance"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"ind_perf_p_unc"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"ind_perf_p_fdr"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"ind_product_measure"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"features_preferred_delay"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"ind_prod_p_unc"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"ind_prod_p_fdr"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"preference_map"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"weights"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"features"}').mkdir(exist_ok=True, parents=True)
        
        Path(f'{self.figure_dir}/{"performance"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"perf_p_fdr"}').mkdir(exist_ok=True, parents=True)
        
        Path(f'{self.figure_dir}/{"ind_feature_performance"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"ind_product_measure"}').mkdir(exist_ok=True, parents=True)

        Path(f'{self.figure_dir}/{"preference_map"}').mkdir(exist_ok=True, parents=True)
        
        self.features = [] #variable to store the features to use in the encoding model

        #dictionary specifying the features to use for a given encoding model name
        self.model_features_dict = helpers.get_models_dict()
        
        #saving the weights of specific layers of the deep neural network models + word2vec and motion (only a subset for memory considerations)
        self.feature_weights_to_save = ['alexnet_layer2','alexnet_layer5','alexnet_layer6','alexnet_layer7',
                                        'word2vec','motion']
        ## save the weights of the final 3 or 4 layers of the transformer models 
        transformer_layers = [transformer+'_layer'+str(layer) for transformer in ['sbert','GPT2_1sent','GPT2_3sent'] for layer in [10,11,12]]
        self.feature_weights_to_save.extend(transformer_layers)
        transformer_layers = [transformer+'_layer'+str(layer) for transformer in ['SimCLR_attention','SimCLR_embedding','SLIP_attention','SLIP_embedding'] for layer in [9,10,11,12]]
        self.feature_weights_to_save.extend(transformer_layers)

        self.final_weight_feature_names = [] #place to save the final order of the features that have their weights saved
        
        #dictionary specifying the filenames of each of the feature spaces
        self.features_dict = {
                       'alexnet':'torchvision_alexnet_imagenet1k_v1',
                       'hubert':'hubert-base-ls960-ft',
                       'sbert':'mpnet-base-v2',
                       'alexnet_layer1':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-3_srp',
                       'alexnet_layer2':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-6_srp',
                       'alexnet_layer3':'torchvision_alexnet_imagenet1k_v1_ReLU-2-8_srp',
                       'alexnet_layer4':'torchvision_alexnet_imagenet1k_v1_ReLU-2-10_srp',
                       'alexnet_layer5':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-13_srp',
                       'alexnet_layer6':'torchvision_alexnet_imagenet1k_v1_ReLU-2-16',
                       'alexnet_layer7':'torchvision_alexnet_imagenet1k_v1_ReLU-2-19',
                       'social':'social',
                       'num_agents':'num_agents',
                       'turn_taking':'turn_taking',
                       'speaking':'speaking',
                       'mentalization': 'mentalization',
                       'valence':'valence',
                       'arousal':'arousal',
                       'motion':'pymoten',
                       'face': 'face',
                       'indoor_outdoor':'indoor_outdoor',
                       'written_text':'written_text',
                       'music':'music',
                       'glove':'glove',
                       'word2vec':'word2vec',
                       'pitch':'pitch',
                       'amplitude':'amplitude',
                       'pixel': 'pixel',
                       'hue':'hue'
                       }
        for layer in self.model_features_dict['sbert']:
            self.features_dict[layer]='downsampled_all-mpnet-base-v2_'+layer.split('_')[1]
        for layer in self.model_features_dict['GPT2_1sent']:
            self.features_dict[layer]='downsampled_GPT2_'+layer.split('_')[2] + '_context-sent-1_embeddings'
        for layer in self.model_features_dict['GPT2_3sent']:
            self.features_dict[layer]='downsampled_GPT2_'+layer.split('_')[2] + '_context-sent-3_embeddings'
        for layer in self.model_features_dict['GPT2_1word']:
            self.features_dict[layer]='GPT2_'+layer.split('_')[2] + '_word'
        for layer in self.model_features_dict['hubert']:
            self.features_dict[layer]='hubert-base-ls960-ft_'+layer.split('_')[1]+'_TR' #embeddings extracted each TR

        for time_chunk in [1,5,4,8,10,15,16,20]:
            for layer in self.model_features_dict['GPT2_'+str(time_chunk)+'s']:
                self.features_dict[layer]='GPT2_'+layer.split('_')[2] + '_running_time_chunk-'+str(time_chunk)

        tracker=2
        for layer in self.model_features_dict['SimCLR_attention']:
            self.features_dict[layer]='slip_vit_b_simclr_yfcc15m_attention-3-'+str(tracker) +'_srp' #get the SRP reduced version
            tracker=tracker+8

        tracker=6
        for layer in self.model_features_dict['SimCLR_embedding']:
            self.features_dict[layer]='slip_vit_b_simclr_yfcc15m_mlp-3-'+str(tracker) +'_srp' #get the SRP reduced version
            tracker=tracker+8

        self.cmaps = plotting_helpers.get_cmaps()
        self.colors_dict = plotting_helpers.get_colors_dict()
        
        self.model_features = self.model_features_dict[self.model] #the names of the feature spaces to use for the encoding model
        print(self.model_features)
        self.feature_names = self.model_features
        
    def trim_fMRI(self,norm=False):
        """ This function trims the fmri data to the data that will be used in the encoding model. 
            It will take the stored fmri data (already concatenated across any runs by load_smooth_denoise_fMRI()) and trim the specified parts (self.included_data_fMRI)
        """
        print('...trimming fMRI...')
        #trim TRs from the concatenated data (concatenated over runs)
        trimmed_data = np.array([])
        trimmed_confounds = []
        run_ends = []
        for section in self.included_data_fMRI:
            start = section[0]
            stop = section[1]
            new_data = np.array(self.fMRI_data)[start:stop] #first dimension should be n samples
            new_confounds = self.confounds[start:stop]
            if(norm):
                new_data = new_data = stats.zscore(new_data,axis=0,nan_policy='omit') #zscore the responses across the samples of each section(from different runs) separately !!
            run_ends.append(len(new_data))
            if(trimmed_data.shape[0]<1):
                trimmed_data = new_data
            else:
                trimmed_data = np.concatenate((trimmed_data,new_data),axis=0) #concatenate along n samples dimension
            trimmed_confounds.extend(new_confounds)
        
        self.fMRI_data = trimmed_data 
        self.confounds = np.array(trimmed_confounds)
        self.run_ends = run_ends


    def load_preprocess_fMRI(self,smooth=False,denoise=False):
            """ finds all runs of the task (assuming BIDS format), iteratively loads, optionally smooths, optionally denoises, and then concatenates them in ascending order 
                and saves the result into the object's fMRI_data """

            #assuming BIDS format
            img = self.data_dir + '/derivatives/'+ self.sid+'/func/'+self.sid+'_task-'+self.task+'_run-*_space-'+self.space+'_desc-preproc_bold.nii.gz'
            print(img)
            runs = []
            for file in glob.glob(img):
                runs.append(file)
            runs.sort() #sort the runs to be ascending order
            #concatenate runs
            fmri_data = np.array([])
            confounds_data = []
            for run in runs:
                print('loading ..run '+str(run))
                print(run)
                img = nibabel.load(run) #load the numpy array
                self.brain_shape = img.shape[:-1] #save the brain_shape for resampling masks later
                whole_brain_mask = nilearn.masking.compute_brain_mask(img, mask_type='whole-brain')
                whole_brain_mask_data = whole_brain_mask.get_fdata()

                if smooth:
                    print('...smoothing with gaussian fwhm='+str(self.smoothing_fwhm)+'...')
                    img = nibabel.processing.smooth_image(img,self.smoothing_fwhm,mode='nearest')
                self.affine = img.affine
                self.brain_shape = img.shape[:-1]
                #load confounds and select which we are using
                confounds_filepath = run.split('_space-'+self.space)[0]+'_desc-confounds_timeseries.tsv' #same filename base without the space label
                confounds_all = pd.read_csv(confounds_filepath,sep='\t')
                confounds_to_use = ['rot_x','rot_y','rot_z','trans_x','trans_y','trans_z',
                                    'framewise_displacement',
                                    'a_comp_cor_00','a_comp_cor_01','a_comp_cor_02','a_comp_cor_03','a_comp_cor_04',
                                    'cosine00', 'cosine01','cosine02','cosine03','cosine04','cosine05','cosine06','cosine07','cosine08','cosine09','cosine10',
                                    'cosine11','cosine12','cosine13','cosine14','cosine15','cosine16','cosine17','cosine18','cosine19','cosine20'
                                    ]
                # confounds_to_use = [ 'framewise_displacement']
                confounds = confounds_all.fillna(0)[confounds_to_use].values #replace nan's with zeroes
                confounds_data.extend(confounds)
                if(denoise):
                    print('...denoising with motion confounds and aCompCor components with their cosine-basis regressors...')
                    #no standardization because all data is used in cross-validation later
                    #no low-pass or high-pass filters because the data is high pass filtered before the aCompCor computations
                    signals = nilearn.masking.apply_mask(img, whole_brain_mask)
                    data = nilearn.signal.clean(signals, detrend=True, standardize=False, confounds=confounds, standardize_confounds=False, low_pass=None, high_pass=None, filter=False, ensure_finite=False) #cosine bases cutoff 128s
                    img = nilearn.masking.unmask(data, whole_brain_mask)
                
                data = img.get_fdata() 

                if fmri_data.shape[0]<1: #if it's the first run, initialize
                    fmri_data = data
                else: #concatenate runs together
                    fmri_data = np.concatenate([fmri_data,data],axis=3)
            
            mask = np.ones(self.brain_shape)
            if self.mask_name!='None':
                #mask the fmri data (only first 3 dim)
                self.mask = helpers.load_mask(self,self.mask_name)
                self.mask_data = self.mask.get_fdata()
                mask = (mask==1) & (self.mask_data==1)
                self.affine = self.mask.affine #change affine to line up fmri data and the mask
                self.fMRI_data = fmri_data[mask].T
            else:
                self.mask = whole_brain_mask_data
                self.affine = whole_brain_mask.affine
                self.fMRI_data = fmri_data[self.mask==1].T
            self.confounds = confounds_data
                
    def unmask_reshape(self, data):
        """ This function puts the given data back into the whole brain (unmasking) and reshapes it back into the specified brain dimensions (self.brain_shape).
            It accounts for any masks that could have been used during analysis, including anatomical and explainable variance masks. 
            It also accounts for multidimensional data (for instance, individual feature performance could have more than one feature in the data)
        """
        mask = np.ones(self.brain_shape)
        mask = mask+self.mask.get_fdata()

        mask = (mask==np.max(mask))*1
        flattened_mask = np.reshape(mask,(-1))

        if data.ndim>2 :
            final_data = np.zeros((data.shape[0],data.shape[1],flattened_mask.shape[0]))
            for curr_outer_slice in range(0,data.shape[0]):
                for curr_inner_slice in range(0,data.shape[1]):
                    final_data[curr_outer_slice][curr_inner_slice][flattened_mask==1] = data[curr_outer_slice][curr_inner_slice]
            final_shape = (data.shape[0],data.shape[1],self.brain_shape[0],self.brain_shape[1],self.brain_shape[2])
        elif data.ndim>1:
            final_data = np.zeros((data.shape[0],flattened_mask.shape[0]))
            for curr_slice in range(0,data.shape[0]):
                final_data[curr_slice][flattened_mask==1] = data[curr_slice]
            final_shape = (data.shape[0],self.brain_shape[0],self.brain_shape[1],self.brain_shape[2])
        else:
            final_data =np.zeros((flattened_mask.shape[0]))
            final_data[flattened_mask==1] = data
            final_shape = (self.brain_shape[0],self.brain_shape[1],self.brain_shape[2])
            
        final_data = np.reshape(final_data,final_shape)
        return final_data

    def get_preference_map(self,temp_ind_product_measure):
        """This helper function returns the indices of the maximally predicting feature for each voxel"""
        temp_transposed = temp_ind_product_measure.T
        # print(temp_transposed)
        nan_col = np.nanmax(temp_transposed, axis=1)
        temp = np.zeros(temp_ind_product_measure.shape[1])
        temp[~np.isnan(nan_col)] = np.nanargmax(temp_transposed[~np.isnan(nan_col),:], axis=1)
        return temp.astype(int)

    def compute_preference_maps(self,restricted=False,threshold=0):
        """This function returns the indices of the first and second maximally predicting feature for each voxel"""

        #get the feature with highest performance for each voxel

        #load all ind feature performances
        #find max performance
        #assign that voxel corresponding label

        if len(self.ind_product_measure)==0:
            self.load_ind_product_measure()

        #mask with fdr corrected pvalues for each feature performance
        # if(restricted):
        #     if(len(self.ind_feature_perf_p_fdr)==0):
        #         self.load_ind_feature_perf_p_fdr()

        temp_ind_product_measure = self.ind_product_measure.copy()
        hemis= ['left','right']

        #restrictions: the ind feature must be above a certain threshold, default is 0 (must be positive)
        if restricted:
            temp_ind_product_measure[temp_ind_product_measure<threshold] = np.nan


        placeholder = np.zeros((1,temp_ind_product_measure.shape[1]))
        temp_ind_product_measure = np.concatenate((placeholder,temp_ind_product_measure))
        temp = np.nanargmax(temp_ind_product_measure.T, axis=1)

        self.preference1_map = (temp).astype(int) #add 1 so that when the preferred index is 0 it is treated separately from np.nan, which gets turned into 0 when cast as int
        
        one_hot_encoded_preference_map = np.eye(temp_ind_product_measure.shape[0])[self.preference1_map.copy()].T.astype(bool)
        temp_ind_product_measure[one_hot_encoded_preference_map] = 0

        temp = np.nanargmax(temp_ind_product_measure.T, axis=1)
        self.preference2_map = (temp).astype(int) #add 1 so that when the preferred index is 0 it is treated separately from np.nan, which gets turned into 0 when cast as int

        fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage') 
        
        preference1_map_surf = []
        preference2_map_surf = []
        for hemi in hemis:
            transform_mesh = fsaverage['pial_'+hemi]
            inner_mesh = fsaverage['white_'+hemi]

            print('self.ind_product_measure',self.ind_product_measure.shape)
            temp = self.unmask_reshape(self.ind_product_measure.copy())
            #add a dimension to begininning that is just 0's (so it will never be the max )
            placeholder = np.reshape(np.zeros(temp[0].shape),(1,temp[0].shape[0],temp[0].shape[1],temp[0].shape[2]))
            temp = np.concatenate((placeholder,temp))
            print('temp',temp.shape)
            nii = nibabel.Nifti1Image(np.transpose(temp, (1, 2, 3, 0)),self.affine)

            temp_ind_product_measure = np.transpose(surface.vol_to_surf(nii, transform_mesh,inner_mesh=inner_mesh,interpolation='nearest'),(1,0))
            if restricted:
                temp_ind_product_measure[temp_ind_product_measure<threshold] = np.nan
            
            print('temp_ind_product_measure',temp_ind_product_measure.shape)

            temp = self.get_preference_map(temp_ind_product_measure)
            pref1_map_surf = temp.astype(int)
            # pref1_map_surf[np.isnan(temp)] = 0
            # pref1_map_surf = pref1_map_surf 
            print(len(pref1_map_surf[np.isnan(pref1_map_surf)]))
            print('pref1_map_surf',pref1_map_surf.shape)
            preference1_map_surf.append(pref1_map_surf) #add 1 so that when the preferred index is 0 it is treated separately from np.nan, which gets turned into 0 when cast as int

            
            one_hot_encoded_preference_map = np.eye(temp_ind_product_measure.shape[0])[pref1_map_surf.copy()].T.astype(bool)
            temp_ind_product_measure[one_hot_encoded_preference_map] = 0

            temp = self.get_preference_map(temp_ind_product_measure)
            pref2_map_surf = temp.astype(int)
            preference2_map_surf.append(pref2_map_surf) #add 1 so that when the preferred index is 0 it is treated separately from np.nan, which gets turned into 0 when cast as int
            
            self.preference1_map_surf = preference1_map_surf
            self.preference2_map_surf = preference2_map_surf 

    
    def plot_preference_maps(self,label='both'):
        """This function plots the maximally predicting voxel across the brain. """

        feature_names = self.feature_names

        file_label = self.sid+'_encoding_model-'+self.model + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
        if self.mask_name is not None:
            file_label = file_label + '_mask-'+self.mask_name

        colors_dict = self.colors_dict

        features_list = ['blank'] + self.feature_names
        surf_cmap = colors.ListedColormap([colors_dict[feature_name] for feature_name in features_list ])
        title=self.sid.replace('sub-NT','') + 'first'

        plotting_helpers.plot_preference_surf(self.preference1_map_surf,self.figure_dir + "/preference_map/" + file_label+'_measure-preference1_map',colors_dict,features_list,threshold=0.001,cmap=surf_cmap,title=title,vmax = len(feature_names))

        if(label =='both'):
            title=self.sid.replace('sub-NT','') + 'second'
            plotting_helpers.plot_preference_surf(self.preference2_map_surf,self.figure_dir + "/preference_map/" + file_label+'_measure-preference2_map',colors_dict,features_list,threshold=0.001,cmap=surf_cmap,title=title,vmax = len(feature_names))

    def banded_ridge_regression(self,outer_folds=10,inner_folds=5,num_alphas=50,backend='torch_cuda',regress_confounds=False,permutations=None):
        """
            This function performs the fitting and testing of the encoding model. Saves results in object variables.

            Parameters
            ----------
            outer_folds : int specifying how many cross validation folds for the outer loop of nested cross validation
            inner_folds: int specifying how many cross validation folds for the inner loop of nested cross validation
            num_alphas: int specifing how many alphas to try in the inner loop
            backend : str, backend for himalaya. could be 'torch_cuda' for use on GPU, or 'numpy' for CPU
            permutations: int, how many permutations to generate for the null distributions
        """

        backend_name = backend
        backend = set_backend(backend, on_error="raise")
        print(backend)

        solver = "random_search"
        solver_function = MultipleKernelRidgeCV.ALL_SOLVERS[solver]

        n_iter = self.random_search_n 

        alphas = np.logspace(-5,10,num_alphas)
        n_targets_batch = 10000 #the number of targets to compute at one time
        n_alphas_batch = 50 
        n_targets_batch_refit = 200

        #https://github.com/gallantlab/himalaya/blob/main/himalaya/kernel_ridge/_random_search.py
        solver_params = dict(n_iter=n_iter, alphas=alphas,
                             n_targets_batch=n_targets_batch,
                             n_alphas_batch=n_alphas_batch,
                             n_targets_batch_refit=n_targets_batch_refit,
                             local_alpha=True,
                             diagonalize_method='svd')

        n_delays = 5 #the number of time delays (in TRs) to use in the encoding model (widens the model)
        delayer = Delayer(delays=[x+1 for x in np.arange(n_delays)])#delays of 1.5-7.5 seconds
        
        preprocess_pipeline = make_pipeline(
            StandardScaler(with_mean=True, with_std=True), 
            delayer, 
            Kernelizer(kernel="linear"),
        )

        #data split
        n_splits_outer = outer_folds
        n_splits_inner = inner_folds
        n_samples = self.fMRI_data.shape[0] #should be 1921
        weight_estimates_sum = []
        performance_sum = []
        individual_feature_performance_sum = []
        individual_product_measure_sum = []
        features_preferred_delay_sum = []

        permuted_scores_list = []
        
        cv_type = 'temp_chunking' 
        #outer loop - 5 fold, 4 folds to get weight estimates and hyperparameters, 1 for evaluating the performance of the model, averaged across the 5 folds
        if cv_type=='temp_chunking': #chunking in continuous chunks (specified by self.chunklen) before splitting into train/test
            cv_outer = GroupKFold(n_splits=n_splits_outer)
            n_chunks = int(n_samples/self.chunklen)
            #set groups so that it chunks the data according to chunk len, and then the chunks are split into training and test
            groups = [str(ind) for ind in range(0,n_chunks) for x in range(0,self.chunklen)]
            if(len(groups)!=n_samples): #add final group that didn't divide evenly into chunklen if necessary
                print('adding outer stragglers')
                diff = n_samples-len(groups)
                groups.extend([str(n_chunks) for x in range(0,diff)])
            splits = cv_outer.split(X=range(0,n_samples),groups=groups)
        elif cv_type=='runs': #train on 1 run, test on the other
            end_run1 = self.included_data_fMRI[0][1]-self.included_data_fMRI[0][0]
            end_run2 = n_samples
            splits = [ (np.arange(0,end_run1+1),np.arange(end_run1+1,end_run2)), ((np.arange(end_run1+1,end_run2),np.arange(0,end_run1+1)) )]
        elif cv_type=='no_shuffle': #
            cv_outer = KFold(n_splits=n_splits_outer,shuffle=False)
            splits = cv_outer.split(X=range(0,n_samples))

        #### load feature spaces
        feature_names = self.model_features_dict[self.model].copy()

        weight_indices_to_save = []
        loaded_features = {}
        for ind,feature_space in enumerate(feature_names):
            # if(feature_space.split('_')[0] != 'run'):
            filepath = self.dir + '/features/'+self.features_dict[feature_space].lower()+'.csv'
            print(filepath)
            data = np.array(pd.read_csv(filepath,header=None)).astype(dtype="float32")
            print(data.shape)
            n_dim = data.shape[1]
            if feature_space in self.feature_weights_to_save:
                weight_indices_to_save.append(ind) #saving weights of specified feature spaces only
            loaded_features[feature_space] = data

        for i, (train_outer, test_outer) in enumerate(splits):
            print('starting cross-validation fold '+str(i+1) +'/'+str(n_splits_outer))
            # print(train_outer)
            Y_train = self.fMRI_data[train_outer]
            Y_test = self.fMRI_data[test_outer]

            Y_train = stats.zscore(Y_train)
            Y_test = stats.zscore(Y_test)
            Y_train = np.nan_to_num(Y_train)
            Y_test = np.nan_to_num(Y_test)
            
            features_train = []
            features_test = []
            features_n_list = []
            feature_names = self.model_features_dict[self.model].copy()
            for feature_space in feature_names:

                data = loaded_features[feature_space] #get the preloaded data
                train = data[train_outer].astype(dtype="float32")
                test = data[test_outer].astype(dtype="float32")
                n_dim = train.shape[1]
                
                features_train.append(train)
                features_test.append(test)
                features_n_list.append(n_dim)

            #add features (regressors) for the run that each data point was from, if there is more than one run
            #need to do this to account for mean differences between multiple runs since we are not normalizing within each run before concatenating the fMRI runs
            
            if len(self.included_data_fMRI)>1:
                train_run_regressors = []
                test_run_regressors = []
                startpoint = 0
                for ind,run in enumerate(self.included_data_fMRI):
                    endpoint = startpoint+run[1]-run[0]
                    
                    train_run_regressors.append(np.array([[1] if ((TR>=startpoint) & (TR<endpoint)) else [-1] for TR in train_outer]).astype(dtype="float32"))

                    #zero out the run weights for the test so we only get the performance for the stimulus related features!!
                    test_run_regressors.append(np.array([[0] for TR in test_outer]).astype(dtype="float32"))


                features_train.append(np.concatenate(train_run_regressors,1))
                features_test.append(np.concatenate(test_run_regressors,1))
                features_n_list.append(2)
                feature_names.append('run_'+str(ind+1))
                startpoint=endpoint+1

            ##### add in fMRIprep confounds as nuisance regressors ######
            if regress_confounds:
                train_confounds = self.confounds[train_outer].astype(dtype="float32")
                test_confounds = self.confounds[test_outer].astype(dtype="float32")
                
                features_train.append(train_confounds)
                features_test.append(np.zeros(test_confounds.shape)) #zero everything out for the test so we only get the performance of the stimulus related features
                features_n_list.append(train_confounds.shape[1])
                feature_names.append('fMRIprep_confounds')
            #############

            print("[features_n,...] =", features_n_list)
            # concatenate the feature spaces
            X_train = np.concatenate(features_train, 1)
            X_test = np.concatenate(features_test, 1)

            n_samples = X_train.shape[0]

            print("(n_samples_train, n_features_total) =", X_train.shape)
            print("(n_samples_test, n_features_total) =", X_test.shape)
            print("[features_n,...] =", features_n_list)

            start_and_end = np.concatenate([[0], np.cumsum(features_n_list)])
            slices = [
                slice(start, end)
                for start, end in zip(start_and_end[:-1], start_and_end[1:])
            ]

            kernelizers_tuples = [(name, preprocess_pipeline, slice_)
                                  for name, slice_ in zip(feature_names, slices)]
            column_kernelizer = ColumnKernelizer(kernelizers_tuples)
            
            #do temporal chunking for the inner loop as well
            cv_inner = GroupKFold(n_splits=n_splits_inner)
            n_chunks = int(n_samples/self.chunklen)
            
            groups = [str(ind) for ind in range(0,n_chunks) for x in range(0,self.chunklen)]
            if len(groups)!=n_samples: #add final group that didn't divide evenly into chunklen if necessary
                diff = n_samples-len(groups)
                groups.extend([str(n_chunks) for x in range(0,diff)])
            inner_splits = cv_inner.split(X=range(0,n_samples),groups=groups)

            mkr_model = MultipleKernelRidgeCV(kernels="precomputed", solver=solver,
                                              solver_params=solver_params, cv=inner_splits)

            pipeline = make_pipeline(
                column_kernelizer,
                mkr_model,
            )
            backend = set_backend(backend, on_error="raise")
            
            # put everything in float 32 for faster processing on GPU
            X_train = np.float32(X_train)
            X_test = np.float32(X_test)
            Y_train = np.float32(Y_train)
            Y_test = np.float32(Y_test)
            
            pipeline.fit(X_train, Y_train)
            print(pipeline)
            # scores_mask = pipeline.score(X_train, Y_train) #
            # print('avg whole brain train performance:' +str(np.nanmean(scores_mask)))
            # backend_name = 'numpy'
            # backend_ = set_backend(backend_name, on_error="raise")#put on CPU for more memory
            scores_mask = pipeline.score(X_test, Y_test) #
            scores_mask = backend.to_numpy(scores_mask)
            print("(n_voxels_mask,) =", scores_mask.shape)
            print('avg whole brain test performance:' +str(np.nanmean(scores_mask)))
            if i==0:
                performance_sum = scores_mask 
            else:
                performance_sum=performance_sum+scores_mask
            num_voxels = scores_mask.shape[0]
            del scores_mask
            # performance_list.append(s cores_mask)

            # disentangle the contribution of the two feature spaces -- individual feature space performances
            Y_test_pred_split = pipeline.predict(X_test, split=True)
            split_scores_mask_product_measure = r2_score_split(Y_test, Y_test_pred_split) #could also be correlation
            if(backend_name=='torch_cuda'):
                curr_ind_prod_measure = np.array([np.array(x.cpu()) for x in split_scores_mask_product_measure])
                # individual_product_measure_list.append([np.array(x.cpu()) for x in split_scores_mask_product_measure])
            else:
                curr_ind_prod_measure = np.array([np.array(x) for x in split_scores_mask_product_measure])
                # individual_product_measure_list.append([np.array(x) for x in split_scores_mask_product_measure])
            
            if i==0:
                individual_product_measure_sum = curr_ind_prod_measure
            else:
                individual_product_measure_sum = individual_product_measure_sum+curr_ind_prod_measure
            del curr_ind_prod_measure,split_scores_mask_product_measure

            #get just the raw performance of each individual feature
            if self.save_individual_feature_performance:
                split_scores_mask_ind_feature_perf = r2_score_split(Y_test, Y_test_pred_split,include_correlation=False) 
                if backend_name=='torch_cuda':
                    curr_ind_feature_perf = np.array([np.array(x.cpu()) for x in split_scores_mask_ind_feature_perf])
                    # individual_feature_performance_list.append([np.array(x.cpu()) for x in split_scores_mask_ind_feature_perf])
                else:
                    curr_ind_feature_perf = np.array([np.array(x) for x in split_scores_mask_ind_feature_perf])
                    # individual_feature_performance_list.append([np.array(x) for x in split_scores_mask_ind_feature_perf])

                if i==0:
                    individual_feature_performance_sum = curr_ind_feature_perf
                else:
                    individual_feature_performance_sum = individual_feature_performance_sum + curr_ind_feature_perf
                del curr_ind_feature_perf,split_scores_mask_ind_feature_perf
                del Y_test_pred_split

            if self.save_hemodynamic_fits:
                print('getting primal coefficients for time delays...')
                primal_coefs = mkr_model.get_primal_coef(column_kernelizer.get_X_fit())
                features_preferred_delay = np.zeros((len(feature_names),num_voxels))
                for ind,curr_coefs in enumerate(primal_coefs):
                    print(feature_names[ind],' getting hemodynamic fits')
                    #average over all of the dimensions in this feature space
                    primal_coef_per_delay_averaged_over_dim = torch.mean(torch.stack(np.array_split(curr_coefs, n_delays, axis=0)),dim=1)
                    features_preferred_delay[ind]=torch.argmax(primal_coef_per_delay_averaged_over_dim,dim=0).numpy()+1 #add one to shift from 0-4 to 1-5
                    del primal_coef_per_delay_averaged_over_dim
                if i==0:
                    features_preferred_delay_sum=features_preferred_delay
                else:
                    features_preferred_delay_sum=features_preferred_delay_sum+features_preferred_delay

                print(features_preferred_delay)
                del features_preferred_delay
                print('features_preferred_delay_sum.shape',features_preferred_delay_sum.shape)


            if self.save_weights:
                if self.save_hemodynamic_fits is False: #only load if these weren't loaded before
                    print('getting primal coefficients for saving weights...')
                    primal_coefs = mkr_model.get_primal_coef(column_kernelizer.get_X_fit())
                average_coefs =[]
                feature_names_to_save = []
                for ind,curr_coefs in enumerate(primal_coefs):
                    if ind in weight_indices_to_save:
                        feature_names_to_save.append(feature_names[ind])
                        primal_coef_per_delay = torch.stack(np.array_split(curr_coefs, n_delays, axis=0))
                        average_coefs.append(torch.mean(primal_coef_per_delay, dim=0).numpy())
                        del primal_coef_per_delay

                average_coefs = np.concatenate(average_coefs) #concatenate all weights, now averaged over the time delays
                #only keep the weights we selected to keep
                print('all saved weights',average_coefs.shape)

                if i==0:
                    weight_estimates_sum=average_coefs
                else:
                    weight_estimates_sum = weight_estimates_sum+average_coefs
                del primal_coefs
                del average_coefs

            if permutations is not None:
                print('shuffling Y_test to get null distribution, ',permutations,' permutations')
                #get kernel weights from the fitted model, to use in refitting the null model
                permuted_scores = np.zeros((permutations,num_voxels))
                for iteration in np.arange(0,permutations):
                    # shuffle BOLD time series in blocks of 10 and then correlate with the predicted time-series
                    # Block permutation preserves autocorrelation statistics of the time series (Kunsch, 1989) 
                    # and thus provides a sensible null hypothesis for these significance tests

                    #shuffle the BOLD time series in chunks to account for temporal autocorrelation
                    #similar to how they did it here: https://gallantlab.org/papers/Deniz.F.etal.2023.pdf
                    # Split the DataFrame into chunks
                    chunks = [Y_test[i:i + self.chunklen,:] for i in range(0, len(Y_test), self.chunklen)]
                    # Shuffle the chunks
                    np.random.shuffle(chunks)
                    # Concatenate the shuffled chunks
                    Y_test_chunked_and_shuffled = np.concatenate(chunks)
                    scores_mask = pipeline.score(X_test, Y_test_chunked_and_shuffled) #Y_test has been shuffled
                    scores_mask = backend.to_numpy(scores_mask)
                    # print('avg whole brain permuted test performance:' +str(np.nanmean(scores_mask)))
                    permuted_scores[iteration,:]=scores_mask
                    del scores_mask

                if i==0:
                    permuted_scores_list = permuted_scores
                else:
                    permuted_scores_list=permuted_scores_list+permuted_scores

            
            debugging=False
            if debugging:
                best_alphas = mkr_model.best_alphas_.cpu().numpy()
                ax = plot_alphas_diagnostic(best_alphas, alphas)
                plt.title("Best alphas selected by cross-validation")
                plt.savefig(self.sid+'_best_alphas.png')
                plt.close()
                
                cv_scores = mkr_model.cv_scores_.cpu().numpy()
                current_max = np.maximum.accumulate(cv_scores, axis=0)
                mean_current_max = np.mean(current_max, axis=1)
                x_array = np.arange(1, len(mean_current_max) + 1)
                ax = plt.plot(x_array, mean_current_max, '-o')
                plt.grid("on")
                plt.xlabel("Number of kernel weights sampled")
                plt.ylabel("L2 negative loss (higher is better)")
                plt.title("Convergence curve, averaged over targets")
                plt.tight_layout()
                plt.savefig(self.sid+'_cv_scores.png')
                plt.close()
            ##save memory by deleting variables
            del features_train, features_test
                
        if self.save_weights:
            print('finished with outer cross-validation')
            print('weights')
            average_weights = weight_estimates_sum/outer_folds
            print(average_weights.shape)
            self.beta_weights = average_weights
            self.final_weight_feature_names = feature_names_to_save

        if self.save_hemodynamic_fits:
            print('features_preferred_delay')
            average_features_preferred_delay = features_preferred_delay_sum/outer_folds
            print(average_features_preferred_delay.shape)
            self.features_preferred_delay = average_features_preferred_delay

        print('all features performance')
        average_performance = performance_sum/outer_folds
        print(average_performance.shape)
        self.performance = average_performance
        
        if self.save_individual_feature_performance:
            print('individual features performance')
            average_individual_feature_performance = individual_feature_performance_sum/outer_folds
            print(average_individual_feature_performance.shape)
            self.ind_feature_performance = average_individual_feature_performance

        print('individual product measure')
        average_individual_product_measure = individual_product_measure_sum/outer_folds
        print(average_individual_product_measure.shape)
        self.ind_product_measure = average_individual_product_measure


        if permutations is not None:
            print('all features performance null')
            average_performance_null = permuted_scores_list/outer_folds
            print(average_performance_null.shape)
            self.performance_null =  average_performance_null

        return

    def permutation_statistics(self):
        """
            This function performs the permutation test of the encoding model results with the null distribution generated by banded_ridge_regression()
            Saves results in object variables.
        """
        # individual analyses were conducted with a nonparametric permutation test 
        # to identify voxels showing significantly above chance prediction performance. 
        # conducted a sign permutation test (5000 iterations). From the empirical null distribution
        # of a prediction performance, one-tailed P values were calculated and adjusted with FDR correction. 
        # prediction performance maps of each model were thresholded at P FDR < 0.05 

        # null distribution was computed by:
        # shuffle BOLD time series in blocks of 10 and then correlate with the predicted time-series
        # Block permutation preserves autocorrelation statistics of the time series (Kunsch, 1989) 
        # and thus provides a sensible null hypothesis for these significance tests
        
        ### all features performance p-values
        def process(voxel_performance,voxel_null_distribution):
            #one-tailed t test for performance
            null_n = voxel_null_distribution.shape[0]
            null_n_over_sample = sum((voxel_null_distribution>voxel_performance).astype(int))
            p = null_n_over_sample/null_n
            return p

        self.perf_p_unc = np.array(Parallel(n_jobs=-1)(delayed(process)(voxel_performance,voxel_null_distribution) for (voxel_performance,voxel_null_distribution) in zip(self.performance, self.performance_null.T)))

        #perform FDR correction
        self.perf_fdr_reject,self.perf_p_fdr = fdrcorrection(self.perf_p_unc, alpha=0.05, method='n', is_sorted=False) #method is BY for dependence between variables, Benjamini/Yekutieli

    def plot_performance(self, label, threshold=None,vmin=None,vmax=None):

        file_label = self.sid+'_encoding_model-'+self.model + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
        if self.mask_name is not None:
            file_label = file_label + '_mask-'+self.mask_name

        output_filepath_surf = ''
        if label=='raw':
            filepath = self.out_dir+'/performance/'+file_label
            img = nibabel.load(filepath+'_measure-perf_raw.nii.gz')
            # cmap='cold_hot'
            title = self.sid
            output_filepath_surf = self.figure_dir + "/performance/" + file_label+'_measure-perf_'+label
        elif label=='stats':
            filepath = self.out_dir+'/perf_p_fdr/'+file_label
            img = nibabel.load(filepath+'_measure-perf_p_fdr.nii.gz')
            title = self.sid+', pvalue<'+str(threshold)
            # print(filepath+'_measure-perf_p_binary.nii.gz')
            #add a small number to each value so that zeroes are plotted!
            performance_p = img.get_fdata()
            threshold = 1-threshold
            # #mask the brain with significant pvalues
            # performance_p[performance_p>0.05]=-1
            performance_p[self.mask==1] = 1-performance_p[self.mask==1] #turn all significant voxels into high values for plotting
            affine = img.affine
            img = nibabel.Nifti1Image(performance_p, affine)
            cmap = 'Greys'
            output_filepath_surf = self.figure_dir + "/perf_p_fdr/" + file_label+'_measure-perf_'+label

        cmap = 'gray_inferno_symmetric'
        plotting_helpers.plot_surface(img,output_filepath_surf,threshold=threshold,vmax=vmax,vmin=vmin,cmap=cmap,title=title,symmetric_cbar=False,colorbar_label='Explained Variance $R^2$')

    def plot_ind_feature_performance(self, label, threshold=None,vmin=None,vmax=None):
        feature_names = self.model_features
        for (ind,feature_name) in enumerate(feature_names):
            print(feature_name)
            try:
                file_label = self.sid+'_encoding_model-'+self.model + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)

                if self.mask_name is not None:
                    file_label = file_label + '_mask-'+self.mask_name


                if label=='raw':
                    filepath = self.out_dir+'/ind_feature_performance/'+file_label
                    img = nibabel.load(filepath+'_measure-ind_perf_raw.nii.gz')
                    affine=img.affine
                    img_data=img.get_fdata()
                    img_data = img_data[ind]
                    img = nibabel.Nifti1Image(img_data,affine)
                    cmap='grey_inferno'
                    plot_label = self.sid + ', ' + label + ', ' + feature_name 
                    plot_threshold = threshold
                    output_filepath_surf = self.figure_dir + "/ind_feature_performance/" + file_label+'_feature-'+feature_name+'_measure-perf_'+label+'_surf.png'

                elif label=='stats':
                    filepath = self.out_dir+'/ind_perf_p_fdr/'+file_label
                    img = nibabel.load(filepath+'_measure-ind_perf_p_fdr.nii.gz')
                    plot_label = self.sid + ', ' + label + ', ' + feature_name + ', pvalue<'+str(threshold)
                    #add a small number to each value so that zeroes are plotted!
                    performance_p = img.get_fdata() 
                    #get the individual feature performance we are interested in
                    performance_p = performance_p[ind]
                    performance_p[self.mask==1] = 1-performance_p[self.mask==1] 
                    plot_threshold = 1-threshold
                    affine = img.affine
                    img = nibabel.Nifti1Image(performance_p, affine)
                    cmap = 'grey_inferno'
                    output_filepath_surf = self.figure_dir + "/ind_perf_p_fdr/" + file_label+'_feature-'+feature_name+'_measure-perf_'+label+'_surf.png'

                plotting_helpers.plot_surface(img,output_filepath_surf,threshold=plot_threshold,vmax=vmax,cmap=cmap,title=plot_label,symmetric_cbar=False,colorbar_label="Explained Variance $R^2$")
            except Exception as e:
                print(e)
                pass

    def plot_ind_product_measure(self, label, threshold=None,vmax=None,cmap='yellow_hot'):
        feature_names = self.model_features
        for (ind,feature_name) in enumerate(feature_names):#self.DNN_index]):
            try:
                print(feature_name)

                file_label = self.sid+'_encoding_model-'+self.model + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)

                if self.mask_name is not None:
                    file_label = file_label + '_mask-'+self.mask_name

                output_filepath_surf = ''
                plot_threshold = threshold
                plot_label = ''
                if label=='raw':
                    filepath = self.out_dir+'/ind_product_measure/'+file_label
                    img = nibabel.load(filepath+'_measure-ind_product_measure_'+label+'.nii.gz')
                    affine=img.affine
                    img_data=img.get_fdata()
                    img_data = img_data[ind]
                    img = nibabel.Nifti1Image(img_data,affine)
                    cmap='grey_inferno'
                    plot_label = self.sid + ', ' + label + ', ' + feature_name 
                    plot_threshold = threshold
                    output_filepath_surf = self.figure_dir + "/ind_product_measure/" + file_label+'_feature-'+feature_name+'_measure-ind_product_measure_'+label+'_surf.png'
                elif label=='stats':
                    filepath = self.out_dir+'/ind_prod_p_fdr/'+file_label
                    img = nibabel.load(filepath+'_measure-ind_prod_p_fdr.nii.gz')
                    plot_label = self.sid + ', ' + label + ', ' + feature_name + ', pvalue<'+str(threshold)
                    #add a small number to each value so that zeroes are plotted!
                    performance_p = img.get_fdata() 
                    #get the individual feature performance we are interested in
                    performance_p = performance_p[ind]
                    performance_p[self.mask==1] = 1-performance_p[self.mask==1] 
                    plot_threshold = 1-threshold
                    affine = img.affine
                    img = nibabel.Nifti1Image(performance_p, affine)
                    cmap = 'grey_inferno'
                    output_filepath_surf = self.figure_dir + "/ind_prod_p_fdr/" + file_label+'_feature-'+feature_name+'_measure-product_measure_'+label+'_surf.png'

                cmap_ = cmap
                plotting_helpers.plot_surface(img,output_filepath_surf,threshold=plot_threshold,vmax=vmax,cmap=cmap_,title=plot_label,symmetric_cbar=False,colorbar_label='Explained Variance $R^2$')
            except Exception as e:
                print(e)
                pass
    
    def save_results(self):
        file_label = self.sid+'_smoothingfwhm-'+str(self.smoothing_fwhm)
        if self.mask_name is not None:
            file_label = file_label + '_mask-'+self.mask_name
        file_label = self.sid+'_encoding_model-'+self.model + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
        if self.mask_name is not None:
            file_label = file_label + '_mask-'+self.mask_name 
        print(len(self.performance))
        
        if len(self.performance)>0:
            print('performance')
            performance = self.unmask_reshape(self.performance)
            img = nibabel.Nifti1Image(performance,self.affine)
            nibabel.save(img, self.out_dir+'/performance/'+file_label+'_measure-perf_raw.nii.gz') 
            print('saved: performance')
            print(img.shape)

        if self.save_weights:
            print(len(self.beta_weights))
            if(len(self.beta_weights)>0):
                print(self.beta_weights)
                # beta_weights = self.beta_weights#self.unmask_reshape(self.beta_weights)
                with h5py.File(self.out_dir+'/weights/'+file_label+'_measure-weights_raw.h5', 'w') as hf:
                    hf.create_dataset("weights",  data=self.beta_weights, compression='gzip',compression_opts=9)
                # img = nibabel.Nifti1Image(beta_weights, self.affine) #don't put back into full brain shape (will be too big)
                # nibabel.save(img, self.out_dir+'/weights/'+file_label+'_measure-weights_raw.nii.gz')  
                print('saved: beta_weights')
                print(img.shape)
        
        if len(self.ind_feature_performance)>0:
            print(self.ind_feature_performance.shape)
            ind_feature_performance = self.unmask_reshape(self.ind_feature_performance)
            img = nibabel.Nifti1Image(ind_feature_performance, self.affine)
            nibabel.save(img, self.out_dir+'/ind_feature_performance/'+file_label+'_measure-ind_perf_raw.nii.gz')
            print('saved: ind_feature_performance')
            print(img.shape)

        if len(self.ind_product_measure)>0:
            print('ind_product_measure')
            print(self.ind_product_measure.shape)
            ind_product_measure = self.unmask_reshape(self.ind_product_measure)
            img = nibabel.Nifti1Image(ind_product_measure, self.affine)
            nibabel.save(img, self.out_dir+'/ind_product_measure/'+file_label+'_measure-ind_product_measure_raw.nii.gz')
            print('saved: ind_product_measure')
            print(img.shape)

        if len(self.features_preferred_delay)>0:
            print('features_preferred_delay')
            print(self.features_preferred_delay.shape)
            features_preferred_delay = self.unmask_reshape(self.features_preferred_delay)
            img = nibabel.Nifti1Image(features_preferred_delay, self.affine)
            nibabel.save(img, self.out_dir+'/features_preferred_delay/'+file_label+'_measure-features_preferred_delay.nii.gz')
            print('saved: features_preferred_delay')
            print(img.shape)

        if len(self.performance_null)>0:
            print(self.performance_null.shape)
            # perf_p_unc = self.unmask_reshape(self.perf_p_unc) #don't put back into full brain shape (will be too big)
            with h5py.File(self.out_dir+'/performance/'+file_label+'_measure-perf_null_distribution.h5', 'w') as hf:
                hf.create_dataset("null_performance",  data=self.performance_null,compression='gzip',compression_opts=9)
            # img = nibabel.Nifti1Image(self.performance_null, self.affine)
            # nibabel.save(img, self.out_dir+'/performance/'+file_label+'_measure-perf_null_distribution.nii.gz')
            print('saved: perf_null_distribution')
            print(img.shape)

        if len(self.perf_p_unc)>0:
            print(self.perf_p_unc.shape)
            perf_p_unc = self.unmask_reshape(self.perf_p_unc)
            img = nibabel.Nifti1Image(perf_p_unc, self.affine)
            nibabel.save(img, self.out_dir+'/perf_p_unc/'+file_label+'_measure-perf_p_unc.nii.gz')
            print('saved: perf_p_unc')
            print(img.shape)
        if len(self.ind_perf_p_unc)>0:
            print(self.ind_perf_p_unc.shape)
            ind_perf_p_unc = self.unmask_reshape(self.ind_perf_p_unc)
            img = nibabel.Nifti1Image(ind_perf_p_unc, self.affine)
            nibabel.save(img, self.out_dir+'/ind_perf_p_unc/'+file_label+'_measure-ind_perf_p_unc.nii.gz')
            print('saved: ind_perf_p_unc')
            print(img.shape)
        if len(self.ind_prod_p_unc)>0:
            print(self.ind_prod_p_unc.shape)
            ind_prod_p_unc = self.unmask_reshape(self.ind_prod_p_unc)
            img = nibabel.Nifti1Image(ind_prod_p_unc, self.affine)
            nibabel.save(img, self.out_dir+'/ind_prod_p_unc/'+file_label+'_measure-ind_prod_p_unc.nii.gz')
            print('saved: ind_prod_p_unc')
            print(img.shape)

        if len(self.perf_p_fdr)>0:
            print(self.perf_p_fdr.shape)
            perf_p_fdr = self.unmask_reshape(self.perf_p_fdr)
            img = nibabel.Nifti1Image(perf_p_fdr, self.affine)
            nibabel.save(img, self.out_dir+'/perf_p_fdr/'+file_label+'_measure-perf_p_fdr.nii.gz')
            print('saved: perf_p_fdr')
            print(img.shape)
        if len(self.ind_perf_p_fdr)>0:
            print(self.ind_perf_p_fdr.shape)
            ind_perf_p_fdr = self.unmask_reshape(self.ind_perf_p_fdr)
            img = nibabel.Nifti1Image(ind_perf_p_fdr, self.affine)
            nibabel.save(img, self.out_dir+'/ind_perf_p_fdr/'+file_label+'_measure-ind_perf_p_fdr.nii.gz')
            print('saved: ind_perf_p_fdr')
            print(img.shape)
        if len(self.ind_prod_p_fdr)>0:
            print(self.ind_prod_p_fdr.shape)
            ind_prod_p_fdr = self.unmask_reshape(self.ind_prod_p_fdr)
            img = nibabel.Nifti1Image(ind_prod_p_fdr, self.affine)
            nibabel.save(img, self.out_dir+'/ind_prod_p_fdr/'+file_label+'_measure-ind_prod_p_fdr.nii.gz')
            print('saved: ind_prod_p_fdr')
            print(img.shape)

        if len(self.preference1_map)>0:
            print(self.preference1_map.shape)
            preference1_map = self.unmask_reshape(self.preference1_map) 
            print(preference1_map.shape)
            img = nibabel.Nifti1Image(preference1_map, self.affine)
            nibabel.save(img, self.out_dir+'/preference_map/'+file_label+'_measure-preference1_map.nii.gz')
            print('saved: preference1_map')
            print(img.shape)
        if len(self.preference2_map)>0:
            print(self.preference2_map.shape)
            preference2_map = self.unmask_reshape(self.preference2_map) 
            img = nibabel.Nifti1Image(preference2_map, self.affine)
            nibabel.save(img, self.out_dir+'/preference_map/'+file_label+'_measure-preference2_map.nii.gz')
            print('saved: preference2_map')
            print(img.shape)

        filename = self.out_dir+'/features/'+file_label+'_features.csv'  
        #save feature names as reference 
        with open(filename, 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(self.model_features)
        myfile.close()

        filename = self.out_dir+'/features/'+file_label+'_weight_features.csv'  
        #save feature names as reference 
        with open(filename, 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(self.final_weight_feature_names)
        myfile.close()

    def load_ind_product_measure(self):

        #load and mask data
        file_label = self.sid+'_encoding_model-'+self.model + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
        if(self.mask_name is not None):
            file_label = file_label + '_mask-'+self.mask_name 

        filepath = self.out_dir+'/ind_product_measure/'+file_label+'_measure-ind_product_measure_raw.nii.gz'

        nii = nibabel.load(filepath)
        nii_data = nii.get_fdata()
        self.load_mask()
        self.ind_product_measure = nii_data[:,(self.mask==1)]
        self.affine=nii.affine
        
    def run(self):
        testing = False
        permutations = None
        print('save_weights',self.save_weights)
        self.load_preprocess_fMRI(smooth=True,denoise=False)
        self.trim_fMRI()

        if testing:
            self.random_search_n = 100
            self.banded_ridge_regression(outer_folds=3, inner_folds=3, num_alphas=5,backend='torch_cuda',permutations=100)
            self.permutation_statistics()
        else:
            self.random_search_n = 1000
            self.banded_ridge_regression(outer_folds=5, inner_folds=5,num_alphas=25,permutations=permutations,regress_confounds=True)
            # self.permutation_statistics()

        # self.compute_preference_maps(restricted=True,threshold=0.01)
        self.save_results()
        self.plot_performance(label='raw',threshold=0.00000001,vmin=0,vmax=None)
        # self.plot_ind_feature_performance(label='raw',threshold=0.0000001,vmin=0,vmax=None)
        # self.plot_ind_product_measure(label='raw',threshold=0.0000001)
        # self.plot_preference_maps(label='first')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=str, default='1')
    parser.add_argument('--task','-task',type=str,default='sherlock')
    parser.add_argument('--space','-space',type=str, default='MNI152NLin2009cAsym_res-2')
    parser.add_argument('--mask','-mask',type=str, default='ISC')
    parser.add_argument('--smoothing-fwhm','-smoothing-fwhm',type=float,default=3.0)
    parser.add_argument('--chunklen','-chunklen',type=int,default=20)
    parser.add_argument('--model','-model',type=str, default=None)
    parser.add_argument('--save-weights','-save-weights',type=int, default=0)
    parser.add_argument('--testing','-testing',type=str,default=None)

    parser.add_argument('--dir', '-dir', type=str,
                        default='/Users/hsmall2/Documents/GitHub/naturalistic-multimodal-movie')
    parser.add_argument('--data_dir', '-data_dir', type=str,
                        default='/Users/hsmall2/Documents/GitHub/Sherlock_ASD/data') #where the derivatives folder for the fMRI data is 
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/hsmall2/Documents/GitHub/naturalistic-multimodal-movie/analysis')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/hsmall2/Documents/GitHub/naturalistic-multimodal-movie/figures')
    args = parser.parse_args()
    print(args)
    EncodingModel(args).run()

if __name__ == '__main__':
    main()