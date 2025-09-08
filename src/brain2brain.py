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
from sklearn.random_projection import SparseRandomProjection
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
from src import encoding,helpers

class Brain2Brain(encoding.EncodingModel):
    def __init__(self, args):
        self.testing = args.testing #specify if you are in testing mode (True) or not (False). Testing mode makes the encoding go faster (fewer cv folds and hyperparameters to choose from)
        self.process = 'Brain2Brain'
        self.dir = args.dir #highest level directory of the BIDS style dataset
        self.data_dir = args.data_dir #data directory of the BIDS style dataset
        self.out_dir = args.out_dir + "/" + self.process #output directory of the BIDS style directory, probably 'analysis'
        self.figure_dir = args.figure_dir +'/'+self.process #figure output directory of the BIDS style directory
        self.type = args.type
        self.sid = args.s_num
        self.population = args.population
        self.task = args.task #the task name of the fmri data that will be used for the encoding model
        self.space = args.space
        self.mask_name = args.mask #the name of the brain mask to use
        self.mask = None #the variable for the mask data to be loaded into (see: load_mask())
        self.mask_affine = None #the variable for the mask affine fata to be loaded into (see: load_mask())
        self.smoothing_fwhm = args.smoothing_fwhm #the smoothing fwhm (full-width at half maximum) of the gaussian used for smoothing brain data
        self.chunklen = args.chunklen #the number of TRs to group the fmri data into (used during encoding model to take care of temporal autocorrelation in the data)
        self.fMRI_data = [] #variable to store the fmri data in, will be [subject X, subject Y]
        self.brain_shape = None #the brain shape that all masks and data will be resampled to. this will be the output size of nii.gz data
        self.affine = [] #variable to store the affine of the fmri data

        # Sherlock specific timing parameters
        self.wait_TR = 2 #number of TRs from the beginning with no data (pause before the movie)
        self.stim_start = 26 #TR of the start of the stimulus we are interested in
        self.intro_start = self.wait_TR # TR of the start of the introductory movie
        self.intro_end = self.stim_start-1 # TR of the end of the introductory movie
        self.repeated_data_fMRI = [(self.intro_start,self.intro_end),(953+self.intro_start,953+self.intro_end)] #TR indices of repeated fMRI data to compute the explainable variance with. every item should indicate the indices of the same stimulus. there should be one entry in this list for every run that will be included
        self.included_data_fMRI = [(self.stim_start+self.wait_TR,946+self.wait_TR),(975+self.wait_TR,1976+self.wait_TR)] #TR indices of the fmri data to use in encoding model, there should be one entry in this list for every run that will be included #Haemy had 27:946, and 973:1976, have to subtract one for python 0-indexing and add one to end bc exclusive slicing in python
        self.confounds = []
        self.save_individual_feature_performance = True

        #variables to store results in:
        self.performance = []
        self.ind_feature_performance = []
        self.performance_null = []
        self.perf_p_unc = []
        
        #creation of necessary output folders:
        Path(f'{self.out_dir}/').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"performance"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"intersubject_correlation"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"ind_feature_performance"}').mkdir(exist_ok=True, parents=True)
       
        Path(f'{self.figure_dir}/{"performance"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"intersubject_correlation"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"ind_feature_performance"}').mkdir(exist_ok=True, parents=True)
        #define other subjects as all subjects from specified population except for this target subject
        self.subjects = [subject for subject in helpers.get_subjects(self.population)['sherlock'] if subject!=self.sid]
        
        
    def trim_fMRI(self,norm=False):
        """ This function trims the fmri data to the data that will be used in the encoding model. 
            It will take the stored fmri data (already concatenated across any runs by load_smooth_denoise_fMRI()) and trim the specified parts (self.included_data_fMRI)
        """
        print('...trimming fMRI...')
        #trim TRs from the concatenated data (concatenated over runs)
        for fMRI_data,confounds in zip(self.fMRI_data,self.confounds):
            trimmed_data = np.array([])
            trimmed_confounds = []
            run_ends = []
            for section in self.included_data_fMRI:
                start = section[0]
                stop = section[1]
                new_data = np.array(fMRI_data)[start:stop] #first dimension should be n samples
                new_confounds = confounds[start:stop]
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
            """ finds all runs of the task (assuming BIDS format), iteratively loads, smooths, concatenates them in ascending order 
                and saves the result into the object's fMRI_data """
            #assuming BIDS format
            img = os.path.join(self.data_dir,'derivatives',self.sid,'func',self.sid+'_task-'+self.task+'_run-*_space-'+self.space+'_desc-preproc_bold.nii.gz')
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
            if self.mask_name is not None:
                #mask the fmri data (only first 3 dim)
                self.load_mask()
                mask = (mask==1) & (self.mask==1)
                self.affine = self.mask_affine #change affine to line up fmri data and the mask

                self.fMRI_data.append(fmri_data[mask].T)
            else:
                self.mask = whole_brain_mask_data
                self.affine = whole_brain_mask.affine
                self.fMRI_data.append(fmri_data[self.mask==1].T)
            self.confounds.append(confounds_data)
                
    def load_mask(self):
        """ This function loads the brain mask that specifies which voxels we are running the encoding model on. The mask is saved as a ndarray in the object
        """
        self.mask = helpers.load_mask(self,self.mask_name)
        self.mask_affine = self.mask.affine
        self.mask = self.mask.get_fdata()
            
            
    def unmask_reshape(self, data):
        """ This function puts the given data back into the whole brain (unmasking) and reshapes it back into the specified brain dimensions (self.brain_shape).
            It accounts for any masks that could have been used during analysis, including anatomical and explainable variance masks. 
            It also accounts for multidimensional data (for instance, individual feature performance could have more than one feature in the data)
        """
        mask = np.ones(self.brain_shape)
        mask = mask+self.mask

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

    def banded_ridge_regression(self,outer_folds=10,inner_folds=5,num_alphas=50,backend='torch_cuda',regress_confounds=False,permutations=None,subjects_in_separate_feature_spaces=False,regress_all_subject_confounds=True):
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

        n_delays = 2 #the number of time delays (in TRs) to use in the encoding model (widens the model)
        delayer = Delayer(delays=[x for x in np.arange(n_delays)])#delays of 0-7.5 seconds
        
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
        permuted_ind_perf_scores_list = []
        permuted_ind_product_scores_list = []
        
        cv_type = 'temp_chunking'
        #outer loop - 10 fold, 9 folds to get weight estimates and hyperparameters, 1 for evaluating the performance of the model, averaged across 10
        if cv_type=='temp_chunking':
            cv_outer = GroupKFold(n_splits=n_splits_outer)
            n_chunks = int(n_samples/self.chunklen)
            #set groups so that it chunks the data according to chunk len, and then the chunks are split into training and test
            groups = [str(ind) for ind in range(0,n_chunks) for x in range(0,self.chunklen)]
            if(len(groups)!=n_samples): #add final group that didn't divide evenly into chunklen if necessary
                print('adding outer stragglers')
                diff = n_samples-len(groups)
                groups.extend([str(n_chunks) for x in range(0,diff)])
            splits = cv_outer.split(X=range(0,n_samples),groups=groups)
        elif cv_type=='runs':
            end_run1 = self.included_data_fMRI[0][1]-self.included_data_fMRI[0][0]
            end_run2 = n_samples
            splits = [ (np.arange(0,end_run1+1),np.arange(end_run1+1,end_run2)), ((np.arange(end_run1+1,end_run2),np.arange(0,end_run1+1)) )]
        elif cv_type=='no_shuffle':
            cv_outer = KFold(n_splits=n_splits_outer,shuffle=False)
            splits = cv_outer.split(X=range(0,n_samples))

        loaded_features = {}
        loaded_confounds = {}
        print(self.subjects)
        for ind,subject in enumerate(self.subjects):
            # if(feature_space.split('_')[0] != 'run'):
            filepath = os.path.join(self.dir,'analysis','TimeSeries',f"{subject}_timeseries_smoothingfwhm-{self.smoothing_fwhm}_srp_mask-{self.mask_name}.h5")
            f = h5py.File(filepath,'r')
            data = f['data'][()]
            n_dim = data.shape[1]
            loaded_features[subject] = data
            
            filepath = os.path.join(self.dir,'analysis','TimeSeries',f"{subject}_confounds.csv")
            data = np.array(pd.read_csv(filepath,header=None)).astype(dtype="float32")
            loaded_confounds[subject] = data
        
        for i, (train_outer, test_outer) in enumerate(splits):
            print('starting cross-validation fold '+str(i+1) +'/'+str(n_splits_outer))
            # print(train_outer)
            Y_train = self.fMRI_data[train_outer]
            Y_test = self.fMRI_data[test_outer]

            Y_train = stats.zscore(Y_train)
            Y_test = stats.zscore(Y_test)
            Y_train = np.nan_to_num(Y_train)
            Y_test = np.nan_to_num(Y_test)
            print('Y_train.shape',Y_train.shape)
            
            features_train = []
            features_test = []
            features_n_list = []
            for subject in self.subjects:
                data = loaded_features[subject] #get the preloaded data
                train = data[train_outer].astype(dtype="float32")
                test = data[test_outer].astype(dtype="float32")
                n_dim = train.shape[1]
                
                features_train.append(train)
                features_test.append(test)
                features_n_list.append(n_dim)
            
            if(subjects_in_separate_feature_spaces):
                feature_names_list = self.subjects.copy()
            else:
                features_train = [np.concatenate(features_train,axis=1)]
                features_test = [np.concatenate(features_test,axis=1)]
                features_n_list = [np.sum(features_n_list)]
                feature_names_list = ['other_subjects']
                
            #add features (regressors) for the run that each data point was from, if there is more than one run
            #need to do this to account for mean differences between multiple runs since we are not normalizing within each run before combining 
            
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
                feature_names_list.append('run_'+str(ind+1))
                startpoint=endpoint+1

            ##### add in fMRIprep confounds as nuisance regressors ######
            if regress_confounds:
                train_confounds = [self.confounds[train_outer].astype(dtype="float32")]
                test_confounds = [self.confounds[test_outer].astype(dtype="float32")]
                
                if(regress_all_subject_confounds):
                    for subject in self.subjects:
                        data = loaded_confounds[subject] #get the preloaded data
                        train = data[train_outer].astype(dtype="float32")
                        test = data[test_outer].astype(dtype="float32")
                        n_dim = train.shape[1]
                        
                        train_confounds.append(train)
                        test_confounds.append(test)
                train_confounds = np.concatenate(train_confounds,axis=1)
                test_confounds = np.concatenate(test_confounds,axis=1)
                
                features_train.append(train_confounds)
                features_test.append(np.zeros(test_confounds.shape)) #zero everything out for the test so we only get the performance of the stimulus related features
                features_n_list.append(train_confounds.shape[1])
                feature_names_list.append('fMRIprep_confounds')

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
                                  for name, slice_ in zip(feature_names_list, slices)]
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
            
            print("Any NaNs in Y_train?", np.isnan(Y_train).any())
            print("Y_train preview:", Y_train[:5])  # Print first few rows
            
            print('Y_train.shape',Y_train.shape)
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
            
            Y_test_pred_split = pipeline.predict(X_test, split=True)
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

            # fast_permutations = True

            if permutations is not None:
                print('shuffling Y_test to get null distribution, ',permutations,' permutations')
                #get kernel weights from the fitted model, to use in refitting the null model
                deltas = mkr_model.deltas_
                permuted_scores = np.zeros((permutations,num_voxels))
                # permuted_ind_product_scores = []
                # permuted_ind_perf_scores = []
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
                    # Y_test_chunked = Y_test.reshape(-1,self.chunklen,Y_test.shape[1])
                    # np.random.shuffle(Y_test_chunked) #breaking the relationship between feature and BOLD series in the test set!
                    # print('permuted Y_test.shape', Y_test_chunked_and_shuffled.shape)
                    #null hypothesis: any observed relationship between the features and the brain responses is due to chance

                    # if(fast_permutations):
                    #     null_pipeline = pipeline #if we want to do this faster, don't fit a whole new model, just use other model with shuffled Y train
                    # else:
                    #     #fit the model again, but with the pre-specified best hyperparameter for this model (best alpha)
                    #     #deltas are np.nan or infinity...
                    #     deltas = torch.from_numpy(np.nan_to_num(deltas.cpu())) #put all np.nan's to zero and infinity to large numbers
                    #     null_model = WeightedKernelRidge(alpha=1, deltas=deltas, kernels="precomputed")

                    #     null_pipeline = make_pipeline(
                    #         column_kernelizer,
                    #         null_model,
                    #     )
                    #     null_pipeline.fit(X_train, Y_train) #fitting new model with shuffled Y train and alphas from the correct fit model
                    
                    scores_mask = pipeline.score(X_test, Y_test_chunked_and_shuffled) #Y_test has been shuffled
                    scores_mask = backend.to_numpy(scores_mask)
                    # print('avg whole brain permuted test performance:' +str(np.nanmean(scores_mask)))
                    permuted_scores[iteration,:]=scores_mask
                    del scores_mask
                    
                    # Y_test_pred_split = null_pipeline.predict(X_test, split=True)
                    # split_scores_mask_product_measure = r2_score_split(Y_test, Y_test_pred_split) #could also be r2
                    # if(backend_name=='torch_cuda'):
                    #     permuted_ind_product_scores.append([np.array(x.cpu()) for x in split_scores_mask_product_measure])
                    # else:
                    #     permuted_ind_product_scores.append([np.array(x) for x in split_scores_mask_product_measure])

                    # #get just the raw performance of each individual feature
                    # split_scores_mask_ind_feature_perf = r2_score_split(Y_test, Y_test_pred_split,include_correlation=False) #could also be r2
                    # if(backend_name=='torch_cuda'):
                    #     permuted_ind_perf_scores.append([np.array(x.cpu()) for x in split_scores_mask_ind_feature_perf])
                    # else:
                    #     permuted_ind_perf_scores.append([np.array(x) for x in split_scores_mask_ind_feature_perf])
                if i==0:
                    permuted_scores_list = permuted_scores
                else:
                    permuted_scores_list=permuted_scores_list+permuted_scores
                # permuted_ind_perf_scores_list.append(permuted_ind_perf_scores)
                # permuted_ind_product_scores_list.append(permuted_ind_product_scores)

            
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

        print('all features performance')
        average_performance = performance_sum/outer_folds
        print(average_performance.shape)
        self.performance = average_performance
        if self.save_individual_feature_performance:
            print('individual features performance')
            average_individual_feature_performance = individual_feature_performance_sum/outer_folds
            print(average_individual_feature_performance.shape)
            self.ind_feature_performance = average_individual_feature_performance
        
        if permutations is not None:
            print('all features performance null')
            average_performance_null = permuted_scores_list/outer_folds
            print(average_performance_null.shape)
            self.performance_null =  average_performance_null

            # print('individual features performance null')
            # individual_feature_performance_null = np.array(permuted_ind_perf_scores_list)    
            # print(individual_feature_performance_null.shape) #should be 10, n voxels
            # average_individual_feature_performance_null = np.mean(individual_feature_performance_null,axis=0)
            # print(average_individual_feature_performance_null.shape)

            
            # print('individual product measure null')
            # individual_product_measure_null = np.array(permuted_ind_product_scores_list)    
            # print(individual_product_measure_null.shape) #should be 10, n voxels
            # average_individual_product_measure_null = np.mean(individual_product_measure_null,axis=0)
            # print(average_individual_product_measure_null.shape)
        
           
            # self.ind_feature_performance_null = average_individual_feature_performance_null
            # self.ind_product_measure_null = average_individual_product_measure_null

            # #plot histograms of null distribution
            # import seaborn as sns
            # flat = average_performance_null.flatten()
            # sns.histplot(flat,stat='percent')
            # # plt.ylim((0,2))
            # plt.savefig('testing_perf.png')
            # plt.close()

            # flat = average_individual_feature_performance_null.flatten()
            # sns.histplot(flat,stat='percent')
            # # plt.ylim((0,2))
            # plt.savefig('testing_ind_feat_perf.png')
            # plt.close()

            # flat = average_individual_product_measure_null.flatten()
            # sns.histplot(flat,stat='percent')
            # # plt.ylim((0,))
            # plt.savefig('testing_ind_prod.png')
            # plt.close()

        # print(self.performance)
        # print(self.performance_null)
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

        # all_ind_feature_performance_null = np.transpose(self.ind_feature_performance_null,(1,0,2)) #reshape so first dimension is the features

        # ind_perf_p_unc_list = []
        # ind_perf_p_fdr_list = []
        # ind_perf_p_fdr_reject_list = []
        # for ind_feature_performance,ind_feature_performance_null in zip(self.ind_feature_performance,all_ind_feature_performance_null):
        #     def process(voxel_performance,voxel_null_distribution):
        #         #one-tailed t test for performance
        #         null_n = voxel_null_distribution.shape[0]
        #         null_n_over_sample = sum((voxel_null_distribution>voxel_performance).astype(int))
        #         p = null_n_over_sample/null_n
        #         return p

        #     ind_perf_p_unc = np.array(Parallel(n_jobs=-1)(delayed(process)(voxel_performance,voxel_null_distribution) for (voxel_performance,voxel_null_distribution) in zip(ind_feature_performance, ind_feature_performance_null.T)))
            
        #     # #fdr correction
        #     ind_perf_fdr_reject,ind_perf_p_fdr = fdrcorrection(ind_perf_p_unc, alpha=0.05, method='n', is_sorted=False) #method is BY for dependence between variables
            
        #     ind_perf_p_unc_list.append(ind_perf_p_unc)
        #     ind_perf_p_fdr_list.append(ind_perf_p_fdr)
        #     ind_perf_p_fdr_reject_list.append(ind_perf_fdr_reject)
        # self.ind_perf_p_unc = np.array(ind_perf_p_unc_list)
        # self.ind_perf_p_fdr = np.array(ind_perf_p_fdr_list)
        # self.ind_perf_p_fdr_reject = np.array(ind_perf_p_fdr_reject_list)

        # all_ind_product_measure_null = np.transpose(self.ind_product_measure_null,(1,0,2)) #reshape so first dimension is the features
        # ind_prod_p_unc_list = []
        # ind_prod_p_fdr_list = []
        # ind_prod_p_fdr_reject_list = []
        # for ind_product_measure,ind_product_measure_null in zip(self.ind_product_measure,all_ind_product_measure_null):
        #     def process(voxel_performance,voxel_null_distribution):
        #         #one-tailed t test for performance
        #         null_n = voxel_null_distribution.shape[0]
        #         null_n_over_sample = sum((voxel_null_distribution>voxel_performance).astype(int))
        #         p = null_n_over_sample/null_n
        #         return p

        #     ind_prod_p_unc = np.array(Parallel(n_jobs=-1)(delayed(process)(voxel_performance,voxel_null_distribution) for (voxel_performance,voxel_null_distribution) in zip(ind_product_measure, ind_product_measure_null.T)))
            
        #     # #fdr correction
        #     ind_prod_fdr_reject,ind_prod_p_fdr = fdrcorrection(ind_prod_p_unc, alpha=0.05, method='n', is_sorted=False) #method is BY for dependence between variables
            
        #     ind_prod_p_unc_list.append(ind_prod_p_unc)
        #     ind_prod_p_fdr_list.append(ind_prod_p_fdr)
        #     ind_prod_p_fdr_reject_list.append(ind_prod_fdr_reject)
        # self.ind_prod_p_unc = np.array(ind_prod_p_unc_list)
        # self.ind_prod_p_fdr = np.array(ind_prod_p_fdr_list)
        # self.ind_prod_p_fdr_reject = np.array(ind_prod_p_fdr_reject_list)

    def leave_one_out_correlation(self,circle_shift=False):
        target_subject = self.fMRI_data.copy() #save target subject's data
        filepath = os.path.join(self.dir,'analysis','TimeSeries',f"{self.sid}_timeseries_smoothingfwhm-{self.smoothing_fwhm}_denoised_normed.h5")
        f = h5py.File(filepath,'r+')
        target_subject = f['data'][()].T.astype('float32')
        print('target_subject',target_subject.shape)
        # for each subject, average all other subjects time series and then correlate with the subject
        predictor_subjects = []
        for subject in self.subjects:
            filepath = os.path.join(self.dir,'analysis','TimeSeries',f"{subject}_timeseries_smoothingfwhm-{self.smoothing_fwhm}_denoised_normed.h5")
            f = h5py.File(filepath,'r+')
            predictor_subjects.append(f['data'][()].T.astype('float16'))
        print(predictor_subjects)
        if(circle_shift): 
            #get random amount to shift by
            shift_values = np.random.randint(0, target_subject.shape[0], size=len(self.subjects))
            print('shift:', shift_values)
            #shift each subject by a different shift value, replacing the original data to be memory efficient
            for i,subject in enumerate(self.subjects):
                predictor_subjects[i] = np.roll(predictor_subjects[i], shift_values[i])
        mat1 = target_subject
        mat2 = np.nanmean(np.array(predictor_subjects),axis=0)
        
        #standardize time-series
        mat1_standardized = mat1#already z-scored(mat1 - np.mean(mat1, axis=1, keepdims=True)) / np.std(mat1, axis=1, ddof=1, keepdims=True)
        mat2_standardized = (mat2 - np.mean(mat2, axis=1, keepdims=True)) / np.std(mat2, axis=1, keepdims=True)
        # Calculate correlation using matrix multiplication
        correlation_matrix = np.einsum('ij,ij->i', mat1_standardized, mat2_standardized) / (mat1.shape[1] - 1)
        explained_variance = correlation_matrix**2 * np.sign(correlation_matrix) #r squared, maintain sign
        self.performance = explained_variance
  
    def plot_performance(self, label, threshold=None,vmin=None,vmax=None):

        file_label = self.sid+'_brain2brain_'+self.type+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
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

        cmap = 'gray_inferno'#self.cmaps['yellow_hot']
        helpers.plot_surface(img,output_filepath_surf,threshold=threshold,vmax=vmax,vmin=vmin,cmap=cmap,title=title,symmetric_cbar=False,colorbar_label='Explained Variance $R^2$')
    
    def save_results(self):
        file_label = self.sid+'_brain2brain_'+self.type+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
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
        if len(self.ind_feature_performance)>0:
            print(self.ind_feature_performance.shape)
            ind_feature_performance = self.unmask_reshape(self.ind_feature_performance)
            img = nibabel.Nifti1Image(ind_feature_performance, self.affine)
            nibabel.save(img, self.out_dir+'/ind_feature_performance/'+file_label+'_measure-ind_perf_raw.nii.gz')
            print('saved: ind_feature_performance')
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
        
    def run(self):
        testing = False
        permutations = None
        if(self.type=='encoding'):
            self.load_preprocess_fMRI(smooth=True,denoise=False)
            self.trim_fMRI()
            if testing:
                self.random_search_n = 100
                self.banded_ridge_regression(outer_folds=3, inner_folds=3, num_alphas=5,backend='numpy',permutations=100)
                self.permutation_statistics()
            else:
                self.random_search_n = 1000 
                self.banded_ridge_regression(outer_folds=5, inner_folds=5,num_alphas=25,permutations=permutations,regress_confounds=True,subjects_in_separate_feature_spaces=False,regress_all_subject_confounds=True)
                # self.permutation_statistics()
        elif(self.type=='correlation'):
            self.load_preprocess_fMRI(smooth=True,denoise=True) #load whole brain mask
            self.leave_one_out_correlation()
            # self.leave_one_out_correlation(circle_shift=True)
            
        self.save_results()
        self.plot_performance(label='raw',threshold=0.01,vmin=None,vmax=None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s_num', type=str, default='1')
    parser.add_argument('--population', '-population', type=str, default='NT')
    parser.add_argument('--task','-task',type=str,default='sherlock')
    parser.add_argument('--space','-space',type=str, default='MNI152NLin2009cAsym_res-2')
    parser.add_argument('--mask','-mask',type=str, default=None)
    parser.add_argument('--smoothing-fwhm','-smoothing-fwhm',type=float,default=3.0)
    parser.add_argument('--chunklen','-chunklen',type=int,default=20)
    parser.add_argument('--testing','-testing',type=str,default=None)
    parser.add_argument('--type','-type',type=str,default='ISC')

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
    Brain2Brain(args).run()

if __name__ == '__main__':
    main()