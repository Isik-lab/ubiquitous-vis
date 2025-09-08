# I/O packages
import os
import argparse
import glob
from pathlib import Path
import nibabel
import nibabel.processing
import h5py

# data manipulation
import numpy as np
import pandas as pd
import nilearn
import nilearn.datasets
import nilearn.masking
import nilearn.signal
from scipy import stats

# for srp reduction
from sklearn.random_projection import SparseRandomProjection

from src import helpers

class TimeSeries:
    def __init__(self, args):
        self.process = 'TimeSeries'
        self.dir = args.dir #highest level directory of the BIDS style dataset
        self.data_dir = args.data_dir #data directory of the BIDS style dataset
        self.out_dir = args.out_dir + "/" + self.process #output directory of the BIDS style directory, probably 'analysis'
        self.figure_dir = args.figure_dir +'/'+self.process #figure output directory of the BIDS style directory
        self.sid = args.s_num
        self.task = args.task #the task name of the fmri data that will be used for the encoding model
        self.space = args.space
        self.mask_name = args.mask #the name of the brain mask to use
        self.mask = None #the variable for the mask data to be loaded into (see: load_mask())
        self.mask_affine = None #the variable for the mask affine fata to be loaded into (see: load_mask())
        self.smoothing_fwhm = args.smoothing_fwhm #the smoothing fwhm (full-width at half maximum) of the gaussian used for smoothing brain data
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
        self.srp = bool(args.srp)
        self.denoise = bool(args.denoise)
        self.norm_runs = bool(args.norm_runs)
        
        #creation of necessary output folders:
        Path(f'{self.out_dir}/').mkdir(exist_ok=True, parents=True)
        
        
    def trim_fMRI(self,norm=False):
        """ This function trims the fmri data to the data that will be used in the encoding model. 
            It will take the stored fmri data (already concatenated across any runs by load_smooth_denoise_fMRI()) and trim the specified parts (self.included_data_fMRI)
        """
        print('...trimming fMRI...')
        #trim TRs from the concatenated data (concatenated over runs)
        all_trimmed_data = []
        all_trimmed_confounds = []
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
            all_trimmed_data.append(trimmed_data)
            all_trimmed_confounds.append(np.array(trimmed_confounds))
            
        self.fMRI_data = np.squeeze(np.array(all_trimmed_data))
        self.confounds = all_trimmed_confounds
        self.run_ends = run_ends

        print('after trimming',self.fMRI_data.shape)

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

    def srp_reduction(self):
        #SRP reduce subject's voxelwise responses
        srp = SparseRandomProjection()
        print(self.fMRI_data.shape)
        if(self.fMRI_data.shape[1]>=6480):
            self.fMRI_data = srp.fit_transform(self.fMRI_data)
        else:
            self.fMRI_data = self.fMRI_data
        
    def save_results(self):
        file_label = self.sid+'_timeseries_smoothingfwhm-'+str(self.smoothing_fwhm)
        if(self.denoise):
            file_label = file_label + '_denoised'
        if(self.norm_runs):
            file_label = file_label + '_normed'
        if(self.srp):
            file_label = file_label + '_srp'
        if self.mask_name is not None:
            file_label = file_label + '_mask-'+self.mask_name 
        
        print(np.array(self.fMRI_data.shape))
        
        if len(self.fMRI_data)>0:
            filepath = os.path.join(self.out_dir,file_label+'.h5')
            # df = pd.DataFrame(self.fMRI_data)
            # df.to_csv(filepath,index=False,header=False)
            with h5py.File(filepath, 'w') as hf:
                hf.create_dataset("data",  data=self.fMRI_data.astype('float32'), compression='gzip',compression_opts=9)
                

            filepath = os.path.join(self.out_dir,self.sid+'_confounds.csv')
            df = pd.DataFrame(self.confounds[0])
            df.to_csv(filepath,index=False,header=False)
            
        print('saving complete')
        
    def run(self):
        self.load_preprocess_fMRI(smooth=True,denoise=self.denoise)
        self.trim_fMRI(self.norm_runs)
        if(self.srp):
            self.srp_reduction()
        self.save_results()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s_num', type=str, default='1')
    parser.add_argument('--task','-task',type=str,default='sherlock')
    parser.add_argument('--space','-space',type=str, default='MNI152NLin2009cAsym_res-2')
    parser.add_argument('--mask','-mask',type=str, default=None)
    parser.add_argument('--smoothing-fwhm','-smoothing-fwhm',type=float,default=3.0)
    parser.add_argument('--chunklen','-chunklen',type=int,default=20)
    parser.add_argument('--testing','-testing',type=str,default=None)
    parser.add_argument('--srp','-srp',type=int,default=0)
    parser.add_argument('--denoise','-denoise',type=int,default=0)
    parser.add_argument('--norm_runs','-norm_runs',type=int,default=0)

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
    TimeSeries(args).run()

if __name__ == '__main__':
    main()