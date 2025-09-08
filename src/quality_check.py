import argparse
import glob
from pathlib import Path

import numpy as np
from scipy import stats
import pandas as pd

import glob

# custom python modules
from src import helpers

class QualityCheck:
    def __init__(self, args):
        self.testing = args.testing #specify if you are in testing mode (True) or not (False). testing mode makes the encoding go faster (cv is less)
        self.process = 'QualityCheck'
        self.dir = args.dir #highest level directory of the BIDS style dataset
        self.data_dir = args.data_dir #data directory of the BIDS style dataset
        self.out_dir = args.out_dir + "/" + self.process #output directory of the BIDS style directory, probably 'analysis'
        self.figure_dir = args.figure_dir +'/'+self.process #figure output directory of the BIDS style directory
        self.population = args.population
        self.sid = 'sub-'+self.population #creating the BIDS style subject id from the given subject number
        self.task = args.task #the task name of the fmri data that will be used for the encoding model
        self.space = args.space
        self.mask_name = args.mask #the name of the brain mask to use
        self.mask = None #the variable for the mask data to be loaded into (see: load_mask())
        self.mask_affine = None #the variable for the mask affine fata to be loaded into (see: load_mask())
        self.smoothing_fwhm = args.smoothing_fwhm #the smoothing fwhm (full-width at half maximum) of the gaussian used for smoothing brain data
        self.chunklen = args.chunklen #the number of TRs to group the fmri data into (used during encoding model to take care of temporal autocorrelation in the data)
        self.model = args.model #the name of the encoding model to run
        self.wait_TR = 2 #number of TRs from the beginning with no data (pause before the movie)
        self.stim_start = 26 #TR of the start of the stimulus we are interested in
        self.intro_start = self.wait_TR # TR of the start of the introductory movie
        self.intro_end = self.stim_start-1 # TR of the end of the introductory movie
        self.repeated_data_fMRI = [(self.intro_start,self.intro_end),(953+self.intro_start,953+self.intro_end)] #TR indices of repeated fMRI data to compute the explainable variance with. every item should indicate the indices of the same stimulus. there should be one entry in this list for every run that will be included
        self.included_data_fMRI = [(self.stim_start+self.wait_TR,946+self.wait_TR),(975+self.wait_TR,1976+self.wait_TR)] #TR indices of the fmri data to use in encoding model, there should be one entry in this list for every run that will be included #Haemy had 27:946, and 973:1976, have to subtract one for python 0-indexing and add one to end bc exclusive slicing in python

        self.explainable_variance_cutoff = 0.1 #voxelwise cutoff for the explainable variance map
        self.threshold = args.motion_threshold
        
        self.subjects = helpers.get_subjects(self.population)
        
        #creation of necessary output folders:
        Path(f'{self.out_dir}/').mkdir(exist_ok=True, parents=True)
            
    def trim(self,data,norm=False):
        """ This function trims the fmri data to the data that will be used in the encoding model. 
            It will take the stored fmri data (already concatenated across any runs by load_smooth_fMRI()) and trim the specified parts (self.included_data_fMRI)
        """
        #trim TRs from the concatenated data (concatenated over runs)
        trimmed_data = np.array([])
        run_ends = []
        for section in self.included_data_fMRI:
            start = section[0]
            stop = section[1]
            new_data = np.array(data)[start:stop] #first dimension should be n samples
            if(norm):
                new_data = stats.zscore(new_data,axis=0,nan_policy='omit') #zscore the responses across the samples of each section(from different runs) separately !!
            if(trimmed_data.shape[0]<1):
                trimmed_data = new_data
            else:
                trimmed_data = np.concatenate((trimmed_data,new_data),axis=0) #concatenate along n samples dimension
        return trimmed_data

    def load_motion_data(self,subject):

        #assuming BIDS format
        filepath = self.data_dir + '/derivatives/'+ subject+'/func/'+subject+'_task-'+self.task+'_run-*_desc-confounds_timeseries.tsv'
        runs = []
        for file in glob.glob(filepath):
            runs.append(file)
        runs.sort() #sort the runs to be ascending order
        #concatenate runs
        data = np.array([])
        for run in runs:
            print('loading ..run '+str(run))
            temp = pd.read_csv(run, sep='\t', header=0)#load the timeseries and get the framewise displacement
            temp = temp['framewise_displacement']
            if(data.shape[0]<1): #if it's the first run, initialize
                data = temp
            else: #concatenate runs together
                data = np.concatenate([data,temp])

        return data        
        
    def run(self):
        
        all_data_prop = {}
        all_data_avg = {}
        for subject in self.subjects[self.task]:
            data = self.load_motion_data(subject)
            trimmed_data = self.trim(data,norm=False)
            if(len(trimmed_data)>0):
                proportion_over = len(trimmed_data[trimmed_data>self.threshold])/len(trimmed_data)
                all_data_prop[subject] = proportion_over
                all_data_avg[subject] = np.mean(trimmed_data)
                
        results = pd.DataFrame(all_data_prop.items(),columns=['subject','proportion_over_'+str(self.threshold)+'mm'])
        results.to_csv(self.out_dir+'/proportion_motion_spikes_'+self.population+'.tsv')
        print(results)
        
        results = pd.DataFrame(all_data_avg.items(),columns=['subject','average_FD'])
        results.to_csv(self.out_dir+'/avg_FD_'+self.population+'.tsv')
        print(results)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population', '-p', type=str, default='NT')
    parser.add_argument('--task','-task',type=str,default='sherlock')
    parser.add_argument('--space','-space',type=str, default='MNI152NLin2009cAsym_res-2') #'native'
    parser.add_argument('--mask','-mask',type=str, default='ISC')
    parser.add_argument('--smoothing-fwhm','-smoothing-fwhm',type=float,default=3.0)
    parser.add_argument('--chunklen','-chunklen',type=int,default=20)
    parser.add_argument('--model','-model',type=str, default=None)
    parser.add_argument('--testing','-testing',type=str,default=None) 
    parser.add_argument('--motion-threshold','-motion-threshold',type=int,default=0.9)
    
    parser.add_argument('--dir', '-dir', type=str,
                        default='/home/hsmall2/scratch4-lisik3/hsmall2/naturalistic-multimodal-movie')
    parser.add_argument('--data_dir', '-data_dir', type=str,
                        default='/home/hsmall2/scratch4-lisik3/Sherlock_ASD/data') #where the derivatives folder for the fMRI data is 
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/home/hsmall2/scratch4-lisik3/hsmall2/naturalistic-multimodal-movie/analysis')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/home/hsmall2/scratch4-lisik3/hsmall2/naturalistic-multimodal-movie/figures')
    args = parser.parse_args()
    QualityCheck(args).run()

if __name__ == '__main__':
    main()