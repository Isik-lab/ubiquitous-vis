import os
import argparse
from pathlib import Path
import json

import pandas as pd
import numpy as np
import nibabel

import nilearn
import nilearn.datasets
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import run_glm, mean_scaling
from nilearn.glm.contrasts import compute_contrast

from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt

from src import helpers

class GLM:
    def __init__(self, args):
        self.process = 'GLM'
        self.dir = args.dir
        self.data_dir = args.data_dir
        self.s_num = str(int(args.s_num)).zfill(2)
        self.sid = 'sub-'+self.s_num
        self.task = args.task
        self.space = args.space
        self.affine = []
        self.brain_shape = None
        self.smoothing_fwhm = args.smoothing_fwhm
        self.slice_time_ref = 0.5
        self.mask_name = args.mask #NOTE: GLM runs in whole brain but results are masked by this parameter
        self.events = []
        self.imgs = []
        self.confounds = []
        self.metadata = []
        self.design_matrix = []
        self.subject_label = self.sid+ "_task-"+ self.task +'_space-'+self.space
        self.out_dir = os.path.join(args.out_dir,self.process,self.sid)
        self.figure_dir = os.path.join(args.figure_dir,self.process,self.sid)
        Path(self.out_dir).mkdir(exist_ok=True, parents=True)
        Path(self.figure_dir).mkdir(exist_ok=True, parents=True)

        #load the parameter file for the specified glm
        params_filepath = 'glm_'+self.task+'.json'
        with open(params_filepath) as json_file:
            glm_params = json.load(json_file)
        json_file.close()

        self.run_groups = glm_params['run_groups']
        self.contrasts = glm_params['contrasts']

        self.n_runs = glm_params['n_runs']
        self.plot_design_matrix = False

    def __str__(self):
        pass

    def load_subject_data(self):
        print('loading subject data...')
        #assumes BIDS layout

        base_file = self.sid + '_task-'+self.task

        #explicit paths is the fastest on the cluster...
        event_files = []
        img_files = []
        confound_files = []
        metadata_files = []
        for run in range(1,self.n_runs+1):
            event_files.append(os.path.join(self.data_dir,self.sid,'func',base_file+'_run-0'+str(run)+'_events.tsv'))
            img_files.append(os.path.join(self.data_dir,'derivatives',self.sid,'func',base_file+'_run-'+str(run)+'_space-'+self.space+'_res-2_desc-preproc_bold.nii.gz'))
            confound_files.append(os.path.join(self.data_dir,'derivatives',self.sid,'func',base_file+'_run-'+str(run)+'_desc-confounds_timeseries.tsv'))
            metadata_files.append(os.path.join(self.data_dir,'derivatives',self.sid,'func',base_file+'_run-'+str(run)+'_space-'+self.space+'_res-2_desc-preproc_bold.json'))
        
        events = []
        imgs = []
        confounds = []
        metadata = []
        for (event,img,confound,metadatum) in zip(event_files,img_files,confound_files,metadata_files):
            events.append(pd.read_csv(event,sep='\t'))
            imgs.append(str(img))
            confounds.append(pd.read_csv(confound,sep='\t'))
            with open(metadatum) as json_file:
                metadata.append(json.load(json_file))

        self.events = events
        self.imgs = imgs
        self.confounds = confounds
        self.metadata = metadata
        
        print(events)
        print(imgs)
        print(confounds)
        print(metadata)


    def get_contrast_values(self, contrast_id):
        # get the basic contrasts of the design first
        contrast = self.contrasts[contrast_id]
        contrast_matrix = np.eye(self.design_matrix.shape[1])
        basic_contrasts = dict(
            [(column, contrast_matrix[i]) for i, column in enumerate(self.design_matrix.columns)]
        )

        # get the first contrast, assumes that the contrasts are subtraction at the lowest level, connected by addition at higher level
        split = contrast.split("&")
        full_split = [x.split("-") for x in split]
        # initialize with first contrast
        contrast_values = basic_contrasts[full_split[0][0]]
        if len(full_split[0]) > 1:
            # on lowest level, subtraction
            contrast_values = contrast_values - basic_contrasts[full_split[0][1]]
        if len(full_split) > 1:
            for x in full_split[1:]:
                curr_contrast_values = basic_contrasts[x[0]]
                if len(x) > 1:
                    # on lowest level, subtraction
                    curr_contrast_values = curr_contrast_values - basic_contrasts[x[1]]
                # on higher level, addition
                contrast_values = contrast_values + curr_contrast_values

        return contrast_values

    def run_glm(self):
        for group_id in self.run_groups.keys():
            textures = []
            design_matrices = []

            ##### MASKING imgs -- MNI template, whole_brain
            
            # nilearn.masking.apply_mask(self.imgs,self.mask)
            for run in self.run_groups[group_id]:  # runs
                run = run - 1
                img = self.imgs[run]
                event = self.events[run]
                confound = self.confounds[run]
                metadata = self.metadata[run]
                
                img = nibabel.load(img) #load the numpy array
                print(img.shape)
                #set the brain_shape 
                self.brain_shape = img.shape[:-1]
                self.affine = img.affine

                #masking and smoothing
                print('...whole brain masking and smoothing with gaussian fwhm='+str(self.smoothing_fwhm)+'...')
                whole_brain_mask = nilearn.masking.compute_brain_mask(img,mask_type='whole-brain')
                masked_smoothed_data = nilearn.masking.apply_mask([img],whole_brain_mask,smoothing_fwhm=self.smoothing_fwhm)
                print(masked_smoothed_data.shape)
                print('...mean scaling....')
                fMRI_data = np.array(mean_scaling(masked_smoothed_data)[0])
                print(fMRI_data.shape)
                
                textures.append(fMRI_data)
                ### FILTER CONFOUNDS from fmriprep preprocessing ###############################################

                # 6 rigid-body transformations, FD, and aCompCor components
                confounds_to_use = ['rot_x', 
                                    'rot_y',
                                    'rot_z',
                                    'trans_x',
                                    'trans_y',
                                    'trans_z',
                                    'framewise_displacement',
                                    'a_comp_cor_00',
                                    'a_comp_cor_01',
                                    'a_comp_cor_02',
                                    'a_comp_cor_03',
                                    'a_comp_cor_04',
                                    'cosine00'
                                    ]
                confound = confound[confounds_to_use].fillna(0)
                confounds_matrix = confound.values
                confounds_names = confound.columns.tolist()

                ### CREATE DESIGN MATRIX with events and confound noise regressors #############################
                n_scans = fMRI_data.shape[0]
                # need to shift frame times because of slice timing correction in fmriprep preprocessing
                # https://reproducibility.stanford.edu/slice-timing-correction-in-fmriprep-and-linear-modeling/
                t_r = metadata["RepetitionTime"]
                frame_times = (t_r * ( np.arange(n_scans) + 0.7))
                
                #make sure trial_type is not read as an integer
                event = event.astype({'trial_type': '<U11'})
                
                design_matrix = make_first_level_design_matrix(
                    frame_times,
                    events=event,
                    hrf_model="glover + derivative",
                    add_regs=confounds_matrix,
                    add_reg_names=confounds_names
                )
                design_matrices.append(design_matrix)

            full_design_matrix = pd.concat(design_matrices)
            full_design_matrix.fillna(0, inplace=True)

             # plot design matrix for debugging later
            if(self.plot_design_matrix):
                fig, ax1 = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
                plot_design_matrix(full_design_matrix, ax=ax1)
                plt.savefig(self.out_dir + self.sid + "_task-"+ self.task+ "_run-"+ group_id
                                + "_glm_design_matrix.svg")
                plt.close()

            self.design_matrix = full_design_matrix

            ### FIT THE GLM on the data  ###########################################
            labels, estimates = run_glm(np.concatenate(textures), full_design_matrix.values)

            ### COMPUTE CONTRASTS #########################################################################
            for contrast_id in self.contrasts.keys():
                  
                contrast_values = self.get_contrast_values(contrast_id)
                print(contrast_id)
                print(contrast_values)
                contrast = compute_contrast(
                    labels, estimates, contrast_values, contrast_type="t"
                )

                if(self.mask_name!='None'):
                    mask_img = helpers.load_mask(self.mask_name)
                    resampled_mask = nilearn.image.resample_img(mask_img, target_affine=self.affine, target_shape=self.brain_shape,interpolation='nearest')
                    mask = nilearn.masking.apply_mask([resampled_mask],whole_brain_mask)

                    #only use the mask inside the whole brain mask for this subject
                    flattened_mask = mask.astype(int).flatten()
                    print(flattened_mask)

                    z_scores = np.zeros(contrast.z_score().shape)
                    weights = np.zeros(contrast.effect.flatten().shape)
                    p_values = np.zeros(contrast.p_value().shape)

                    print(z_scores.shape)
                    print(flattened_mask.shape)

                    z_scores[flattened_mask==1] = contrast.z_score()[flattened_mask==1]
                    weights[flattened_mask==1] = contrast.effect.flatten()[flattened_mask==1]

                    #fdr correction within the mask
                    unc_p_values_masked = contrast.p_value()[flattened_mask==1]
                    reject, p_values_masked = fdrcorrection(unc_p_values_masked,alpha=0.05, method='n', is_sorted=False)
                    p_values[flattened_mask==1] = p_values_masked


                else:
                    z_scores = contrast.z_score()
                    weights = contrast.effect.flatten()
                    unc_p_values = contrast.p_value()
                    #fdr correction
                    reject, p_values = fdrcorrection(unc_p_values, alpha=0.05, method='n', is_sorted=False)

                #put back into 3D space!!
                z_scores_img = nilearn.masking.unmask(z_scores,whole_brain_mask)
                weights_img = nilearn.masking.unmask(weights,whole_brain_mask)
                p_values_img = nilearn.masking.unmask(p_values,whole_brain_mask)

                print('SHAPE')
                print(z_scores_img)
                
                ##### SAVE DATA IMG ########
                z_scores_name = os.path.join(self.out_dir,self.sid+ "_task-"+ self.task+'_space-'+self.space+ "_run-"+ group_id +"_measure-zscore_contrast-"+contrast_id+ ".nii.gz")
                nibabel.save(z_scores_img, z_scores_name)

                weights_name = os.path.join(self.out_dir,self.sid+ "_task-"+ self.task+'_space-'+self.space+ "_run-"+ group_id +"_measure-weights_contrast-"+contrast_id+ ".nii.gz")
                nibabel.save(weights_img, weights_name)

                p_values_name = os.path.join(self.out_dir,self.sid+ "_task-"+ self.task+'_space-'+self.space+ "_run-"+ group_id +"_measure-pvalue_contrast-"+contrast_id+ ".nii.gz")
                nibabel.save(p_values_img, p_values_name)

    def run(self):
        self.load_subject_data()

        self.run_glm()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s_num', '-s', type=str, default='1')
    parser.add_argument('--task','-task',type=str,default='SIpointlights')
    parser.add_argument('--space','-space',type=str,default='MNI152NLin2009cAsym')
    parser.add_argument('--smoothing-fwhm','-smoothing-fwhm',type=float,default=0)
    parser.add_argument('--mask','-mask',type=str,default='None')
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
    GLM(args).run()

if __name__ == '__main__':
    main()
