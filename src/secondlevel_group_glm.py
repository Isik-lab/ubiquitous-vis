import os
import argparse
from pathlib import Path
import json

import numpy as np
import nibabel

from joblib import Parallel, delayed
from statsmodels.stats.multitest import fdrcorrection

from nilearn.glm.second_level import make_second_level_design_matrix
from nilearn.glm.second_level import SecondLevelModel
from nilearn.image import threshold_img
from nilearn.glm import threshold_stats_img #for statistical testing

from src import plotting_helpers
from src import glm
from src import helpers

class SecondLevelGroup(glm.GLM):
    def __init__(self, args):
        self.process = 'SecondLevelGroup'
        self.dir = args.dir
        self.data_dir = args.dir + '/data'
        self.in_dir = args.out_dir + '/GLM'
        self.out_dir = args.out_dir + "/" + self.process
        self.subjects = []
        self.population = args.population
        self.sid = 'sub-'+self.population
        self.task = args.task
        self.space = args.space
        self.smoothing_fwhm = args.smoothing_fwhm #change?
        self.fMRI_data = []
        self.brain_shape = []
        self.affine = []
        # self.weights = []
        # self.contrast_z_scores = []
        self.figure_dir = args.figure_dir + "/" + self.process
        Path(f'{self.out_dir}/{"glm_weights"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"glm_zscores"}').mkdir(exist_ok=True, parents=True)
        
        Path(f'{self.figure_dir}/{"glm_weights"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"glm_zscores"}').mkdir(exist_ok=True, parents=True)
        self.file_label ='_smoothingfwhm-'+str(self.smoothing_fwhm)

        #load the parameter file for the specified glm
        params_filepath =os.path.join(self.dir,'scripts','glm_'+self.task+'.json')
        with open(params_filepath) as json_file:
            glm_params = json.load(json_file)
        json_file.close()

        run_groups = glm_params['run_groups']
        self.run_group = next(iter(run_groups)) #the first run group specified should be using all of the runs
        self.contrasts = glm_params['contrasts']
        self.subjects = helpers.get_subjects(self.population)
        self.cmaps = plotting_helpers.get_cmaps()


    def compile_data(self):
        # self.load_mask()
        #weights and contrast z-scores
        print(self.subjects)
        
        label = 'zscore'
        zscore_dict = {}
        for contrast in self.contrasts:
            all_data= []
            subjects_included = []
            for subject in self.subjects[self.task]:
                try:
                    # print(subject)
                    filepath = os.path.join(self.in_dir, subject,subject+ "_task-"+ self.task+'_space-'+self.space+ "_run-"+ self.run_group +  "_measure-"+label+ "_contrast-"+contrast+".nii.gz")
                    nii = nibabel.load(filepath)
                    affine = nii.affine
                    all_data.append(nii)
                    subjects_included.append(subject)
                    #
                except Exception as e:
                    print(e)
                    pass

            # all_data = np.array(all_data)
            ## TODO use nilearn to do a group level analysis with stats!!
            #create confounds for the second level group analysis
            design_matrix = make_second_level_design_matrix(
                subjects_included, ##could include extra info here like age, https://nilearn.github.io/dev/auto_examples/05_glm_second_level/plot_second_level_design_matrix.html#sphx-glr-auto-examples-05-glm-second-level-plot-second-level-design-matrix-py
            )
            second_level_model = SecondLevelModel(smoothing_fwhm=self.smoothing_fwhm, n_jobs=2)
            second_level_model = second_level_model.fit(
                all_data, design_matrix=design_matrix
            )
            z_map = second_level_model.compute_contrast(output_type="z_score")
            zscore_dict[contrast] = z_map
            self.brain_shape = z_map.shape
            self.affine = affine

            nibabel.save(z_map, os.path.join(self.out_dir,'glm_zscores',self.sid+self.file_label+'_measure-'+label+'_contrast-'+contrast+'.nii.gz'))

        label = 'weights'
        weights_dict = {}
        for contrast in self.contrasts:
            all_data= []
            for subject in self.subjects[self.task]:
                try:
                    filepath = os.path.join(self.in_dir, subject, subject+ "_task-"+ self.task+'_space-'+self.space+ "_run-"+ self.run_group + "_measure-"+label+ "_contrast-"+contrast+".nii.gz")
                    nii = nibabel.load(filepath)
                    data = nii.get_fdata()
                    affine = nii.affine
                    all_data.append(data)
                except Exception as e:
                    print(e)
                    pass

            all_data = np.array(all_data)
            all_data_mean = np.nanmean(all_data,axis=0)
            weights_dict[contrast]=all_data_mean
            self.brain_shape = all_data_mean.shape
            self.affine = affine

            img = nibabel.Nifti1Image(all_data_mean, affine)
            nibabel.save(img, os.path.join(self.out_dir,'glm_weights',self.sid+self.file_label+'_measure-'+label+'_contrast-'+contrast+'.nii.gz'))

        self.zscore_dict = zscore_dict
        self.weights_dict = weights_dict

    def plot_weights(self,threshold=0.01,vmin=None,vmax=None):
        
        for contrast in self.contrasts:
            filepath = os.path.join(self.out_dir,'glm_weights',self.sid+self.file_label+'_measure-weights_contrast-'+contrast+'.nii.gz')
            img = nibabel.load(filepath)
                
            # if(feature_name=='DNN_6'): #only plot up to DNN 5
            # 	break
            print(contrast)

            plot_filepath = os.path.join(self.figure_dir,"glm_weights", self.sid+self.file_label+'_measure-weights_contrast-'+contrast)
            # threshold=None
            vmax=None
            title=contrast
            cmap = 'yellow_hot'
            # helpers.plot_img_volume(weights,filepath,threshold,vmax)
            plotting_helpers.plot_surface(img,plot_filepath,threshold=threshold,vmax=vmax,title=title,symmetric_cbar=False,cmap=cmap,colorbar_label='weight')

    def plot_zscores(self,FDR_correction=True,threshold=None,vmin=None,vmax=None,symmetric_cbar=True,cmap='yellow_hot'):

        for contrast in self.contrasts:
            print(contrast)
            filepath = os.path.join(self.out_dir,'glm_zscores',self.sid+self.file_label+'_measure-zscore_contrast-'+contrast+'.nii.gz')
            img = nibabel.load(filepath)
            if FDR_correction:
                thresholded_map, threshold_corrected = threshold_stats_img(
                    img, alpha=threshold, height_control="fdr",two_sided=False #only care about positive direction?
                        )
            else:
                threshold_dict = {0.05: 1.72,
                                  0.01: 2.5,
                                  0.001: 3.6}
                thresholded_map, threshold_corrected = threshold_stats_img(
                    img, alpha=threshold_dict[threshold], height_control=None,two_sided=False #only care about positive direction?
                        )

            title=''

            plot_filepath = os.path.join(self.figure_dir, "glm_zscores", self.sid+self.file_label+'_measure-zscore_contrast-'+contrast)
            plotting_helpers.plot_surface(thresholded_map,plot_filepath,threshold=0.0000000001,vmin=vmin,vmax=vmax,cmap=cmap,title=title,symmetric_cbar=symmetric_cbar,colorbar_label='z-score')
            #threshold is very small number for surface visualization since we are plotting an already thresholded map
            
    def run(self):

        print(self.subjects)
        self.compile_data()
        print('compiled data')

        self.plot_weights('raw',threshold=0.000001,vmax=None)
        self.plot_zscores('raw',threshold=0.01,vmax=None)

        if(self.do_stats):
            self.plot_performance('stats',threshold=0.00001)
        # self.plot_ind_feature_performance('stats',threshold=0.05)
        # self.plot_weights('raw')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task','-task',type=str,default='SIpointlights')
    parser.add_argument('--mask','-mask',type=str, default=None)
    parser.add_argument('--smoothing-fwhm','-smoothing-fwhm',type=float,default=3.0)
    parser.add_argument('--chunklen','-chunklen',type=int,default=30)
    parser.add_argument('--feature-of-interest','-feature-of-interest',type=str,default='None')
    parser.add_argument('--population','-population',type=str,default='NT')


    parser.add_argument('--dir', '-dir', type=str,
                        default='/Users/hsmall2/Documents/GitHub/naturalistic-multimodal-movie')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/hsmall2/Documents/GitHub/naturalistic-multimodal-movie/analysis')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/hsmall2/Documents/GitHub/naturalistic-multimodal-movie/figures')
    args = parser.parse_args()
    SecondLevelGroup(args).run()

if __name__ == '__main__':
    main()