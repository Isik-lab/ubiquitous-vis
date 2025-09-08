# I/O packages
import os
import argparse
from pathlib import Path
import csv
import nibabel

# data manipulation packages
import numpy as np
import pandas as pd
import nilearn

from joblib import Parallel, delayed
from statsmodels.stats.multitest import fdrcorrection
from brainiak.isc import permutation_isc
from scipy.stats import wilcoxon

# custom python modules
from src import plotting_helpers
from src import helpers
from src import encoding

class SecondLevelGroup(encoding.EncodingModel):
    def __init__(self, args):
        self.process = 'SecondLevelGroup'
        self.dir = args.dir
        self.data_dir = args.dir + '/data'
        self.in_dir = args.out_dir + '/EncodingModel'
        self.in_dir_GLM = args.out_dir + '/SecondLevelIndividual'
        self.out_dir = args.out_dir + "/" + self.process
        self.subjects = []
        self.population = args.population
        self.sid = 'sub-'+self.population
        self.task = args.task
        self.mask = args.mask
        self.mask_name = self.mask
        self.model = args.model
        self.feature_of_interest = args.feature_of_interest
        self.smoothing_fwhm = args.smoothing_fwhm #change?
        self.chunklen = args.chunklen
        self.fMRI_data = []
        self.brain_shape = []
        self.affine = []
        self.feature_names = []
        self.features = []
        self.weights = []
        self.save_weights = False
        self.model_performance = []
        self.model_performance_null_distribution = []
        self.figure_dir = args.figure_dir + "/" + self.process
        Path(f'{self.out_dir}/{"performance"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"ind_feature_performance"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"ind_product_measure"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"added_variance"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"unique_variance"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"spatial_correlation"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"preference_map"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"features"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"perf_p_unc"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"perf_p_fdr"}').mkdir(exist_ok=True, parents=True)

        Path(f'{self.figure_dir}/{"performance"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"ind_feature_performance"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"ind_product_measure"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"added_variance"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"unique_variance"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"spatial_correlation"}').mkdir(exist_ok=True, parents=True)

        Path(f'{self.figure_dir}/{"preference_map"}').mkdir(exist_ok=True, parents=True)

        self.beta_weights = []
        self.predicted_time_series = []
        self.performance = []
        self.performance_null_distribution = []
        self.perf_p_fdr = []
        self.ind_feature_performance = []
        self.ind_product_measure = []
        self.added_variance = []
        self.ind_product_measure_proportion = []
        self.ind_feature_perf_p_fdr = []
        self.ind_feature_performance_null_distribution =[]
        self.preference1_map = []
        self.preference2_map = []
        self.model_features = []
        self.perf_p_unc = []
        self.perf_fdr_reject = []
        self.perf_p_fdr = []
        self.ind_perf_p_unc = []
        self.ind_perf_fdr_reject = []
        self.ind_perf_p_fdr = []
        self.ind_prod_p_unc = []
        self.ind_prod_fdr_reject = []
        self.ind_prod_p_fdr = []
        self.performance_null = []
        self.features_preferred_delay=[]
        self.final_weight_feature_names=[]

        self.subjects = helpers.get_subjects(self.population)
        self.models_dict = helpers.get_models_dict()
        self.cmaps = plotting_helpers.get_cmaps()
        self.colors_dict = plotting_helpers.get_colors_dict()


        self.file_label = '_encoding_model-'+self.model + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
        if(self.model=='cross_subject'):
            self.in_dir = os.path.join(self.dir,'analysis','Brain2Brain')
            self.file_label = '_brain2brain_encoding_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
        if(self.model=='correlation'):
            self.in_dir = os.path.join(self.dir,'analysis','Brain2Brain')
            self.file_label = '_brain2brain_correlation_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
        if(self.mask_name!=''):
            self.file_label = self.file_label +'_mask-'+self.mask
        
        self.extra_label = '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)+'_mask-'+self.mask
        self.file_label_glm = '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_mask-'+self.mask

        self.combined_features = helpers.get_combined_features()

    def get_feature_index(self, subject,feature, selection_model=''):
        if selection_model=='':
            file_label = 'encoding_model-'+self.model+self.extra_label
        else:
            file_label = 'encoding_model-'+selection_model+self.extra_label
        filename = self.in_dir+'/features/'+file_label+'_features.csv'
        file = open(filename, "r")
        data = list(csv.reader(file, delimiter=','))[0]
        file.close()
        return data.index(feature)

    def compile_data(self,measures=['performance','ind_product_measure']):
        model_features = self.model.split('_')
        if self.feature_of_interest is not None:
            sub_model_features = [feature for feature in model_features if feature!=self.feature_of_interest]
            #always alphabetical, ignore the case
            sub_model_features = sorted(sub_model_features, key=str.casefold)
            sub_model = '_'.join(sub_model_features)
        else:
            sub_model=self.model

        if('performance' in measures):
            label = 'perf_raw'
            all_data_performance = []
            for subject in self.subjects[self.task]:
                try:
                    nii = nibabel.load(self.in_dir+'/performance/'+subject+self.file_label+'_measure-'+label+'.nii.gz')
                    if(self.model=='correlation'):
                        nii = nibabel.processing.smooth_image(nii,fwhm=0.0,mode='nearest') #no additional smoothing for ISC
                    else:
                        nii = nibabel.processing.smooth_image(nii,fwhm=self.smoothing_fwhm,mode='nearest')
                    performance = nii.get_fdata()
                    performance[performance<0] = 0 #clip negative responses to 0
                    affine = nii.affine
                    all_data_performance.append(performance)
                    # print(subject)
                except Exception as e:
                    print(e)
                    pass
            all_data_performance = np.array(all_data_performance)
            if(self.model=='correlation'):
                #square correlations to get explained variance (to compare to R^2 of encoding models)
                all_data_performance = all_data_performance**2 * np.sign(all_data_performance)
                #Fisher z transform before averaging!!!
                all_data_performance = np.arctanh(all_data_performance)
                #average
                model_performance = np.nanmean(np.array(all_data_performance),axis=0)
                #inverse fisher z transform!
                model_performance = np.tanh(model_performance)
            else: #just average encoding model performances
                model_performance = np.nanmean(all_data_performance,axis=0)
            self.brain_shape = model_performance.shape
            self.affine = affine

            img = nibabel.Nifti1Image(model_performance, affine)
            nibabel.save(img, self.out_dir+'/performance/'+self.sid+self.file_label+'_measure-'+label+'.nii.gz')
        
        if('added_variance' in measures):
            label = 'perf_raw'
            all_added_variance = []
            subject_added_variance = []
            for feature_name in self.feature_names:
                all_data_added_variance = []
                for subject in self.subjects[self.task]:
                    try:
                        base_model,subtract_models = helpers.get_added_variance_models(feature_name)
                        
                        all_subtract_models = []
                        for subtract_model in subtract_models:
                            enc_file_label = '_encoding_model-'+subtract_model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                            if(self.mask_name!=None):
                                enc_file_label = enc_file_label + '_mask-'+self.mask_name
                            filepath = self.in_dir+'/performance/'+subject+enc_file_label+'_measure-'+label+'.nii.gz'
                            nii1 = nibabel.load(filepath)
                            nii1 = nibabel.processing.smooth_image(nii1,fwhm=self.smoothing_fwhm,mode='nearest')
                            all_subtract_models.append(nii1.get_fdata())
                        
                        subtract_model = np.max(all_subtract_models,axis=0)
                        subtract_model[subtract_model<0] = 0
                        
                        enc_file_label = '_encoding_model-'+base_model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                        if(self.mask_name!=None):
                            enc_file_label = enc_file_label + '_mask-'+self.mask_name
                        filepath =self.in_dir+'/performance/'+subject+enc_file_label+'_measure-'+label+'.nii.gz'
                        nii1 = nibabel.load(filepath)
                        nii1 = nibabel.processing.smooth_image(nii1,fwhm=self.smoothing_fwhm,mode='nearest')

                        base_model = nii1.get_fdata()
                        base_model[base_model<0] = 0
                        
                        performance = base_model-subtract_model
                        affine=nii1.affine
                        
                        all_data_added_variance.append(performance)
                        # print(subject)
                    except Exception as e:
                        # print(e)
                        pass
                all_data_added_variance = np.array(all_data_added_variance)
                avg_data_added_variance = np.nanmean(all_data_added_variance,axis=0)
                all_added_variance.append(avg_data_added_variance)
                subject_added_variance.append(all_data_added_variance)
            
            added_variance = np.array(all_added_variance)
            subject_added_variance = np.array(subject_added_variance)
            img = nibabel.Nifti1Image(added_variance, affine)
            nibabel.save(img, self.out_dir+'/added_variance/'+self.sid+self.file_label+'_measure-added_variance_raw.nii.gz' )

        if('unique_variance' in measures):
            label = 'perf_raw'
            all_unique_variance = []
            subject_unique_variance = []
            for feature_name in self.feature_names:
                all_data_unique_variance = []
                for subject in self.subjects[self.task]:
                    try:
                        base_model,subtract_model = helpers.get_unique_variance_models(feature_name)
                        enc_file_label = '_encoding_model-'+subtract_model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                        if(self.mask_name!=None):
                            enc_file_label = enc_file_label + '_mask-'+self.mask_name
                        filepath = self.in_dir+'/performance/'+subject+enc_file_label+'_measure-'+label+'.nii.gz'
                        nii1 = nibabel.load(filepath)
                        nii1 = nibabel.processing.smooth_image(nii1,fwhm=self.smoothing_fwhm,mode='nearest')
                        
                        subtract_model = nii1.get_fdata()
                        subtract_model[subtract_model<0] = 0
                        
                        enc_file_label = '_encoding_model-'+base_model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                        if(self.mask_name!=None):
                            enc_file_label = enc_file_label + '_mask-'+self.mask_name
                        filepath =self.in_dir+'/performance/'+subject+enc_file_label+'_measure-'+label+'.nii.gz'
                        nii1 = nibabel.load(filepath)
                        nii1 = nibabel.processing.smooth_image(nii1,fwhm=self.smoothing_fwhm,mode='nearest')

                        base_model = nii1.get_fdata()
                        base_model[base_model<0] = 0
                        
                        performance = base_model-subtract_model
                        affine=nii1.affine
                        
                        all_data_unique_variance.append(performance)
                        # print(subject)
                    except Exception as e:
                        print(e)
                        pass
                all_data_unique_variance = np.array(all_data_unique_variance)
                avg_data_unique_variance = np.nanmean(all_data_unique_variance,axis=0)
                all_unique_variance.append(avg_data_unique_variance)
                subject_unique_variance.append(all_data_unique_variance)
        
            unique_variance = np.array(all_unique_variance)
            subject_unique_variance = np.array(subject_unique_variance)
            img = nibabel.Nifti1Image(unique_variance, affine)
            nibabel.save(img, self.out_dir+'/unique_variance/'+self.sid+self.file_label+'_measure-unique_variance_raw.nii.gz' )

        if self.do_stats:
            label = 'perf_null_distribution'
            all_data_performance = []
            for subject in self.subjects[self.task]:
                try:
                    #load the raw performance data for each subject
                    nii = nibabel.load(self.in_dir+'/performance/'+subject+self.file_label+'_measure-'+label+'.nii.gz')
                    performance = nii.get_fdata()
                    affine = nii.affine
                    all_data_performance.append(performance)
                except Exception as e:
                    print(e)
                    pass

            all_data_performance = np.array(all_data_performance)
            model_performance_null_distribution = np.nanmean(all_data_performance,axis=0)

            img = nibabel.Nifti1Image(model_performance_null_distribution, affine)
            nibabel.save(img, self.out_dir+'/performance/'+self.sid+self.file_label+'_measure-'+label+'.nii.gz')
        if('ind_product_measure' in measures):
            all_ind_product_measure= []
            subject_ind_product_measure = []
            for feature_name in self.feature_names:
                # print(feature_name)
                all_data_ind_product_measure = []
                for subject in self.subjects[self.task]:
                    try:
                        nii = nibabel.load(self.in_dir+'/ind_product_measure/'+subject+self.file_label+'_measure-ind_product_measure_raw.nii.gz')
                        data = nii.get_fdata()
                        data[data<0] = 0 #clip response values to 0
                        #clip voxels with performance less than 0 to 0
                        perf_nii = nibabel.load(self.in_dir+'/performance/'+subject+self.file_label+'_measure-'+label+'.nii.gz')
                        perf_data = perf_nii.get_fdata()
                        data[:,perf_data<0] = 0
                        if self.scale_by=='total_variance':
                            data = data/data.sum(axis=0,keepdims=1) #normalize
                        if feature_name in self.combined_features:
                            for (ind,sub_feature_name) in enumerate(self.models_dict[feature_name]):
                                feature_ind = self.get_feature_index(subject,sub_feature_name)

                                sub_data = data[feature_ind]
                                if ind==0:
                                    overall = sub_data
                                else:
                                    overall = overall+sub_data
                            data = overall
                        else:
                            feature_index = self.get_feature_index(subject,feature_name)
                            data = data[feature_index]
                        # print(feature_index)
                        nii = nibabel.Nifti1Image(data,nii.affine)
                        #smooth after adding together layers if nec
                        nii = nibabel.processing.smooth_image(nii,fwhm=self.smoothing_fwhm,mode='nearest')
                        all_data_ind_product_measure.append(nii.get_fdata())
                    except Exception as e:
                        print(e)
                        pass
                all_data_ind_product_measure = np.array(all_data_ind_product_measure)
                avg_ind_product_measure = np.nanmean(all_data_ind_product_measure,axis=0)
                all_ind_product_measure.append(avg_ind_product_measure)
                subject_ind_product_measure.append(all_data_ind_product_measure)
            ind_product_measure = np.array(all_ind_product_measure) #mask it for computing the preference maps, which need a flat array
            subject_ind_product_measure = np.array(subject_ind_product_measure)
            img = nibabel.Nifti1Image(ind_product_measure, affine)
            nibabel.save(img, self.out_dir+'/ind_product_measure/'+self.sid+self.file_label+'_measure-ind_product_measure_raw.nii.gz' )

        if('spatial_correlation' in measures):
            label = 'voxelwise_correlation'
            all_spcorr = []
            all_ind_spcorr = []
            #load the features that are in the spatial correlation file
            filename = self.dir + '/analysis/SecondLevelIndividual/spatial_correlation/individual_maps/'+self.file_label+'_measure-voxelwise_correlation_alexnet-.csv'
            comparison_df = pd.read_csv(filename)
            # Convert the 'comparison' column into a list
            comparison_list = comparison_df['comparison'].tolist()
            for ind,feature in enumerate(comparison_list):
                all_data_spcorr = []
                for subject in self.subjects[self.task]:
                    try:
                        nii = nibabel.load(self.dir+'/analysis/SecondLevelIndividual/spatial_correlation/individual_maps/'+subject+self.file_label+'_measure-'+label+'_alexnet-'+'.nii.gz')
                        nii = nibabel.processing.smooth_image(nii,fwhm=self.smoothing_fwhm,mode='nearest')
                        spcorr = nii.get_fdata()[ind]
                        all_data_spcorr.append(spcorr)
                        # print(subject)
                    except Exception as e:
                        print(e)
                        pass
                    
                ind_spcorr = np.array(all_data_spcorr)
                avg_spcorr = np.nanmean(ind_spcorr,axis=0) 
                all_spcorr.append(avg_spcorr)
                all_ind_spcorr.append(ind_spcorr)
            
            spcorr = np.array(all_spcorr)
            ind_spcorr = np.array(all_ind_spcorr)
            

            img = nibabel.Nifti1Image(spcorr, affine)
            nibabel.save(img, self.out_dir+'/spatial_correlation/'+self.sid+self.file_label+'_measure-'+label+'_alexnet-.nii.gz')
        # all_ind_feat_perf= []
        # for feature_name in self.feature_names:
        #     # print(feature_name)
        #     all_data_ind_feat_perf = []
        #     for subject in self.subjects[self.task]:
        #         try:
        #             nii = nibabel.load(self.in_dir+'/ind_feature_performance/'+subject+self.file_label+'_measure-ind_perf_raw.nii.gz')
        #             data = nii.get_fdata()
        #             data[data<0] = 0 #clip to zero
        #             if feature_name in self.combined_features:
        #                 for (ind,sub_feature_name) in enumerate(self.models_dict[feature_name]):
        #                     feature_ind = self.get_feature_index(subject,sub_feature_name)

        #                     sub_data = data[feature_ind]
        #                     if ind==0:
        #                         overall = sub_data
        #                     else:
        #                         overall = overall+sub_data
        #                 data = overall
        #             else:
        #                 feature_index = self.get_feature_index(subject,feature_name)
        #                 data = data[feature_index]
        #             nii = nibabel.Nifti1Image(data,nii.affine)
        #             #smooth after adding together layers if nec
        #             nii = nibabel.processing.smooth_image(nii,fwhm=self.smoothing_fwhm,mode='nearest')
        #             all_data_ind_feat_perf.append(nii.get_fdata())
        #         except Exception as e:
        #             print(e)
        #             pass
        #     all_data_ind_feat_perf = np.array(all_data_ind_feat_perf)
        #     avg_ind_feat_perf = np.nanmean(all_data_ind_feat_perf,axis=0)
        #     all_ind_feat_perf.append(avg_ind_feat_perf)
        # ind_feature_performance = np.array(all_ind_feat_perf) #mask it for computing the preference maps, which need a flat array
        # img = nibabel.Nifti1Image(ind_feature_performance, affine)
        # nibabel.save(img, self.out_dir+'/ind_feature_performance/'+self.sid+self.file_label+'_measure-ind_perf_raw.nii.gz' )

        #MASK all data
        self.mask = helpers.load_mask(self,self.mask_name)
        if('performance' in measures):
            self.model_performance = model_performance[self.mask.get_fdata()==1]
            self.all_model_performance = all_data_performance[:,self.mask.get_fdata()==1]
        # if ind_feature_performance.shape[0]>1:
        #     self.ind_feature_performance = ind_feature_performance[:,self.mask==1]
        if ('ind_product_measure' in measures):
            self.ind_product_measure = ind_product_measure[:,self.mask.get_fdata()==1]
            self.all_ind_product_measure =subject_ind_product_measure[:,:,self.mask.get_fdata()==1]
        if ('added_variance' in measures):
            self.added_variance = added_variance[:,self.mask.get_fdata()==1]
            self.all_added_variance =subject_added_variance[:,:,self.mask.get_fdata()==1]
        if ('unique_variance' in measures):
            self.unique_variance = unique_variance[:,self.mask.get_fdata()==1]
            self.all_unique_variance =subject_unique_variance[:,:,self.mask.get_fdata()==1]
        if ( 'spatial_correlation' in measures):
            # print(spcorr.shape)
            # print(ind_spcorr.shape)
            self.spcorr = spcorr[:,self.mask.get_fdata()==1]
            self.all_spcorr = ind_spcorr[:,:,self.mask.get_fdata()==1]

        #null distribution already masked
        if self.do_stats:
            self.model_performance_null_distribution = model_performance_null_distribution
    def plot_difference(self,label,threshold=0.000001,vmin=None,vmax=None,group='',cmap='coolwarm'):
        filepath = self.out_dir+'/ind_product_measure/'+self.sid+self.file_label
        if label=='raw':
            img = nibabel.load(filepath+'_measure-ind_product_measure_raw.nii.gz')
            img_data = img.get_fdata()
            affine = img.affine

        # for (ind,feature_name) in enumerate(self.feature_names):
        # print(feature_name)

        ind_perf_1 = img_data[0] #get first feature

        ind_perf_2 = img_data[1] #get second feature

        ind_perf = ind_perf_1-ind_perf_2
        ind_perf = nibabel.Nifti1Image(ind_perf,affine)

        title = ''
        plotting_helpers.plot_surface(ind_perf,self.figure_dir + "/ind_product_measure/" + self.sid+self.file_label+'_diff-'+self.feature_names[0]+'-'+self.feature_names[1]+'_measure-ind_product_measure_'+label,threshold=threshold,vmin=vmin,vmax=vmax,title=title,symmetric_cbar=True,cmap=cmap,colorbar_label='difference (proportion of total $R^2$)')
    def plot_model_performance_difference(self,model1,model2,threshold=0.000001,vmin=None,vmax=None,group='',cmap='coolwarm'):

        file_label = '_encoding_model-'+ model1 + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)+'_mask-'+self.mask_name
        filepath = self.out_dir+'/performance/'+self.sid+file_label+'_measure-perf_raw.nii.gz'
        img = nibabel.load(filepath)
        ind_perf_1 = img.get_fdata() #first model
        affine = img.affine

        file_label = '_encoding_model-'+model2 + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)+'_mask-'+self.mask_name
        filepath = self.out_dir+'/performance/'+self.sid+file_label+'_measure-perf_raw.nii.gz'
        img = nibabel.load(filepath)
        ind_perf_2 = img.get_fdata() #second model
        affine = img.affine

        ind_perf = ind_perf_1-ind_perf_2
        ind_perf = nibabel.Nifti1Image(ind_perf,affine)

        title = ''
        plotting_helpers.plot_surface(ind_perf,self.figure_dir + "/performance/" + self.sid+self.file_label+'_diff-'+model1+'-'+model2+'_measure-performance',threshold=threshold,vmin=vmin,vmax=vmax,title=title,symmetric_cbar=True,cmap=cmap,colorbar_label='difference ($R^2$)')
    def plot_ind_feature_performance(self,label,threshold=0.000001,vmin=0,vmax=1,cmap='yellow_hot'):
        filepath = self.out_dir+'/ind_feature_performance/'+self.sid+self.file_label

        if label=='raw':
            img = nibabel.load(filepath+'_measure-ind_perf_raw.nii.gz')
            img_data = img.get_fdata()
            affine = img.affine
        for (ind,feature_name) in enumerate(self.feature_names):
            print(feature_name)

            ind_perf = img_data[ind]
            ind_perf = nibabel.Nifti1Image(ind_perf,affine)

            filepath = self.figure_dir + "/ind_feature_performance/" +self.sid+self.file_label+'_feature-'+feature_name+'_measure-ind_perf_'+label+'.png'
            title=''
            cmap_ = self.cmaps[cmap]
            plotting_helpers.plot_surface(ind_perf,self.figure_dir + "/ind_feature_performance/" + self.sid+self.file_label+'_measure-ind_feature_performance_'+label+'_feature-'+feature_name,threshold=threshold,vmin=vmin,vmax=vmax,title=title,symmetric_cbar=False,cmap=cmap_,colorbar_label='Feature performance $R^2$')
    
    def plot_unique_variance(self,label,threshold=0.000001,vmin=None,vmax=None,group='',cmap='yellow_hot'):
        filepath = self.out_dir+'/unique_variance/'+self.sid+self.file_label

        img = nibabel.load(filepath+'_measure-unique_variance_raw.nii.gz')
        img_data = img.get_fdata()
        affine = img.affine
        if(label=='stats'):
            stats_img = nibabel.load(self.out_dir+'/unique_variance/'+self.sid+self.file_label+'_measure-unique_variance_p_fdr.nii.gz')
            stats_mask = stats_img.get_fdata()
            print('getting rid of non-sig values')
            print(stats_mask.shape)
            print(img_data.shape)
            print(np.sum(stats_mask==0))
            print(np.sum(stats_mask[:,self.mask==1]==0))
            img_data[stats_mask==0]=0 #zero out any voxels that failed to reject hypothesis

        
        for (ind,feature_name) in enumerate(self.feature_names):
            print(feature_name)
            ind_perf = img_data[ind]
            ind_perf = nibabel.Nifti1Image(ind_perf,affine)

            title=''#feature_name
            cmap_ = cmap
            plotting_helpers.plot_surface(ind_perf,self.figure_dir + "/unique_variance/" + self.sid+self.file_label+'_measure-unique_variance_'+label+'_feature-'+feature_name,threshold=threshold,vmin=vmin,vmax=vmax,title=title,symmetric_cbar=False,cmap=cmap_,colorbar_label='unique variance $R^2$')
    
    
    def plot_added_variance(self,label,threshold=0.000001,vmin=None,vmax=None,group='',cmap='yellow_hot'):
        filepath = self.out_dir+'/added_variance/'+self.sid+self.file_label

        img = nibabel.load(filepath+'_measure-added_variance_raw.nii.gz')
        img_data = img.get_fdata()
        affine = img.affine
        if(label=='stats'):
            stats_img = nibabel.load(self.out_dir+'/added_variance/'+self.sid+self.file_label+'_measure-added_variance_p_fdr.nii.gz')
            stats_mask = stats_img.get_fdata()
            print('getting rid of non-sig values')
            print(stats_mask.shape)
            print(img_data.shape)
            print(np.sum(stats_mask==0))
            print(np.sum(stats_mask[:,self.mask==1]==0))
            img_data[stats_mask==0]=0 #zero out any voxels that failed to reject hypothesis

        
        for (ind,feature_name) in enumerate(self.feature_names):
            print(feature_name)
            ind_perf = img_data[ind]
            ind_perf = nibabel.Nifti1Image(ind_perf,affine)

            title=''#feature_name
            cmap_ = cmap
            plotting_helpers.plot_surface(ind_perf,self.figure_dir + "/added_variance/" + self.sid+self.file_label+'_measure-added_variance_'+label+'_feature-'+feature_name,threshold=threshold,vmin=vmin,vmax=vmax,title=title,symmetric_cbar=False,cmap=cmap_,colorbar_label='Added explained variance $R^2$')
    
    def plot_ind_product_measure(self,label,threshold=0.000001,vmin=None,vmax=None,group='',cmap='yellow_hot',title=''):
        filepath = self.out_dir+'/ind_product_measure/'+self.sid+self.file_label

        img = nibabel.load(filepath+'_measure-ind_product_measure_raw.nii.gz')
        img_data = img.get_fdata()
        affine = img.affine
        if(label=='stats'):
            stats_img = nibabel.load(self.out_dir+'/ind_product_measure/'+self.sid+self.file_label+'_measure-product_measure_p_fdr.nii.gz')
            stats_mask = stats_img.get_fdata()
            # print('getting rid of non-sig values')
            # print(stats_mask.shape)
            # print(img_data.shape)
            # print(np.sum(stats_mask==0))
            # print(np.sum(stats_mask[:,self.mask==1]==0))
            img_data[stats_mask==0]=0 #zero out any voxels that failed to reject hypothesis

        if group=='':
            for (ind,feature_name) in enumerate(self.feature_names):
                # print(feature_name)
                ind_perf = img_data[ind]
                ind_perf = nibabel.Nifti1Image(ind_perf,affine)

                filepath = self.figure_dir + "/ind_product_measure/" +self.sid+self.file_label+'_measure-ind_product_measure_'+label+'_feature-'+feature_name+'_vmax-'+str(vmax)
                # title=''#feature_name
                title = ''
                if(feature_name=='alexnet'):
                    title = 'Image Model (AlexNet)'
                elif(feature_name=='sbert'):
                    title = 'Sentence Model (sBERT)'
                cmap_ = cmap
                plotting_helpers.plot_surface(ind_perf,filepath,threshold=threshold,vmin=vmin,vmax=vmax,title=title,symmetric_cbar=False,cmap=cmap_,colorbar_label='Explained variance $R^2$')
        else:
            overall = []
            for (ind,feature_name) in enumerate(group_dict[group]):
                # print(feature_name)
                feature_ind = self.get_feature_index('sub-06',feature_name)

                ind_perf = img_data[feature_ind]
                if ind==0:
                    overall = ind_perf
                else:
                    overall = overall+ind_perf
            ind_perf = nibabel.Nifti1Image(overall,affine)
            filepath = self.figure_dir + "/ind_product_measure/" +self.sid+self.file_label+'_group-'+group+'_measure-ind_product_measure_'+label+'.png'
            title=group+' features'
            vmax = group_vmax[group]
            cmap_ = cmap
            plotting_helpers.plot_surface(ind_perf,self.figure_dir + "/ind_product_measure/" + self.sid+self.file_label+'_measure-ind_product_measure_'+label+'_group-'+group,threshold=threshold,vmin=vmin,vmax=vmax,title=title,symmetric_cbar=True,cmap=cmap_,colorbar_label='proportion of total explained variance $R^2$')
    def plot_spatial_correlation(self,label,threshold=0.000001,vmin=None,vmax=None,group='',cmap='yellow_hot'):
        filepath = self.out_dir+'/spatial_correlation/'+self.sid+self.file_label
        
        img = nibabel.load(filepath+'_measure-voxelwise_correlation_alexnet-.nii.gz')
        img_data = img.get_fdata()
        # print(img_data.shape)
        affine = img.affine
        if(label=='stats'):
            # stats_img = nibabel.load(self.out_dir+'/ind_product_measure/'+self.sid+self.file_label+'_measure-product_measure_p_fdr.nii.gz')
            # stats_mask = stats_img.get_fdata()
            # print('getting rid of non-sig values')
            # print(stats_mask.shape)
            # print(img_data.shape)
            # print(np.sum(stats_mask==0))
            # print(np.sum(stats_mask[:,self.mask==1]==0))
            # img_data[stats_mask==0]=0 #zero out any voxels that failed to reject hypothesis
            pass
        filename = self.dir + '/analysis/SecondLevelIndividual/spatial_correlation/individual_maps/'+self.file_label+'_measure-voxelwise_correlation_alexnet-.csv'
        comparison_df = pd.read_csv(filename)
        # Convert the 'comparison' column into a list
        comparison_list = comparison_df['comparison'].tolist()
        for (ind,feature_name) in enumerate(comparison_list):
            # print(feature_name)
            ind_perf = img_data[ind]
            # print(ind_perf.shape)
            ind_perf = nibabel.Nifti1Image(ind_perf,affine)
            # print(ind_perf.shape)

            filepath = self.figure_dir + "/spatial_correlation/" +self.sid+self.file_label+'_measure-spcorr_'+label+'_alexnet-'+feature_name
            title=''#feature_name
            cmap_ = cmap
            plotting_helpers.plot_surface(ind_perf,filepath,threshold=threshold,vmin=vmin,vmax=vmax,title=title,symmetric_cbar=False,cmap=cmap_,colorbar_label='r')

    def plot_performance(self, label, threshold=None,vmin=None,vmax=None,symmetric_cbar=True,cmap='yellow_hot',title=''):
        file_label = self.sid+self.file_label#self.sid+'_encoding_model-'+self.model + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
        
        # if self.mask_name is not None:
        #     file_label = file_label + '_mask-'+self.mask_name

        filepath = self.out_dir+'/performance/'+file_label
        
        img = nibabel.load(filepath+'_measure-perf_raw.nii.gz')
        # title = self.sid
        if(label=='stats'):
            stats_img = nibabel.load(filepath+'_measure-perf_p_fdr.nii.gz')
            stats_mask = stats_img.get_fdata()
            
            data = img.get_fdata()
            data[stats_mask==0] = 0 #zero out any voxels that failed to reject hypothesis
            # print(len(data[data>0]))
            # print(len(data[data==0]))
            #remake img
            img = nibabel.Nifti1Image(data,affine=img.affine)
            
        # if label=='stats':
        #     img = nibabel.load(filepath+'_measure-perf_p_fdr.nii.gz')
        #     title = self.sid+', pvalue<'+str(threshold)
        #     #add a small number to each value so that zeroes are plotted!
        #     performance_p = img.get_fdata()
        #     threshold = 1-threshold
        #     # #mask the brain with significant pvalues
        #     # performance_p[performance_p>0.05]=-1
        #     performance_p[self.mask==1] = 1-performance_p[self.mask==1] #turn all significant voxels into high values for plotting
        #     affine = img.affine
        #     img = nibabel.Nifti1Image(performance_p, affine)
        #     # cmap = 'Greys'
        # vmin=None
        # if(vmax is None):
        #     vmax = np.max(img.get_fdata()-0.03)
        cmap = cmap
        # title=''
        colorbar_label = 'Explained Variance $R^2$'
        if(self.model=='correlation'):
            colorbar_label = 'Explained Variance $r^2$'
        plotting_helpers.plot_surface(img,self.figure_dir + "/performance/" + file_label+'_measure-perf_'+label,threshold=threshold,vmin=vmin,vmax=vmax,cmap=cmap,title=title,symmetric_cbar=symmetric_cbar,colorbar_label=colorbar_label)

    def permutation_brainiak(self,load=False,iterations=10000):
        import datetime
        
        if not load:
            print('starting permutation testing')
            now = datetime.datetime.now()
            print(now)
            observed,p,distribution = permutation_isc(self.all_model_performance, group_assignment=None, pairwise=False,
                        summary_statistic='mean', n_permutations=iterations,
                        side='right', random_state=None)
            print('ended permutation testing')
            now = datetime.datetime.now()
            print(now)
            self.p_unc = p
            data = self.unmask_reshape(self.p_unc)
            img = nibabel.Nifti1Image(data,self.affine)
            nibabel.save(img, self.out_dir+'/performance/'+self.sid+self.file_label+'_measure-perf_p_unc.nii.gz')
        else:
            print('loading precomputed uncorrected p map')
            nii = nibabel.load(self.out_dir+'/performance/'+self.sid+self.file_label+'_measure-perf_p_unc.nii.gz')
            self.p_unc = nii.get_fdata()
        
        self.fdr_reject,self.p_fdr = fdrcorrection(self.p_unc, alpha=0.001, method='p', is_sorted=False)
        self.fdr_reject[self.model_performance<0.001] = 0 #value cutoff
        data = self.unmask_reshape(self.fdr_reject)
        img = nibabel.Nifti1Image(data,self.affine)
        nibabel.save(img, self.out_dir+'/performance/'+self.sid+self.file_label+'_measure-perf_p_fdr.nii.gz')
        
    def stats_wilcoxon(self):
        print('performing signed permutation test...')
        #performance stats
        # print(self.all_model_performance.shape)
        def process(voxel_performance):
            x = voxel_performance
            x = np.around(x, decimals=4) #needs to ensure that theoretically identically values are not numerically distinct
                                                                        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html#r996422d5c98f-4
            result = wilcoxon(x,alternative='greater',zero_method='zsplit',nan_policy='omit')
            pvalue = result.pvalue
            return pvalue

        self.perf_p_unc = np.array(Parallel(n_jobs=-1)(delayed(process)(voxel_performance) for voxel_performance in self.all_model_performance.T))
        self.perf_fdr_reject,self.perf_p_fdr = fdrcorrection(self.perf_p_unc, alpha=0.001, method='p', is_sorted=False) #method is BH
        # self.perf_fdr_reject,self.perf_p_fdr,a,b = statsmodels.stats.multitest.multipletests(self.perf_p_unc, alpha=0.05, method='bonferroni')

        # print(np.sum(self.perf_fdr_reject),len(self.perf_fdr_reject))
        data = self.unmask_reshape(self.perf_fdr_reject)
        img = nibabel.Nifti1Image(data,self.affine)
        nibabel.save(img, self.out_dir+'/performance/'+self.sid+self.file_label+'_measure-perf_p_fdr.nii.gz')
        
        #product measure stats
        if(len(self.feature_names)>0):
            ind_features = self.feature_names
            ind_feature_perf_p_unc = []
            ind_feature_perf_p_fdr = []
            ind_feature_perf_fdr_reject = []

            for (ind,feature) in enumerate(ind_features):
                performance = self.all_ind_product_measure[ind]
                # print(self.all_ind_product_measure.shape)
                # print('performance',performance.shape)
                def process(voxel_performance):
                    x = voxel_performance
                    # print(x)
                    # print(x.shape)
                    # print(np.nanmean(x))
                    x = np.around(x, decimals=4) #needs to ensure that theoretically identically values are not numerically distinct
                                                                                #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html#r996422d5c98f-4
                    result = wilcoxon(x,alternative='greater',nan_policy='omit')
                    pvalue = result.pvalue
                    return pvalue
                perf_p_unc = np.array(Parallel(n_jobs=-1)(delayed(process)(voxel_performance) for voxel_performance in performance.T))
                perf_fdr_reject,perf_p_fdr = fdrcorrection(perf_p_unc, alpha=0.001, method='p', is_sorted=False)
                # perf_fdr_reject,perf_p_fdr,a,b = statsmodels.stats.multitest.multipletests(perf_p_unc, alpha=0.001, method='bonferroni')

                ind_feature_perf_p_unc.append(perf_p_unc)
                ind_feature_perf_p_fdr.append(perf_p_fdr)
                ind_feature_perf_fdr_reject.append(perf_fdr_reject)

            self.ind_feature_perf_p_unc = np.array(ind_feature_perf_p_unc)
            self.ind_feature_perf_p_fdr = np.array(ind_feature_perf_p_fdr)
            self.ind_feature_perf_fdr_reject = np.array(ind_feature_perf_p_fdr)


            # print(self.ind_feature_perf_fdr_reject.shape)
            img = nibabel.Nifti1Image(self.unmask_reshape(self.ind_feature_perf_fdr_reject),self.affine)
            nibabel.save(img, self.out_dir+'/ind_product_measure/'+self.sid+self.file_label+'_measure-product_measure_p_fdr.nii.gz') 

    def stats(self):

        #performance stats
        def process(voxel_performance,voxel_null_distribution):
            #one-tailed t test for performance
            null_n = voxel_null_distribution.shape[0]
            null_n_over_sample = sum((voxel_null_distribution>voxel_performance).astype(int))
            p = null_n_over_sample/null_n
            self.iterations = null_n
            return p

        self.perf_p_unc = np.array(Parallel(n_jobs=-1)(delayed(process)(voxel_performance,voxel_null_distribution) for (voxel_performance,voxel_null_distribution) in zip(self.model_performance, self.model_performance_null_distribution.T)))
        self.perf_fdr_reject,self.perf_p_fdr = fdrcorrection(self.perf_p_unc, alpha=0.001, method='n', is_sorted=False) #method is BY for dependence between variables
        file_label = '_encoding_model-'+self.model + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)+'_mask-'+self.mask_name


        img = nibabel.Nifti1Image(self.unmask_reshape(self.perf_p_fdr),self.affine)
        nibabel.save(img, self.out_dir+'/performance/'+self.sid+file_label+'_measure-perf_p_fdr.nii.gz')

        #ind feature performance stats

        if len(self.ind_feature_performance_null_distribution)>0:
            ind_features = self.feature_names
            ind_feature_perf_p_unc = []
            ind_feature_perf_p_fdr = []
            ind_feature_perf_fdr_reject = []

            results = self.ind_feature_performance_null_distribution
            feature_null_distributions = np.reshape(results,(results.shape[1],results.shape[2],results.shape[0])) #reshape to #features,#voxels,#iterations

            for (ind,feature) in enumerate(ind_features):
                voxelwise_null_distribution = feature_null_distributions[ind]
                performance = self.ind_feature_performance[ind]
                def process(voxel_performance,voxel_null_distribution):
                    null_n = voxel_null_distribution.shape[0]
                    null_n_over_sample = sum((voxel_null_distribution>voxel_performance).astype(int))
                    p = null_n_over_sample/null_n
                    return p
                perf_p_unc = np.array(Parallel(n_jobs=-1)(delayed(process)(voxel_performance,voxel_null_distribution) for (voxel_performance,voxel_null_distribution) in zip(performance, voxelwise_null_distribution)))
                perf_fdr_reject,perf_p_fdr = fdrcorrection(perf_p_unc, alpha=0.001, method='n', is_sorted=False)

                ind_feature_perf_p_unc.append(perf_p_unc)
                ind_feature_perf_p_fdr.append(perf_p_fdr)
                ind_feature_perf_fdr_reject.append(perf_fdr_reject)

            self.ind_feature_perf_p_unc = np.array(ind_feature_perf_p_unc)
            self.ind_feature_perf_p_fdr = np.array(ind_feature_perf_p_fdr)
            self.ind_feature_perf_fdr_reject = np.array(ind_feature_perf_p_fdr)

            file_label = '_encoding_model-'+self.model + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)+'_mask-'+self.mask_name


            img = nibabel.Nifti1Image(self.unmask_reshape(self.ind_feature_perf_p_fdr),self.affine)
            nibabel.save(img, self.out_dir+'/ind_feature_performance/'+self.sid+file_label+'_measure-ind_perf_p_fdr.nii.gz') 

    def run(self):
    	pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task','-task',type=str,default='sherlock')
    parser.add_argument('--mask','-mask',type=str, default='ISC')
    parser.add_argument('--model','-model',type=str,default='full')
    parser.add_argument('--smoothing-fwhm','-smoothing-fwhm',type=float,default=3.0)
    parser.add_argument('--chunklen','-chunklen',type=int,default=30)
    parser.add_argument('--feature-of-interest','-feature-of-interest',type=str,default='None')
    parser.add_argument('--population','-population',type=str,default='NT')
    parser.add_argument('--stats','-stats',type=bool,default=False)


    parser.add_argument('--dir', '-dir', type=str,
                        default='/Users/hsmall2/Documents/GitHub/deep_nat_lat')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/hsmall2/Documents/GitHub/deep_nat_lat/analysis')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/hsmall2/Documents/GitHub/deep_nat_lat/figures')
    args = parser.parse_args()
    SecondLevelGroup(args).run()

if __name__ == '__main__':
    main()