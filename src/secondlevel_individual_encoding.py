import os
import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.stats
import statsmodels.stats.multitest
import nibabel
import nilearn

from nilearn import surface
from nilearn.glm import threshold_stats_img
import h5py

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import colors

from joblib import Parallel, delayed
import pickle
from tqdm.autonotebook import tqdm

from matplotlib.colors import LinearSegmentedColormap

import warnings

from src import encoding
from src import helpers
from src import plotting_helpers
from src import stats_helpers

plt.rcParams.update({'font.size': 16,'font.family': 'Arial'})

warnings.filterwarnings("ignore", message="`legacy_format` will default to `False` in release 0.11. Dataset fetchers will then return pandas dataframes by default instead of recarrays.")
warnings.filterwarnings("ignore", message="Mean of empty slice.")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

class SecondLevelIndividual(encoding.EncodingModel):

    def __init__(self, args, measure_suffix=''):
        self.process = 'SecondLevelIndividual'
        self.dir = args.dir
        self.measure_suffix = measure_suffix  # For speech_only_eval support
        self.data_dir = os.path.join(args.dir,'data')
        self.enc_dir = os.path.join(args.out_dir,'EncodingModel')
        self.glm_dir = os.path.join(args.out_dir, 'GLM')
        self.out_dir = os.path.join(args.out_dir,self.process)
        self.population = args.population
        self.sid = 'sub-'+self.population
        self.subjects = []
        self.glm_task = ['SIpointlights','language']
        self.enc_task = 'sherlock'
        self.mask_name = args.mask
        self.space = args.space
        self.model = args.model
        self.models = [self.model]
        self.smoothing_fwhm = args.smoothing_fwhm
        self.chunklen = args.chunklen
        self.fMRI_data = []
        self.included_data_features = []
        self.vsm = VoxelSelectionManager(self)
        self.mm = MaskManager(self)

        self.brain_shape = (97,115,97)
        self.affine = []
        self.feature_names = []
        self.run_groups = {'SIpointlights':[('12','3'),('23','1'),('13','2')],'language':[('1','2'),('2','1')]} #localizer, response
        self.all_runs = {'SIpointlights':'123','language':'12'}
        self.localizer_contrasts = {'SIpointlights':{'interact&no_interact','interact-no_interact'},'language':{'intact-degraded'}}
        self.MT = ['MT']
        self.ISC = ['ISC']
        self.STS = ['pSTS','aSTS']
        self.language =['aTemp','pTemp','frontal']
        self.language_ROI_names = ['aTemp','pTemp','frontal']
        self.localizer_masks = {'interact&no_interact':self.MT,'interact-no_interact':self.STS,'intact-degraded':self.language,
                                'motion pointlights':self.MT,'SI pointlights':self.STS, 'language':self.language, 'DMN': self.STS,
                                'motion':self.MT,'social interaction':self.STS,'num_agents':self.STS, 'alexnet':self.ISC,
                                'social':self.STS,'valence':self.STS,'face':self.STS,'mentalization':self.ISC, 'arousal':self.ISC,
                                'SLIP':self.ISC,'SimCLR':self.ISC,'CLIP':self.language, 'GPT2':self.language,
                                'SLIP_attention':self.ISC,'SimCLR_attention':self.ISC,'SLIP_embedding':self.ISC,'SimCLR_embedding':self.ISC,
                                'glove':self.language,'sbert':self.ISC,'word2vec':self.language, 'hubert':self.language,
                                'speaking':self.ISC,'indoor_outdoor':self.ISC,'pitch':self.ISC,'amplitude':self.ISC,
                                'turn_taking':self.ISC,'written_text':self.ISC,'music':self.ISC,'pixel':self.ISC,'hue':self.ISC,'none':self.ISC}
        self.perc_top_voxels = str(args.perc_top_voxels)
        self.n_voxels_all = {
                '1': {'MT':34,'STS':85,'temporal_language':85,'frontal':32,'pSTS':60,'aSTS':25,'pTemp':60,'aTemp':25},
                '2.5':{'MT':86,'STS':212,'temporal_language':212,'frontal':79,'pSTS':150,'aSTS':62,'pTemp':150,'aTemp':62},
                '5': {'MT':172,'STS':425,'temporal_language':425,'frontal':158,'pSTS':300,'aSTS':125,'pTemp':300,'aTemp':125},
                '7.5': {'MT':259,'STS':637,'temporal_language':637,'frontal':236,'pSTS':450,'aSTS':187,'pTemp':450,'aTemp':187},
                '10': {'MT':345,'STS':850,'temporal_language':850,'frontal':315,'pSTS':600,'aSTS':250,'pTemp':600,'aTemp':250},
                '12.5':{'MT':432,'STS':1062,'temporal_language':1062,'frontal':394,'pSTS':750,'aSTS':312,'pTemp':750,'aTemp':312},
                '15':{'MT':518,'STS':1275,'temporal_language':1275,'frontal':473,'pSTS':900,'aSTS':375,'pTemp':900,'aTemp':375},
                '20':{'MT':690,'STS':1700,'temporal_language':1700,'frontal':630,'pSTS':1200,'aSTS':500,'pTemp':1200,'aTemp':500}
        }
        self.response_contrasts = {'SIpointlights':{'interact','no_interact'},'language':{'intact','degraded'}}
        self.cached_masks = {}
        self.cached_subject_masks = {}
        self.save_weights = False
        self.scale_by = None
        self.group_encoding_weights = []
        self.group_encoding_performance = []
        self.subj_encoding_localizer_masks = []
        self.subj_glm_localizer_masks = []
        self.glm_results = None
        self.performance_stats = []
        self.figure_dir = args.figure_dir + "/" + self.process
        Path(f'{self.out_dir}/{"weights"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"performance"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"localizer_masks"}').mkdir(exist_ok=True, parents=True)

        Path(f'{self.figure_dir}/{"localizer_masks"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"glm_zscores"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"localizer_overlap_maps"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"difference_maps"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"response_similarity"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"preference_map"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"map"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"overlap"}').mkdir(exist_ok=True, parents=True)

        self.enc_file_label = '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
        self.enc_file_label_new = '_smoothingfwhm-'+str(self.smoothing_fwhm)
        self.glm_file_label = '_smoothingfwhm-'+str(self.smoothing_fwhm)

        if(self.mask_name!=None):
            self.enc_file_label = self.enc_file_label + '_mask-'+self.mask_name
            self.enc_file_label_new = self.enc_file_label_new + '_mask-'+self.mask_name + '_chunklen-'+str(self.chunklen)
            # self.glm_file_label = self.glm_file_label + '_mask-' +self.mask_name
            self.mask = helpers.load_mask(self,self.mask_name)
        else:
            self.enc_file_label_new = self.enc_file_label_new + '_chunklen-'+str(self.chunklen)

        self.labels_dict = {
            # 'SimCLR':'SimCLR only',
            # 'SLIP':'SLIP only',
            'GPT2_1sent':'GPT2',
            'interact-no_interact':'social interaction',
            'interact&no_interact':'motion',
            'intact-degraded':'language',
            'interact':'interacting pointlights',
            'no_interact':'non-interacting pointlights',
            'intact':'intact speech',
            'degraded':'degraded speech',
            'num_agents':'number of agents',
            'turn_taking':'turn taking',
            'written_text':'written text',
            'sbert+word2vec':'sbert+word2vec',
            'L':'left',
            'R':'right',
            'frontal_language':'frontal',
            'temporal_language':'temporal',
            'post_temporal_language':'pTemp',
            'ant_temporal_language':'aTemp',
            'cross_subject':'cross-subject',
            'alexnet_layer7':'AlexNet layer 7',
            'alexnet_layer6':'AlexNet layer 6',
            'alexnet_layer5':'AlexNet layer 5',
            'alexnet_layer4':'AlexNet layer 4',
            'alexnet_layer3':'AlexNet layer 3',
            'alexnet_layer2':'AlexNet layer 2',
            'alexnet_layer1':'AlexNet layer 1',
            'sbert_layer12':'sBERT layer 12',
            'sbert_layer11':'sBERT layer 11',
            'sbert_layer10':'sBERT layer 10',
            'sbert_layer9':'sBERT layer 9',
            'sbert_layer8':'sBERT layer 8',
            'sbert_layer7':'sBERT layer 7',
            'sbert_layer6':'sBERT layer 6',
            'sbert_layer5':'sBERT layer 5',
            'sbert_layer4':'sBERT layer 4',
            'sbert_layer3':'sBERT layer 3',
            'sbert_layer2':'sBERT layer 2',
            'sbert_layer1':'sBERT layer 1',
            'vision':'AlexNet+motion',
            'vision_simclr':'SimCLR+motion',
            'language':'HuBERT+word2vec+GPT2',
            'vision_transformers':'SimCLR+motion',
            'language_transformers':'HuBERT+word2vec+GPT2',
            'alexnet': 'Image Model',
            'alexnet_eps0.3': 'Image Model',
            'motion':'Motion Model',
            'hubert':'Speech Model',
            'hubert_chunk120':'Speech Model',
            'word2vec':'Word Model',
            'sbert':'Sentence Model',
            'GPT2_chunk120':'Sentence Model',
            'SimCLR_attention_eps0.3': 'SimCLR attention',
            'SimCLR_embedding_eps0.3': 'SimCLR embedding',
            'SimCLR_eps0.3': 'SimCLR'
            }
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

        self.model_features_dict = helpers.get_models_dict()
        for layer in self.model_features_dict['sbert']:
            self.features_dict[layer]='downsampled_all-mpnet-base-v2_'+layer.split('_')[1]
        self.combined_features = helpers.get_combined_features()

        self.feature_names = self.model_features_dict[self.model]
        self.plot_features_dict = self.model_features_dict ## Default is to plot all features
        self.plot_features = self.plot_features_dict[self.model]

        self.cmaps = plotting_helpers.get_cmaps()
        self.colors_dict = plotting_helpers.get_colors_dict()
        self.subjects = helpers.get_subjects(self.population)

        # Clipping control parameters
        self.clip_negative_performance = False
        self.clip_negative_product_measure = False
        self.mask_by_performance = True

    def get_feature_index(self, feature, weight=False, selection_model=''):
        # Simplified feature file naming: encoding_model-[model_name].csv
        if(selection_model==''):
            file_label = 'encoding_model-'+self.model
        else:
            file_label = 'encoding_model-'+selection_model

        if weight:
            filename = os.path.join(self.enc_dir,'features',file_label+'_weight_features.csv')
            file = open(filename, "r")
            data = list(csv.reader(file, delimiter=','))[0]
            file.close()
            #compute all indices for all features given their feature space sizes
            indices = {}
            current_index = 0
            n_features_dict = {'motion':2530,
                               'alexnet_layer1': 6480,
                               'alexnet_layer2': 6480,
                               'alexnet_layer3': 6480,
                               'alexnet_layer4': 6480,
                               'alexnet_layer5': 6480,
                               'alexnet_layer6': 4096,
                               'alexnet_layer7': 4096,
                               'word2vec':300,
                               'sbert_layer10':768,
                               'sbert_layer11':768,
                               'sbert_layer12':768}
            for feature_name in data:
                indices[feature_name] = (current_index,current_index+n_features_dict[feature_name])
                current_index = current_index + n_features_dict[feature_name]
            return np.arange(indices[feature][0],indices[feature][1],1)
        else:
            filename = os.path.join(self.enc_dir,'features',file_label+'_features.csv')
            file = open(filename, "r")
            data = list(csv.reader(file, delimiter=','))[0]
            file.close()
            return data.index(feature)   
    def get_preference_map(self,data,measure='ind_product_measure'):
        if(measure=='ind_product_measure'):
            temp_transposed = data.T
            # print(temp_transposed)
            nan_col = np.nanmax(temp_transposed, axis=1)
            temp = np.zeros(data.shape[1])
            temp[~np.isnan(nan_col)] = np.nanargmax(temp_transposed[~np.isnan(nan_col),:], axis=1)
        elif(measure=='structured_variance'):
            #get the min performance needed be same as full model (90%)
            full_model_data = data[-1]*0.9
             #restrict to positive full model R
            data[:,full_model_data<=0]=np.nan
            feature_data = (data>=full_model_data)*1.0 #binary whether or not each feature reaches 95% of the full model
            temp_transposed = feature_data.T
            nan_col = np.nanmax(temp_transposed, axis=1)
            temp = np.zeros(data.shape[1])
            temp[~np.isnan(nan_col)] = np.nanargmax(temp_transposed[~np.isnan(nan_col),:], axis=1) #take the first index that reached 95% (forward direction)
            
        return temp.astype(int)  
    def generate_preference_maps(self,load=False,measure='ind_product_measure',restricted=False,threshold=0,features=[],color_dict=None,file_tag='',views=['lateral','ventral'],roi_thickness=0,width_mm=None,fontsize=7):
        fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')
        # Try new ordering first (mask before chunklen)
        enc_file_label_new = '_encoding_model-'+self.model + '_smoothingfwhm-'+str(self.smoothing_fwhm)
        if(self.mask_name!=None):
            enc_file_label_new = enc_file_label_new + '_mask-'+self.mask_name + '_chunklen-'+str(self.chunklen)
        else:
            enc_file_label_new = enc_file_label_new + '_chunklen-'+str(self.chunklen)

        # Old ordering (chunklen before mask) as fallback
        enc_file_label_old = '_encoding_model-'+self.model + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
        if(self.mask_name!=None):
            enc_file_label_old = enc_file_label_old + '_mask-'+self.mask_name

        for subject_ind,subject in enumerate(self.subjects['sherlock']):
            #load the feature product measure data
            if(measure=='ind_product_measure'):
                # Try both orderings for product measure file
                enc_response_path_new = os.path.join(self.enc_dir,'ind_product_measure',subject+enc_file_label_new+'_measure-ind_product_measure_raw.nii.gz')
                enc_response_path_old = os.path.join(self.enc_dir,'ind_product_measure',subject+enc_file_label_old+'_measure-ind_product_measure_raw.nii.gz')

                if os.path.exists(enc_response_path_new):
                    enc_response_img = nibabel.load(enc_response_path_new)
                elif os.path.exists(enc_response_path_old):
                    enc_response_img = nibabel.load(enc_response_path_old)
                else:
                    raise FileNotFoundError(f"Could not find product measure file with either ordering:\n  New: {enc_response_path_new}\n  Old: {enc_response_path_old}")

                enc_response_data = enc_response_img.get_fdata()

                # Try both orderings for performance file
                enc_performance_path_new = os.path.join(self.enc_dir,'performance',subject+enc_file_label_new+'_measure-perf_raw.nii.gz')
                enc_performance_path_old = os.path.join(self.enc_dir,'performance',subject+enc_file_label_old+'_measure-perf_raw.nii.gz')

                if os.path.exists(enc_performance_path_new):
                    enc_performance_img = nibabel.load(enc_performance_path_new)
                elif os.path.exists(enc_performance_path_old):
                    enc_performance_img = nibabel.load(enc_performance_path_old)
                else:
                    raise FileNotFoundError(f"Could not find performance file with either ordering:\n  New: {enc_performance_path_new}\n  Old: {enc_performance_path_old}")

                enc_performance_data = enc_performance_img.get_fdata()
            elif(measure=='structured_variance'):
                data = [] #first entry is the full model (assumes that self.model is the full model)
                for feature in features:
                    # Try new ordering first (mask before chunklen)
                    enc_file_label_new = '_encoding_model-'+feature + '_smoothingfwhm-'+str(self.smoothing_fwhm)
                    if(self.mask_name!=None):
                        enc_file_label_new = enc_file_label_new + '_mask-'+self.mask_name + '_chunklen-'+str(self.chunklen)
                    else:
                        enc_file_label_new = enc_file_label_new + '_chunklen-'+str(self.chunklen)

                    # Old ordering (chunklen before mask) as fallback
                    enc_file_label_old = '_encoding_model-'+feature + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                    if(self.mask_name!=None):
                        enc_file_label_old = enc_file_label_old + '_mask-'+self.mask_name

                    enc_response_path_new = os.path.join(self.enc_dir,'performance',subject+enc_file_label_new+'_measure-perf_raw.nii.gz')
                    enc_response_path_old = os.path.join(self.enc_dir,'performance',subject+enc_file_label_old+'_measure-perf_raw.nii.gz')

                    if os.path.exists(enc_response_path_new):
                        enc_response_img = nibabel.load(enc_response_path_new)
                    elif os.path.exists(enc_response_path_old):
                        enc_response_img = nibabel.load(enc_response_path_old)
                    else:
                        raise FileNotFoundError(f"Could not find performance file with either ordering:\n  New: {enc_response_path_new}\n  Old: {enc_response_path_old}")

                    subtract_model_data = enc_response_img.get_fdata()
                    data.append(subtract_model_data)
                
                temp = np.array(data)
            
            hemis= ['left','right']
            preference1_map_surf = []
            preference2_map_surf = []
            for hemi in hemis:
                transform_mesh = fsaverage['pial_'+hemi]
                inner_mesh = fsaverage['white_'+hemi]

                if(measure=='ind_product_measure'):
                    temp = enc_response_data.copy()
                    temp[temp<0] = 0 #clip response values to 0
                    # temp[:,enc_performance_data<0] = 0 #clip any voxels with neg performance to 0
                    final_temp = []
                    for ind,feature in enumerate(features):
                        if(feature in self.combined_features):
                            for (ind,sub_feature_name) in enumerate(self.model_features_dict[feature]):
                                feature_ind = self.get_feature_index(sub_feature_name)
                                sub_data = temp[feature_ind]
                                if(ind==0):
                                    overall = sub_data
                                else:
                                    overall = overall+sub_data
                            data = overall
                        else:
                            feature_index = self.get_feature_index(feature)
                            data = temp[feature_index]
                        final_temp.append(data)
                    temp = np.array(final_temp)
                #add a dimension to beginning that is just 0's (so it will never be the max )
                placeholder = np.reshape(np.zeros(temp[0].shape),(1,temp[0].shape[0],temp[0].shape[1],temp[0].shape[2]))
                temp_added_placeholder = np.concatenate((placeholder,temp))
                nii = nibabel.Nifti1Image(np.transpose(temp_added_placeholder, (1, 2, 3, 0)),enc_response_img.affine)

                n_points_to_sample = 50
                temp_ind_product_measure = np.transpose(surface.vol_to_surf(nii, transform_mesh,inner_mesh=inner_mesh,depth = np.linspace(0, 1, n_points_to_sample),interpolation='nearest'),(1,0))
                if restricted:
                    temp_ind_product_measure[temp_ind_product_measure<threshold] = np.nan
                
                performance = surface.vol_to_surf(enc_performance_img, transform_mesh,inner_mesh=inner_mesh,depth = np.linspace(0, 1, n_points_to_sample),interpolation='nearest')
                temp_ind_product_measure[:,(performance<0)]=np.nan #nan out values that do not have positive performance

                preference1_map = self.get_preference_map(temp_ind_product_measure,measure=measure)
                preference1_map_surf.append(preference1_map)

                one_hot_encoded_preference_map = np.eye(temp_ind_product_measure.shape[0])[preference1_map.copy()].T.astype(bool)
                temp_ind_product_measure[one_hot_encoded_preference_map] = 0

                preference2_map = self.get_preference_map(temp_ind_product_measure,measure=measure)
                preference2_map_surf.append(preference2_map)

            #plot on surface
            # performance = surface.vol_to_surf(enc_performance_img, transform_mesh,inner_mesh=inner_mesh,depth = np.linspace(0, 1, n_points_to_sample),interpolation='nearest')
            # print(performance.shape)
            # preference1_map[performance<0] = 0 #make sure to make any values that have negative performance have no feature
            features_list = ['blank'] + features
            surf_cmap = colors.ListedColormap([self.colors_dict[feature_name] for feature_name in features_list ])
            file_label = subject+'_encoding_model-'+self.model + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
            title = 'Participant '+ str(subject_ind+1)
            ROI_niis = []
            glm_file_label_ = '_smoothingfwhm-'+str(self.smoothing_fwhm)
            ROIs = ['social interaction','language']
            ROI_colors = ['white','black']
            for ind,localizer_contrast in enumerate(['interact-no_interact','intact-degraded']):
                file_label_ = subject+glm_file_label_+'_mask-'+'_'.join(self.localizer_masks[localizer_contrast])#mask
                ROI_file = self.out_dir + '/localizer_masks/'+file_label_+'_glm_loc-'+localizer_contrast+'_run-all_binary.nii.gz'
                try:
                    ROI_niis.append(nibabel.load(ROI_file))
                except Exception as e:
                    print(e)
            plotting_helpers.plot_preference_surf(preference1_map_surf,os.path.join(self.figure_dir,"preference_map",file_label+'_measure-'+measure+'_preference1_map_'+file_tag),ROI_niis=ROI_niis,ROIs=ROIs,ROI_colors=ROI_colors,color_dict=color_dict,cmap=surf_cmap,threshold=0.001,title=title,vmax = len(features),views=views,roi_thickness=roi_thickness,width_mm=width_mm,fontsize=fontsize)   
    def glm_voxel_selection(self, load=False, plot_ind=True, plot=True, plot_stacked=True,
                         response_label='ind_feature_performance', localizers_to_plot=[],
                         localizer_label_dict={}, plot_noise_ceiling=False, stats_to_do=None,
                         parametric=True, filepath_tag='', extraction_threshold=0,
                         average_posterior_anterior=False, average_left_right=False,
                         figure_tag='', restrict_legend=False,
                         plot_lines=False, label_region=True,
                         legend_below=True, include_cross_subject=False,
                         clip_negative_to_zero=True, save_source_data=False, overlay_points=False,
                         exclude_outliers=False, outlier_iqr_multiplier=3):
        """
        For GLM, select voxels and collect responses using VoxelSelectionManager.
        exclude_outliers: drop a participant's encoding response in a region if it is beyond
        outlier_iqr_multiplier * IQR from the quartiles of that region (per hemisphere, model, and feature),
        i.e. Tukey's far out fence by default. The excluded values are saved to a csv in out_dir.
        """
        print('Running GLM voxel selection...')

        if not load:
            results = self.vsm.collect_glm_voxel_selection(
                response_label=response_label,
                filepath_tag=filepath_tag,
                extraction_threshold=extraction_threshold,
                include_cross_subject=include_cross_subject
            )
            
        base_filename = f"{self.sid}{self.enc_file_label}_model-{self.model}_perc_top_voxels-{self.perc_top_voxels}_glm_localizer_{response_label}_{filepath_tag}"
        summary_path = os.path.join(self.out_dir, base_filename + '_summary.csv')
        results = pd.read_csv(summary_path)
        # Drop rows where 'num_voxels' is missing (NaN)
        results = results.dropna(subset=['num_voxels'])
        # Optional: reset index
        results = results.reset_index(drop=True)
        pkl_path = os.path.join(self.out_dir, base_filename +'_voxelwise.pkl')


        # a row is cross-validated when its localizer and response come from separate runs (several run groups averaged)
        results['cross_validated'] = [len(runs.split(',')) > 1 for runs in results['averaged_localizer_runs'].astype(str)]
        # all-runs localizer with a response from the same task: the glm response is not independent of the localizer
        same_task_all_runs = (results['glm_task_localizer'] == results['glm_task_response']) & ~results['cross_validated']

        # --- Plotting ---
        if plot:
            localizer_contrasts = [loc + ' (' + localizer_label_dict[loc] + ')' for loc in localizers_to_plot]

            width_ratios = [1 if (average_posterior_anterior) else len(self.localizer_masks[loc]) for loc in localizers_to_plot]
            if ('language' in localizers_to_plot):
                if('frontal' in self.localizer_masks['language']):
                    width_ratios[2] = 2
                    if not average_posterior_anterior:
                        width_ratios[2]=3
            
            self.plot_glm_response(results[~same_task_all_runs], average_posterior_anterior, column_group='localizer_contrast_label',
                                col_order=localizer_contrasts, label_dict=localizer_label_dict, width_ratios=width_ratios,
                                loc_name='localizer_contrast', model_label='model', selection_model='',
                                file_label='glm_localizer_glm_response', filepath_tag=filepath_tag,label_region=label_region,
                                legend_below=legend_below, save_source_data=save_source_data,
                                overlay_points=overlay_points)

            # encoding responses come from the all-runs localizer (the same voxels as Figures 4 and 5)
            results = results[~results['cross_validated']].copy()
            # drop the empty placeholder rows (localizer task without the other task's response, no runs), so the
            # de-duplication of the all-runs rows never keeps a placeholder over the participant's real row
            results = results[results['averaged_localizer_runs'].notna()]

            if exclude_outliers:
                results = self.exclude_encoding_outliers(results, base_filename, outlier_iqr_multiplier)

            # Only plot individual encoding responses for feature-specific measures
            #
            if response_label != 'performance':
                hue = 'enc_feature_name'
            if response_label == 'performance':
                if self.clip_negative_performance:
                    results['encoding_response'][results['encoding_response'] < 0] = 0
                hue = 'model'
                
            self.plot_enc_response(results, average_posterior_anterior, average_left_right, column_group='localizer_contrast_label',
                                        col_order=localizer_contrasts, label_dict=localizer_label_dict, width_ratios=width_ratios,
                                        response_label=response_label, hue=hue, loc_name='localizer_contrast',
                                        model_label='model', file_label='glm_localizer_enc_response-'+response_label, plot_stacked=plot_stacked,
                                        plot_noise_ceiling=plot_noise_ceiling, stats_to_do=stats_to_do,
                                        parametric=parametric, filepath_tag=filepath_tag, figure_tag=figure_tag,
                                        restrict_legend=restrict_legend, label_region=label_region,
                                        save_source_data=save_source_data, overlay_points=overlay_points)

        return summary_path,pkl_path       
    def exclude_encoding_outliers(self, results, base_filename, iqr_multiplier=3):
        """Remove encoding responses beyond iqr_multiplier * IQR from the quartiles, computed across participants
        within each localizer contrast, region, hemisphere, model, and feature. Saves the removed rows to a csv."""
        group_cols = ['localizer_contrast', 'mask', 'hemisphere', 'model', 'enc_feature_name']
        # model performance has no feature name; fill the missing group labels so every row is grouped
        grouped = results['encoding_response'].groupby([results[col].fillna('') for col in group_cols])
        q1 = grouped.transform(lambda x: x.quantile(0.25))
        q3 = grouped.transform(lambda x: x.quantile(0.75))
        iqr = q3 - q1
        is_outlier = (results['encoding_response'] < q1 - iqr_multiplier * iqr) | (results['encoding_response'] > q3 + iqr_multiplier * iqr)
        outliers = results[is_outlier]
        outlier_path = os.path.join(self.out_dir, base_filename + f'_outliers_iqr-{iqr_multiplier}.csv')
        outliers[['subject'] + group_cols + ['encoding_response', 'num_voxels']].drop_duplicates().to_csv(outlier_path, index=False)
        summary = outliers.drop_duplicates(subset=['subject', 'mask', 'hemisphere', 'model'])
        print(f'excluded {len(summary)} outlier participant-region values ({iqr_multiplier} x IQR):')
        for _, row in summary.iterrows():
            print(f"   {row['subject']} {row['hemisphere']} {row['mask']} ({row['model']}): {row['encoding_response']:.2f}")
        print('outliers saved:', outlier_path)
        return results[~is_outlier].copy()

    def save_data(self, data, filename, save_type='csv'):
        """
        Save data to a file.

        Args:
            data: DataFrame, list, dict, or numpy array
            filename: full path without extension (e.g., '/path/to/file')
            save_type: 'csv', 'pkl', or 'npz'
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if save_type == 'csv':
            if not isinstance(data, pd.DataFrame):
                raise ValueError("For 'csv' save_type, data must be a pandas DataFrame.")
            save_path = filename + '.csv'
            data.to_csv(save_path, index=False)
            print(f"Saved CSV to {save_path}")

        elif save_type == 'pkl':
            save_path = filename + '.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved PKL to {save_path}")

        elif save_type == 'npz':
            save_path = filename + '.npz'
            if isinstance(data, dict):
                np.savez(save_path, **data)
            elif isinstance(data, np.ndarray):
                np.savez(save_path, data=data)
            else:
                raise ValueError("For 'npz' save_type, data must be a dict or numpy array.")
            print(f"Saved NPZ to {save_path}")

        else:
            raise ValueError(f"Unsupported save_type: {save_type}")

        return save_path 
    def prepare_glm_dataframe(self, results, average_posterior_anterior, loc_name, label_dict):
        # Filter for relevant ROIs and subjects
        regions = self.MT + self.STS + self.language
        results = results[results['mask'].isin(regions)]
        results = results[results['subject'].isin(self.subjects['sherlock'])]

        # Average posterior and anterior if requested
        if average_posterior_anterior:
            for prev_mask in self.STS:
                results['mask'].replace(prev_mask, 'STS', inplace=True)
            for prev_mask in ['pTemp','aTemp']:
                results['mask'].replace(prev_mask, 'temporal', inplace=True)

            average_these = ['glm_weight', 'proportion_voxels', 'encoding_response', 'num_voxels']
            if 'cross-subject encoding' in results.columns:
                average_these.append('cross-subject encoding')
            columns = [col for col in results.columns if col not in average_these and not results[col].isna().all()]
            results = pd.pivot_table(data=results, values=average_these, index=columns, aggfunc='mean').reset_index()

        # Replace labels
        results.replace(self.labels_dict, inplace=True)
        results['localizer_contrast_label'] = [f"{val} ({label_dict[val]})" for val in results[loc_name]]
        results['hemi_mask'] = [f"{h} {m}" for h, m in zip(results['hemisphere'], results['mask'])]
        results['Condition'] = results['glm_response_contrast']

        # Compute unilaterality of voxel distribution
        group_columns = ['subject', loc_name, 'mask']
        df_pivot = results.pivot_table(index=group_columns, columns='hemisphere',
                                    values='proportion_voxels', aggfunc='mean').reset_index()
        df_pivot['unilaterality of glm localizer'] = (df_pivot['right'] - df_pivot['left']).abs() / (df_pivot['right'] + df_pivot['left'])
        results = pd.merge(results, df_pivot[group_columns + ['unilaterality of glm localizer']], on=group_columns, how='left')

        # Sort for consistent plotting order
        hemi_mask_ID_dict = {
            'left MT': 0, 'right MT': 1, 'left STS': 2, 'right STS': 3,
            'left pSTS': 2, 'left aSTS': 3, 'right pSTS': 4, 'right aSTS': 5,
            'left language': 10, 'right language': 11,
            'left pTemp': 6, 'left aTemp': 7, 'left temporal language': 7, 'left temporal': 7, 
            'right pTemp': 8, 'right aTemp': 9, 'right temporal language': 9, 'right temporal': 9,
            'left frontal language': 10, 'left frontal': 10, 'right frontal language': 11,  'right frontal': 11
        }
        results['hemi_mask_ID'] = [hemi_mask_ID_dict[x] for x in results['hemi_mask']]
        results = results.sort_values(by='hemi_mask_ID', ascending=True)

        # Drop duplicate entries per subject/mask/condition
        results = results.drop_duplicates(subset=['subject', 'hemi_mask', 'Condition', 'localizer_contrast_label'])

        return results   
    def prepare_encoding_dataframe(self, results, average_posterior_anterior, average_left_right, loc_name, label_dict, subject_group='SIpointlights',hue='',model_label=''):
        # Filter ROIs and subjects
        all_regions = self.MT + self.STS + self.language
        results = results[results['mask'].isin(all_regions)]
        results = results[results['subject'].isin(self.subjects[subject_group])]

        # Average posterior/anterior if requested
        if average_posterior_anterior:
            for prev_mask in self.STS:
                results['mask'].replace(prev_mask, 'STS', inplace=True)
            for prev_mask in ['aTemp','pTemp']:
                results['mask'].replace(prev_mask, 'temporal', inplace=True)

            # Build list of columns to average, only including those that exist
            # num_voxels differs between the posterior and anterior regions, so it is averaged too (not an index column)
            average_these = ['glm_weight', 'encoding_response', 'proportion_voxels', 'num_voxels']
            if 'cross-subject encoding' in results.columns:
                average_these.append('cross-subject encoding')

            columns = [col for col in results.columns if col not in average_these]

            # Replace all-NaN index columns with 0 to avoid pivot issues
            for col in columns:
                if results[col].isna().all():
                    results[col] = 0

            results = pd.pivot_table(data=results, values=average_these, index=columns, aggfunc='mean').reset_index()

        if average_left_right:
            results['hemisphere'] = 'both' #replace all hemisphere info with both
            # Build list of columns to average, only including those that exist
            average_these = ['glm_weight', 'encoding_response', 'proportion_voxels', 'num_voxels']
            if 'cross-subject encoding' in results.columns:
                average_these.append('cross-subject encoding')

            columns = [col for col in results.columns if col not in average_these]

            # Replace all-NaN index columns with 0 to avoid pivot issues
            for col in columns:
                if results[col].isna().all():
                    results[col] = 0

            results = pd.pivot_table(data=results, values=average_these, index=columns, aggfunc='mean').reset_index()

        # Replace labels
        results = results.replace(self.labels_dict)
        results['localizer_contrast_label'] = [f"{val} ({label_dict[val]})" for val in results[loc_name]]
        results['hemi_mask'] = [f"{h} {m}" for h, m in zip(results['hemisphere'], results['mask'])]
        results['hemi_localizer_contrast_mask'] = [
            f"{h} {val} {m}" for val, h, m in zip(results[loc_name], results['hemisphere'], results['mask'])
        ]

        # Add consistent plot labels
        results['Feature Space'] = results[hue]
        results['Feature_Space'] = results[hue]  # sometimes used redundantly
        results['Model'] = results[model_label]

        # Sort ROIs for plotting
        hemi_mask_ID_dict = {
            'left MT': 0, 'right MT': 1, 'left STS': 2, 'right STS': 3,
            'left pSTS': 2, 'left aSTS': 3, 'right pSTS': 4, 'right aSTS': 5,
            'left language': 10, 'right language': 11,
            'left pTemp': 6, 'left aTemp': 7, 'left temporal language': 7, 'left temporal': 7, 
            'right pTemp': 8, 'right aTemp': 9, 'right temporal language': 9, 'right temporal': 9,
            'left frontal language': 10, 'left frontal': 10, 'right frontal language': 11,  'right frontal': 11,
            'both MT': 0, 'both STS':1, 'both temporal': 2, 'both frontal':5,
            'both pSTS':1,'both aSTS': 2,'both pTemp':3, 'both aTemp':4,
        }
        results['hemi_mask_ID']= [hemi_mask_ID_dict[x] for x in results['hemi_mask']]
        results = results.sort_values(by='hemi_mask_ID', ascending=True)

        return results
    def plot_glm_response(
        self,
        results,
        average_posterior_anterior,
        column_group,
        col_order,
        width_ratios,
        label_dict,
        loc_name,
        model_label,
        selection_model,
        file_label,
        filepath_tag,
        parametric=True,
        label_region=True,
        legend_below=True,
        save_source_data=False,
        overlay_points=False
    ):
        # 1. Prepare Data
        df = self.prepare_glm_dataframe(
            results=results,
            average_posterior_anterior=average_posterior_anterior,
            loc_name=loc_name,
            label_dict=label_dict
        )

        # Save the plotted per-subject data (posterior/anterior already averaged within subject)
        if save_source_data:
            source_df = df.sort_values(by=['hemi_mask_ID', 'subject'])[
                ['subject', 'localizer_contrast_label', 'hemisphere', 'mask', 'Condition', 'glm_weight', 'num_voxels']
            ].rename(columns={'localizer_contrast_label': 'localizer', 'mask': 'region',
                              'Condition': 'condition', 'glm_weight': 'beta_weight'})
            avg_tag = '_avg_post_ant' if average_posterior_anterior else ''
            source_path = os.path.join(
                self.figure_dir,
                f"{self.sid}{self.enc_file_label}_model-{selection_model}_{file_label}_perc_top_voxels-{self.perc_top_voxels}_glm_response{avg_tag}_source_data.csv"
            )
            source_df.to_csv(source_path, index=False)
            print('glm response source data saved:', source_path)

        # 2. Set plot params
        params = {
            'x': 'hemi_mask',
            'y': 'glm_weight',
            'hue': 'Condition',
            'hue_order': ['interacting pointlights', 'non-interacting pointlights', 'intact speech', 'degraded speech'],
            # interacting pointlights dark purple, non-interacting light purple
            'palette': {'interacting pointlights':'#54278F','non-interacting pointlights':'#BCBDDC', 'intact speech':self.colors_dict['GPT2_chunk120'],'degraded speech':self.colors_dict['hubert']}#'plasma'
        }

        # 3. Bar + Point plots
        bar_fig, point_fig = plotting_helpers.plot_bar_and_strip(
            data=df,
            column_group=column_group,
            col_order=col_order,
            width_ratios=width_ratios,
            params=params,
            sharey=False,
            legend_below=legend_below,
            height=4.6,
            aspect=1.2,
            overlay_points=overlay_points
        )

        # 4. Define regions
        MT = self.MT
        ISC = self.ISC
        STS = ['STS'] if average_posterior_anterior else self.STS
        language = ['temporal','frontal'] if average_posterior_anterior else self.language_ROI_names

        # bars with the individual points drawn on are saved under their own name
        region_sets = {'point': point_fig, 'bar_points' if overlay_points else 'bar': bar_fig}

        for label, fig in region_sets.items():
            for ax_row in fig.axes:
                for ax in ax_row:
                    region_type = ax.get_title().split('(')[1].split(')')[0]
                    localizer_label = ax.get_title().split(' = ')[1]
                    regions = (
                        MT if region_type == 'MT'
                        else ISC if region_type == 'ISC'
                        else language if region_type == 'language'
                        else STS
                    )
                    masks = [f"{hemi} {roi}" for roi in regions for hemi in ['left', 'right']]

                    # Filter data for the panel
                    panel_df = df[df[column_group] == localizer_label]
                    independent_var = 'Condition'

                    # 5. Stats + annotate
                    # Define all possible comparison pairs
                    all_comparison_pairs = [
                        ('interacting pointlights', 'non-interacting pointlights'),
                        ('intact speech', 'degraded speech')
                    ]

                    # Filter to only pairs where both conditions exist in the data
                    available_conditions = set(panel_df['Condition'].unique())
                    # print(f"DEBUG: Available conditions in panel_df: {available_conditions}")
                    comparison_pairs = [
                        pair for pair in all_comparison_pairs
                        if pair[0] in available_conditions and pair[1] in available_conditions
                    ]
                    # print(f"DEBUG: Filtered comparison_pairs: {comparison_pairs}")
                    # print(f"DEBUG: masks: {masks}")

                    pairs, pvals = stats_helpers.compute_pairwise_stats(
                        data=panel_df,
                        masks=masks,
                        condition_var='Condition',
                        value_var='glm_weight',
                        parametric=parametric,
                        comparison_pairs=comparison_pairs
                    )
                    # print(f"DEBUG: Returned pairs: {len(pairs)}, pvals: {len(pvals)}")

                    # Only apply annotations if there are valid pairs
                    if len(pairs) > 0:
                        plotting_helpers.apply_stat_annotations(
                            ax=ax,
                            pairs=pairs,
                            pvals=pvals,
                            data=panel_df,
                            params=params
                        )
                    else:
                        print(f"DEBUG: Skipping stat annotations (no valid pairs for these masks)")

                    # 6. Cosmetic
                    ax.set_xticklabels([f"{t.get_text().split(' ')[1]}\n{t.get_text().split(' ')[0]}" for t in ax.get_xticklabels()])
                    ax.set_title(localizer_label)
                    # for side, xpos in zip(['left', 'right'], [0.25, 0.75]):
                    #     ax.text(xpos, -0.17, side, transform=ax.transAxes, ha='center', va='top',fontsize=20)
                    
                    if(label_region):
                        loc_key = ax.get_title().split('(')[0].strip()
                        region_label_dict = {
                            'motion': 'motion\nresponsive',
                            'social interaction': 'social\ninteraction\nselective',
                            'language': 'language\nselective',
                        }
                        contrast_label_dict = {
                            'motion': '(interacting +\nnot interacting)',
                            'social interaction': '(interacting -\nnot interacting)',
                            'language': '(intact - degraded)',
                        }
                        region_text = region_label_dict.get(loc_key, loc_key + '\nregions')
                        contrast_text = contrast_label_dict.get(loc_key, '')
                        ax.text(0.5, 1.20, region_text, fontsize=plotting_helpers._fs(14), fontweight='normal', color='black',
                                ha='center', va='bottom', transform=ax.transAxes)
                        ax.text(0.5, 1.18, contrast_text, fontsize=plotting_helpers._fs(14), fontweight='normal', color='black',
                                ha='center', va='top', transform=ax.transAxes)

            fig.set_axis_labels("", "beta weight")
            fig.set_titles("")
            if average_posterior_anterior:
                filepath = os.path.join(
                    self.figure_dir,
                    f"{self.sid}{self.enc_file_label}_model-{selection_model}_{file_label}_perc_top_voxels-{self.perc_top_voxels}_glm_response_{label}_avg_post_ant.pdf"
                )
            else:
                filepath = os.path.join(
                    self.figure_dir,
                    f"{self.sid}{self.enc_file_label}_model-{selection_model}_{file_label}_perc_top_voxels-{self.perc_top_voxels}_glm_response_{label}.pdf"
                )
            plotting_helpers.save_figure(fig.fig, filepath)
            plt.close(fig.fig)
            
        #stats comparing social interaction selectivity
        fit = stats_helpers.compare_selectivity_between_SI_and_language(
            data=df, #dataframe with all regions
            condition='interacting pointlights',
            subtract_condition='non-interacting pointlights',
            roi_column='hemi_mask',
            subject_column='subject',
            value_column='glm_weight'
        )
        csv_path = os.path.join(self.out_dir, f"{self.sid}{self.enc_file_label}_model-{self.model}_{file_label}_perc_top_voxels-{self.perc_top_voxels}_glm_response_{filepath_tag}_social_interaction_selectivity.csv")
        fit.to_csv(csv_path)
        
        #stats comparing language selectivity
        fit = stats_helpers.compare_selectivity_between_SI_and_language(
            data=df, #dataframe with all regions
            condition='intact speech',
            subtract_condition='degraded speech',
            roi_column='hemi_mask',
            subject_column='subject',
            value_column='glm_weight'
        )
        csv_path = os.path.join(self.out_dir, f"{self.sid}{self.enc_file_label}_model-{self.model}_{file_label}_perc_top_voxels-{self.perc_top_voxels}_glm_response_{filepath_tag}_language_selectivity.csv")
        fit.to_csv(csv_path)
        if (not average_posterior_anterior):
            #stats comparing selectivity in posterior and anterior
            fit = stats_helpers.compare_selectivity_within_region_type(
                data=df, #dataframe with all regions
                condition='interacting pointlights',
                subtract_condition='non-interacting pointlights',
                roi_column='hemi_mask',
                subject_column='subject',
                value_column='glm_weight'
                )
            csv_path = os.path.join(self.out_dir, f"{self.sid}{self.enc_file_label}_model-{self.model}_{file_label}_perc_top_voxels-{self.perc_top_voxels}_glm_response_{filepath_tag}_social_interaction_selectivity_in_p_Vs_a.csv")
            fit.to_csv(csv_path)
            
            #stats comparing selectivity in posterior and anterior
            fit = stats_helpers.compare_selectivity_within_region_type(
                data=df, #dataframe with all regions
                condition='intact speech',
                subtract_condition='degraded speech',
                roi_column='hemi_mask',
                subject_column='subject',
                value_column='glm_weight'
                )
            csv_path = os.path.join(self.out_dir, f"{self.sid}{self.enc_file_label}_model-{self.model}_{file_label}_perc_top_voxels-{self.perc_top_voxels}_glm_response_{filepath_tag}_language_selectivity_in_p_Vs_a.csv")
            fit.to_csv(csv_path)
            
            # Plot hemispheric distribution of selective voxels
            # plotting_helpers.plot_hemispheric_distribution(
            #     self=self,
            #     results=df,
            #     column_group=column_group,
            #     col_order=col_order,
            #     loc_name=loc_name,
            #     hue_key='Condition',
            #     value_key='proportion_voxels',
            #     file_tag=f"glm_{file_label}",
            #     restrict_to_feature='interacting pointlights'  # or None to show all
            # )
            
            # region_stats_df = stats_helpers.compare_hemispheric_distribution(
            #     data=df,
            #     region_column='hemi_mask',
            #     hemisphere_column='hemisphere',
            #     value_column='proportion_voxels',
            #     subject_column='subject'
            # )
            # csv_path = os.path.join(self.out_dir, f"{self.sid}{self.enc_file_label}_model-{self.model}_{file_label}_perc_top_voxels-{self.perc_top_voxels}_glm_response_{label}_{filepath_tag}_ROI_hemispheric_distribution.csv")
            # region_stats_df.to_csv(csv_path)
            
            # plotting_helpers.plot_hemispheric_distribution_correlations(
            #     self=self,
            #     results=df,
            #     file_label=file_label,
            #     # subset_regions=MT+STS+language
            #     )  
    def plot_enc_response(
        self,
        results,
        average_posterior_anterior,
        average_left_right,
        column_group,
        col_order,
        width_ratios,
        response_label,
        label_dict,
        hue,
        loc_name,
        model_label,
        file_label,
        filepath_tag,
        plot_stacked=True,
        plot_noise_ceiling=False,
        stats_to_do=None,
        parametric=True,
        figure_tag='',
        restrict_legend=False,
        label_region=True,
        save_source_data=False,
        overlay_points=False
    ):
        # 1. Prepare data
        df = self.prepare_encoding_dataframe(
            results=results,
            average_posterior_anterior=average_posterior_anterior,
            average_left_right=average_left_right,
            loc_name=loc_name,
            label_dict=label_dict,
            hue=hue,
            model_label=model_label
        )

        temp_df = df.drop_duplicates(
            subset=['subject', 'hemi_mask', 'Feature Space', 'Model', 'localizer_contrast_label']
        )

        # Save the plotted per-subject data (posterior/anterior already averaged within subject),
        # plus the across-subject mean/median per region and feature (the stacked bar segment heights)
        if save_source_data:
            source_df = temp_df.sort_values(by=['hemi_mask_ID', 'subject'])[
                ['subject', 'localizer_contrast_label', 'hemisphere', 'mask', 'Feature Space', 'encoding_response']
            ].rename(columns={'localizer_contrast_label': 'localizer', 'mask': 'region',
                              'Feature Space': 'feature', 'encoding_response': 'explained_variance'})
            source_stem = os.path.join(
                self.figure_dir,
                f"{self.sid}{self.enc_file_label}_model-{self.model}_{file_label}_perc_top_voxels-{self.perc_top_voxels}_{filepath_tag}_{figure_tag}"
            )
            source_df.to_csv(source_stem + '_source_data.csv', index=False)
            stacked_df = (
                source_df.groupby(['localizer', 'hemisphere', 'region', 'feature'], sort=False)['explained_variance']
                .agg(mean='mean', median='median', n_subjects='count')
                .reset_index()
            )
            stacked_df.to_csv(source_stem + '_stacked_source_data.csv', index=False)
            print('encoding response source data saved:', source_stem + '_source_data.csv', source_stem + '_stacked_source_data.csv')

        # 2. Plot setup
        independent_var = 'Model' if hue not in ['enc_feature_name'] else 'Feature Space'
        plot_features = self.models if hue not in ['enc_feature_name'] else self.plot_features
        params = {
            'x': 'hemi_mask',
            'y': 'encoding_response',
            'hue': independent_var,
            'hue_order': [self.labels_dict.get(f, f) for f in plot_features],
            'palette': self.colors_dict
        }

        bar_fig, point_fig, median_bar_fig = plotting_helpers.plot_bar_and_strip(
            data=temp_df,
            column_group=column_group,
            col_order=col_order,
            width_ratios=width_ratios,
            params=params,
            height=5,
            aspect=1,
            also_plot_median=True,
            overlay_points=overlay_points
        )

        region_sets = {'bar_points' if overlay_points else 'bar': bar_fig, 'point': point_fig, 'median_bar': median_bar_fig}
        MT = self.MT
        ISC = self.ISC
        STS = ['STS'] if average_posterior_anterior else self.STS
        language = ['temporal','frontal'] if average_posterior_anterior else self.language_ROI_names

        if average_left_right:
            hemis = ['both']
        else:
            hemis = ['left','right']
        pvalue_results = []
        stats_details = []  # each test's estimate, CI, SE, df and t, for the stats table (from the first figure only)

        for label, fig in region_sets.items():
            for ax_row in fig.axes:
                for ax in ax_row:
                    title_text = ax.get_title().split(' = ')[1]
                    region_type = title_text.split('(')[1].split(')')[0]
                    regions = (
                        MT if region_type == 'MT'
                        else ISC if region_type == 'ISC'
                        else language if region_type == 'language'
                        else STS
                    )
                    masks = [f"{hemi} {r}" for r in regions for hemi in hemis]
                    
                    panel_df = temp_df[temp_df[column_group] == title_text]
                    if(stats_to_do!=None):
                        stats_out = stats_helpers.run_statistical_comparisons(
                            data=panel_df,
                            masks=masks,
                            mode=stats_to_do,
                            condition_levels=params['hue_order'],
                            condition_var='enc_feature_name',
                            value_var='encoding_response',
                            correction='bonferroni',
                            parametric=parametric,
                            return_details=True
                        )
                        pairs, pvals = stats_out[0], stats_out[1]
                        if label == list(region_sets)[0] and len(stats_out) > 2:
                            stats_details.append(stats_out[2].assign(localizer=title_text))

                        plotting_helpers.apply_stat_annotations(ax, pairs, pvals, panel_df, params)
                        pvalue_results.append(dict(zip(pairs, pvals)))
                    # print(ax.get_xticklabels())
                    ax.set_xticklabels([plotting_helpers.region_tick_label(t.get_text()) for t in ax.get_xticklabels()])
                    # for hemi_text, xpos in zip(['left', 'right'], [0.25, 0.75]):
                    #     ax.text(xpos, -0.1, hemi_text, transform=ax.transAxes, ha='center', va='top')
                    
                    if(label_region):
                        loc_key = title_text.split('(')[0].strip()
                        region_label_dict = {
                            'motion': 'motion\nresponsive',
                            'social interaction': 'social\ninteraction\nselective',
                            'language': 'language\nselective',
                        }
                        contrast_label_dict = {
                            'motion': '(interacting +\nnot interacting)',
                            'social interaction': '(interacting -\nnot interacting)',
                            'language': '(intact - degraded)',
                        }
                        region_text = region_label_dict.get(loc_key, loc_key + '\nregions')
                        contrast_text = contrast_label_dict.get(loc_key, '')
                        ax.text(0.5, 1.20, region_text, fontsize=plotting_helpers._fs(14), fontweight='normal', color='black',
                                ha='center', va='bottom', transform=ax.transAxes)
                        ax.text(0.5, 1.18, contrast_text, fontsize=plotting_helpers._fs(14), fontweight='normal', color='black',
                                ha='center', va='top', transform=ax.transAxes)

            fig.fig.subplots_adjust(top=0.85)
            fig.set_axis_labels("", "explained variance $R^2$")
            fig.set_titles("")
            plotting_helpers.save_figure(
                fig.fig,
                os.path.join(
                    self.figure_dir,
                    f"{self.sid}{self.enc_file_label}_model-{self.model}_{file_label}_perc_top_voxels-{self.perc_top_voxels}_{label}_{filepath_tag}_{figure_tag}.pdf"
                )
            )
            plt.close(fig.fig)

        # full statistics of the tests annotated on the figure (estimate, CI, SE, t, df, corrected p)
        stats_details = [d for d in stats_details if not d.empty]
        if stats_details:
            stats_details = pd.concat(stats_details, ignore_index=True)
            stats_label = f"{self.model}_{file_label}_{stats_to_do}_{filepath_tag}_{figure_tag}"
            stats_details.to_csv(os.path.join(self.out_dir, f"{self.sid}{self.enc_file_label}_{stats_label}_stats.csv"), index=False)
            # against zero: only the five DNN feature spaces go in the table (none of the annotated features is significant)
            dnn_features = ['Motion Model', 'Image Model', 'Speech Model', 'word2vec_chunk120', 'Sentence Model']
            stats_helpers.generate_latex_lme_details_table(
                stats_details, os.path.join(self.dir, 'tables', f"lme_{stats_to_do}_{self.model}_{filepath_tag}_{figure_tag}.tex"),
                mode=stats_to_do, features=dnn_features if stats_to_do == 'compare_to_zero' else None,
                first=[f for f in stats_details.get('reference', pd.Series(dtype=str)).unique() if '+motion' in f])  # vision - language

        # 3. Optional extra plots
        if plot_stacked:
            localizers=[c.split(' (')[0] for c in col_order]
            self.temporal_merge_set = ['aTemp', 'pTemp']
            if average_posterior_anterior:
                order_dict = {}
                for loc in localizers:
                    masks = self.localizer_masks[loc]
                    merged_mask = []
                    for m in masks:
                        if m in self.STS:
                            if 'STS' not in merged_mask:
                                merged_mask.append('STS')
                        elif m in self.temporal_merge_set:
                            if 'temporal' not in merged_mask:
                                merged_mask.append('temporal')
                        else:
                            merged_mask.append(m)
                    order_dict[loc] = [f"{h} {m}" for m in merged_mask for h in hemis]
            else:
                order_dict = {
                    loc: [f"{h} {m}" for m in self.localizer_masks[loc] for h in hemis]
                    for loc in localizers
                }
            noise_ceiling_means = None
            noise_ceiling_sems = None
            for use_median in (False, True):
                plotting_helpers.plot_stacked_bars(
                    self=self,
                    results=temp_df,
                    localizers=localizers,
                    localizer_masks=self.localizer_masks,
                    order_dict=order_dict,
                    palette=self.colors_dict,
                    feature_names=self.plot_features,
                    hue=hue,
                    width_ratios=width_ratios,
                    stats_to_do=stats_to_do,
                    pvalue_dict={k: v for d in pvalue_results for k, v in d.items()},
                    plot_noise_ceiling=plot_noise_ceiling,
                    noise_ceiling_means=noise_ceiling_means,
                    noise_ceiling_sems=noise_ceiling_sems,
                    restrict_legend=restrict_legend,
                    response_label=response_label,
                    file_label=file_label,
                    file_suffix=f"{filepath_tag}_{figure_tag}",
                    use_median=use_median,
                    label_region=label_region,
                )
            
        if hue!='model': #ie if we are plotting performance, we don't do the anova
            anova_df = stats_helpers.run_anova_per_region_pair_per_hemisphere(
                data=temp_df,
                plot_features=[self.labels_dict.get(feature,feature) for feature in self.plot_features]
            )
            # Shorten filename to avoid OS limits (255 chars)
            filename_parts = [
                self.sid,
                f"mdl-{self.model}",
                file_label.replace('glm_localizer_', 'glm_'),
                f"top{self.perc_top_voxels}pct",
                f"resp-{response_label}",
                label,
                filepath_tag,
                figure_tag,
                "anova_interact.csv"
            ]
            csv_filename = "_".join(p for p in filename_parts if p)  # Filter empty strings
            csv_path = os.path.join(self.out_dir, csv_filename)
            anova_df.to_csv(csv_path, index=False)
            
            output_path = os.path.join(self.dir,'tables',f"pairwise_anova_{self.model}_{filepath_tag}_{figure_tag}.tex")
            stats_helpers.generate_latex_anova_pairwise_table(anova_df,output_path,include_hemi=~average_left_right)
    def generate_binary_localizer_maps_glm(self,plot=True,glm_task='SIpointlights'):
        print('generating binary glm voxel selection maps:')
        glm_file_label = '_smoothingfwhm-'+str(self.smoothing_fwhm)
        
        for localizer_contrast in self.localizer_contrasts[glm_task]:
            for subject in tqdm(self.subjects[glm_task],desc=localizer_contrast):
                combined_masks = np.zeros(self.brain_shape)
                for mask in self.localizer_masks[localizer_contrast]:
                    subject_map = np.zeros(self.brain_shape)
                    file_label = subject+glm_file_label+'_mask-'+mask#+'_encoding_model-'+self.model

                    filepath = self.out_dir + '/localizer_masks/'+file_label+'_glm_loc-'+localizer_contrast+'_run-'+self.all_runs[glm_task]+'.nii.gz'
                    localizer_map_img = nibabel.load(filepath)
                    localizer_map = localizer_map_img.get_fdata()
                    subject_map = subject_map+(localizer_map>0)*1.0
                    subject_map_img = nibabel.Nifti1Image(subject_map,localizer_map_img.affine)
                    # if(plot):
                    #   map_filename = self.figure_dir + '/localizer_masks/'+file_label+'_glm_loc-'+localizer_contrast+'_run-all_binary.pdf'
                    #   helpers.plot_map(subject_map_img,map_filename,threshold=0,vmax=len(self.run_groups[glm_task]),cmap='Greens')
                    subject_map = (subject_map>0)*1.0 #binarize to save to compare to enc localizer
                    map_img_filename = self.out_dir + '/localizer_masks/'+file_label+'_glm_loc-'+localizer_contrast+'_run-all_binary.nii.gz'
                    binary_map_img = nibabel.Nifti1Image(subject_map,localizer_map_img.affine)
                    nibabel.save(binary_map_img,map_img_filename)

                    combined_masks = combined_masks+subject_map
                
                file_label = subject+glm_file_label+'_mask-'+'_'.join(self.localizer_masks[localizer_contrast])#'sub-all_encoding_model-'+self.model
                combined_masks_map_img = nibabel.Nifti1Image(combined_masks,localizer_map_img.affine)
                map_img_filename = self.out_dir + '/localizer_masks/'+file_label+'_glm_loc-'+localizer_contrast+'_run-all_binary.nii.gz'
                nibabel.save(combined_masks_map_img,map_img_filename)
                if(plot):
                    color = self.colors_dict[localizer_contrast]
                    cmap = LinearSegmentedColormap.from_list('my_gradient', ((0.000, color),(1.000, color)))
                    map_filename = self.figure_dir + '/localizer_masks/'+file_label+'_glm_loc-'+localizer_contrast+'_run-all_binary'
                    plotting_helpers.plot_surface(nii=combined_masks_map_img,filename=map_filename,ROI_niis=[combined_masks_map_img],ROIs=[localizer_contrast],ROI_colors=['white'],views=['lateral'],threshold=0,vmax=None,cmap=cmap)
    def generate_binary_localizer_maps_enc(self,model=None,plot=True,label='ind_feature_performance'):
        print('generating binary maps of voxels with high '+label+' in the encoding model...')
        enc_file_label = '_encoding_model-'+model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
        # print(label)
        if(label=='performance'):
            features_loc = [self.model]
            measure_label = 'perf_raw'
        if(label=='ind_feature_performance'):
            measure_label = 'ind_perf_raw'
            features_loc = self.plot_features_dict[model]#feature_names#['motion','alexnet','social','face','valence']
        elif(label=='ind_product_measure'):
            measure_label = 'ind_product_measure_raw'
            features_loc = self.plot_features_dict[model]
        elif(label=='unique_variance'):
            measure_label = 'unique_var'
            features_loc = self.plot_features_dict[model]#feature_names
        elif(label=='shared_variance'):
            measure_label = 'shared_var'
            features_loc = ['glove-social','DNN-social','DNN-glove']#'speaking','glove','DNN']

        for feature_name in features_loc:
            for subject in tqdm(self.subjects['SIpointlights'],desc=feature_name):
                combined_masks = np.zeros(self.brain_shape)
                for mask in self.localizer_masks[feature_name]:
                    #get filepath
                    file_label = subject+enc_file_label+'_mask-'+mask+'_measure-'+measure_label#'_encoding_model-'+self.model
                    filepath = self.out_dir + '/localizer_masks/'+file_label+'_enc_feature_loc-'+feature_name+'.nii.gz'
                    localizer_map_img = nibabel.load(filepath)
                    localizer_map = localizer_map_img.get_fdata()
                    subject_map= localizer_map
                    subject_map_img = nibabel.Nifti1Image(subject_map,localizer_map_img.affine)
                    # if(plot):
                    #   map_filename = self.figure_dir + '/localizer_masks/'+file_label+'_enc_feature_loc-'+feature_name+'.pdf'
                    #   helpers.plot_img_volume(subject_map_img,map_filename,threshold=0,vmax=1,cmap='Greens')
                    subject_map = (localizer_map>0)*1.0 
                    map_img_filename = self.out_dir + '/localizer_masks/'+file_label+'_enc_feature_loc-'+feature_name+'_binary.nii.gz'
                    binary_map_img = nibabel.Nifti1Image(subject_map,localizer_map_img.affine)
                    nibabel.save(binary_map_img,map_img_filename)
                    combined_masks = combined_masks+subject_map

                file_label = subject+enc_file_label+'_mask-'+'_'.join(self.localizer_masks[feature_name])+'_measure-'+measure_label#'sub-all_encoding_model-'+self.model
                combined_masks_map_img = nibabel.Nifti1Image(combined_masks,localizer_map_img.affine)
                map_img_filename = self.out_dir + '/localizer_masks/'+file_label+'_enc_feature_loc-'+feature_name+'_binary.nii.gz'
                nibabel.save(combined_masks_map_img,map_img_filename)
                if(plot):
                    color = self.colors_dict[feature_name]
                    cmap = LinearSegmentedColormap.from_list('my_gradient', ((0.000, color),(1.000, color)))
                    map_filename = self.figure_dir + '/localizer_masks/'+file_label+'_enc_feature_loc-'+feature_name+'_binary'
                    plotting_helpers.plot_surface(nii=combined_masks_map_img,filename=map_filename,ROI_niis=[combined_masks_map_img],ROIs=[feature_name],ROI_colors=['black'],views=['lateral'],threshold=0,vmax=None,cmap=cmap)
    def compute_response_similarity(self, load=False, plot=True, selection_type='top_percent', pvalue=None, 
                                 response='weights', split_hemi=True, axes={}, filepath_tag='', average_posterior_anterior=False,
                                 plot_matrix=True, plot_bars=False, bar_points=False, save_source_data=False, plot_cbar=True):
        """save_source_data: save each participant's response similarity (Spearman r) for every plotted region
        pair and feature space as a csv next to the matrix figure (with average_posterior_anterior, the pairs of
        posterior and anterior regions are averaged within each participant)."""

        ##### 1. Prepare mask pairs (NO collapsing) #####
        def prepare_mask_pairs():
            if selection_type == 'top_percent':
                suffix = '_binary'
            elif selection_type == 'all_significant':
                suffix = f'_sig-{pvalue}'

            social = ['interact-no_interact']
            lang = ['intact-degraded']
            motion = ['interact&no_interact']

            names_social = [(n, f'glm-SIpointlights-{roi}') for n in social for roi in self.localizer_masks['SI pointlights']]
            names_lang = [(n, f'glm-language-{roi}') for n in lang for roi in self.localizer_masks['language']]
            names_motion = [(n, f'glm-SIpointlights-{roi}') for n in motion for roi in self.localizer_masks['motion pointlights']]

            names = names_motion + names_social + names_lang
            if(split_hemi):
                hemis = ['left','right']
            else:
                hemis = ['both']
            
            pairs = []
            for (n1, t1) in names:
                for (n2, t2) in names:
                    for hemi1 in hemis:
                        for hemi2 in hemis:
                            key = tuple(sorted([(n1, t1), (n2, t2)]))
                            pairs.append((key[0], key[1], (hemi1, hemi2)))

            pairs = list(set(pairs))
            pairs.sort()
            return pairs, names

        ##### 2. Preload data #####
        def preload_data(pairs):
            files = {}
            for subject in self.subjects['sherlock']: ####TODO set back to 'sherlock'!!!! and figure out a way to do run-123 for subjects without language loc!
                if response == 'weights':
                    data_filepath = f"{self.enc_dir}/{response}/{subject}_encoding_model-{self.model}{self.enc_file_label}_measure-{response}_raw.h5"
                    files[('response', subject)] = data_filepath
                else:
                    if response == 'ind_product_measure':
                        data_filepath = f"{self.enc_dir}/{response}/{subject}_encoding_model-{self.model}{self.enc_file_label}_measure-ind_product_measure_raw.nii.gz"
                    else:
                        data_filepath = f"{self.enc_dir}/{response}/{subject}{self.enc_file_label}_measure-{response}.nii.gz"
                    files[('response', subject)] = nibabel.load(data_filepath).get_fdata()

                for ((mask1, type1), (mask2, type2), (hemi1, hemi2)) in pairs:
                    for mask, mtype in [(mask1, type1), (mask2, type2)]:
                        ROI = mtype.split('-')[-1]
                        if 'encoding' in mtype:
                            localizer_file = f"{subject}_encoding_model-{self.model}_smoothingfwhm-{self.smoothing_fwhm}_chunklen-{self.chunklen}_mask-{ROI}_measure-ind_product_measure_raw_enc_feature_loc-{mask}.nii.gz"
                        else:
                            task = mtype.split('-')[1]
                            if subject not in self.subjects[task]:
                                continue
                            localizer_file = f"{subject}_smoothingfwhm-{self.smoothing_fwhm}_mask-{ROI}_glm_loc-{mask}_run-{self.all_runs[task]}.nii.gz"

                        full_path = os.path.join(self.out_dir, 'localizer_masks', localizer_file)
                        localizer = nibabel.load(full_path).get_fdata()
                        files[('localizer', mask, mtype, subject)] = localizer

            return files

        ##### 3. Compute similarity for one subject #####
        def process_subject(subject, pairs, files):
            results = []

            if response == 'weights':
                with h5py.File(files[('response', subject)], 'r') as f:
                    data = f['weights'][()]
            else:
                data = files[('response', subject)]

            for ((mask1, type1), (mask2, type2), (hemi1, hemi2)) in pairs:
                try:
                    roi1 = type1.split('-')[-1]
                    roi2 = type2.split('-')[-1]

                    if (roi1 == roi2) & (hemi1 == hemi2):
                        continue

                    mask1_data = files[('localizer', mask1, type1, subject)] > 0
                    mask2_data = files[('localizer', mask2, type2, subject)] > 0

                    if(split_hemi):
                        self.mask = helpers.load_mask(self, f"{hemi1}-{self.mask_name}")
                    else:
                        self.mask = helpers.load_mask(self, f"{self.mask_name}")
                    mask1_brain = self.mask.get_fdata()

                    if(split_hemi):
                        self.mask = helpers.load_mask(self, f"{hemi2}-{self.mask_name}")
                    else:
                        self.mask = helpers.load_mask(self, f"{self.mask_name}")
                    mask2_brain = self.mask.get_fdata()

                    mask1_data[mask1_brain == 0] = 0
                    mask2_data[mask2_brain == 0] = 0

                    for axis in axes.keys():
                        features = axes[axis]
                        features_idx = np.concatenate([
                            np.atleast_1d(self.get_feature_index(f, weight=(response == 'weights'))) 
                            for f in features
                        ])

                        if response == 'weights':
                            self.mask = helpers.load_mask(self, self.mask_name)
                            mask1_flat = mask1_data[self.mask.get_fdata() == 1]
                            mask2_flat = mask2_data[self.mask.get_fdata() == 1]
                        else:
                            mask1_flat = mask1_data
                            mask2_flat = mask2_data

                        data1 = np.nanmean(data[:, mask1_flat], axis=1)
                        data2 = np.nanmean(data[:, mask2_flat], axis=1)

                        if len(data1) > 0:
                            corr1 = data1[features_idx]
                            corr2 = data2[features_idx]
                            valid = ~(np.isnan(corr1) | np.isnan(corr2))

                            r, p = scipy.stats.spearmanr(corr1[valid], corr2[valid])
                        else:
                            r, p = np.nan, np.nan

                        row = (subject, axis, f"{mask1}-{roi1}", f"{mask2}-{roi2}", type1.split('-')[1], type2.split('-')[1],
                            hemi1, hemi2, f"{hemi1}_{hemi2}", r, r, p)
                        results.append(row)
                except Exception as e:
                    # print(e)
                    continue

            return results

        ##### 4. Collapsing function (post-hoc averaging) #####
        def collapse_label(label):
            parts = label.split('-')
            name = '-'.join(parts[:-1])
            roi = parts[-1]
            if roi in self.STS:
                roi_collapsed = 'STS'
            elif roi in ['aTemp','pTemp']:
                roi_collapsed = 'temporal'
            else:
                roi_collapsed = roi
            return f"{name}-{roi_collapsed}"

        ##### 5. Main execution starts here #####

        pairs, names = prepare_mask_pairs()
        names = [
            f"{contrast}-{mask_type.split('-')[-1]}" 
            for (contrast, mask_type) in names
        ]
        file_label = f"{self.sid}{self.enc_file_label}_model-{self.model}_{response}_response_similarity"
        result_file = os.path.join(self.out_dir, f"{file_label}_perc_top_voxels-{self.perc_top_voxels}_enc_feature_localizer_results_{filepath_tag}.csv")

        file_label_new = f"{self.sid}{self.enc_file_label_new}_model-{self.model}_{response}_response_similarity"
        result_file_new = os.path.join(self.out_dir, f"{file_label_new}_perc_top_voxels-{self.perc_top_voxels}_enc_feature_localizer_results_{filepath_tag}.csv")

        if load and not os.path.exists(result_file) and os.path.exists(result_file_new):
            result_file = result_file_new

        if load and not os.path.exists(result_file) and not os.path.exists(result_file_new):
            raise FileNotFoundError(
                f"load=True but no cached result file exists. Tried:\n"
                f"  {result_file}\n  {result_file_new}\n"
                f"Set load=False to compute, or pass a matching filepath_tag."
            )

        if not load:
            files = preload_data(pairs)
            all_results = Parallel(n_jobs=1)(
                delayed(process_subject)(subj, pairs, files) 
                for subj in tqdm(self.subjects['sherlock']) 
            )
            all_results = pd.DataFrame(np.vstack(all_results), columns=[
                'subject', 'axis', 'mask1_name', 'mask2_name', 'mask1_measure_label',
                'mask2_measure_label', 'hemi1', 'hemi2', 'hemi_label', 'corr', 'corr1', 'pvalue1'
            ])
            all_results.to_csv(result_file, index=False)
        
        all_results = pd.read_csv(result_file)
        numeric_cols = ['corr', 'corr1', 'pvalue1']
        for col in numeric_cols:
            all_results[col] = pd.to_numeric(all_results[col], errors='coerce')

        # Auto-detect split_hemi from CSV: if it has only 'both' rows, override to False
        unique_hemis = set(all_results['hemi1'].unique()) | set(all_results['hemi2'].unique())
        if split_hemi and unique_hemis == {'both'}:
            print(f"  [compute_response_similarity] CSV only has hemi='both' — overriding split_hemi=False for plotting")
            split_hemi = False

        all_results['original_mask1_name'] = all_results['mask1_name']
        all_results['original_mask2_name'] = all_results['mask2_name']

        ##### Apply post-hoc posterior/anterior averaging #####
        if average_posterior_anterior:
            all_results['mask1_name'] = all_results['mask1_name'].apply(collapse_label)
            all_results['mask2_name'] = all_results['mask2_name'].apply(collapse_label)

            collapsed_names = []
            seen = set()
            for name in names:
                roi = name.split('-')[-1]
                label = collapse_label(name)
                if label not in seen:
                    collapsed_names.append(label)
                    seen.add(label)
            names = collapsed_names


        ##### Plotting #####
        if plot:
            for i, axis in enumerate(axes.keys()):
                if plot_matrix:
                    similarity_matrix, _ = plotting_helpers.plot_similarity_matrix(
                        results_df=all_results,
                        names=names,
                        axis=axis,
                        split_hemi=split_hemi,
                        label_dict=self.labels_dict,
                        output_path=os.path.join(self.figure_dir, 'response_similarity',
                            f"{file_label}_perc_top_voxels-{self.perc_top_voxels}-{axis}_similarity_matrix_{filepath_tag}.png"),
                        plot_cbar=plot_cbar
                    )
                if save_source_data:
                    source_df = all_results[(all_results['axis'] == axis) & (all_results['mask1_name'] != all_results['mask2_name'])].copy()
                    # put each pair in the matrix's region order (the matrix averages both orders of a pair)
                    swap = source_df['mask1_name'].map(names.index) > source_df['mask2_name'].map(names.index)
                    for col1, col2 in [('mask1_name', 'mask2_name'), ('original_mask1_name', 'original_mask2_name'), ('hemi1', 'hemi2')]:
                        source_df.loc[swap, [col1, col2]] = source_df.loc[swap, [col2, col1]].values
                    label = lambda mask_label: plotting_helpers.similarity_region_label(mask_label, newline=False)
                    source_df = pd.DataFrame({
                        'subject': source_df['subject'],
                        'feature_space': plotting_helpers.SIMILARITY_TITLE_DICT.get(axis, axis),
                        'hemisphere1': source_df['hemi1'], 'region1': source_df['mask1_name'].map(label),
                        'hemisphere2': source_df['hemi2'], 'region2': source_df['mask2_name'].map(label),
                        'spearman_r': source_df['corr']})
                    if not split_hemi:
                        source_df = source_df.drop(columns=['hemisphere1', 'hemisphere2'])
                    # one value per participant per plotted region pair
                    source_df = source_df.groupby([col for col in source_df.columns if col != 'spearman_r'],
                                                  sort=False, as_index=False)['spearman_r'].mean()
                    source_path = os.path.join(self.figure_dir, 'response_similarity',
                        f"{file_label}_perc_top_voxels-{self.perc_top_voxels}-{axis}_similarity_matrix_{filepath_tag}_source_data.csv")
                    source_df.to_csv(source_path, index=False)
                    print('response similarity source data saved:', source_path)
                fit, results_df = stats_helpers.run_similarity_superregion_LME(all_results, axis,split_hemi=split_hemi,average_posterior_anterior=average_posterior_anterior)
                if plot_bars:
                    plotting_helpers.plot_similarity_bars(
                        results_df=all_results,
                        names=names,
                        axis=axis,
                        color=plotting_helpers.get_colors_dict()[axis],
                        show_points=bar_points,
                        contrasts=results_df,
                        output_path=os.path.join(self.figure_dir, 'response_similarity',
                            f"{file_label}_perc_top_voxels-{self.perc_top_voxels}-{axis}_similarity_bars_{filepath_tag}.png"),
                    )
                csv_path = os.path.join(self.out_dir, f"{file_label}_perc_top_voxels-{self.perc_top_voxels}_{filepath_tag}_{axis}_response_similarity_stats.csv")
                fit.to_csv(csv_path, index=False)

                csv_path = os.path.join(self.out_dir, f"{file_label}_perc_top_voxels-{self.perc_top_voxels}_{filepath_tag}_{axis}_response_similarity_pairwise_comparisons.csv")
                results_df.to_csv(csv_path, index=False)
                hemi_label = 'hemis_together'
                if(split_hemi):
                    hemi_label = 'split_hemis'
                output_path = os.path.join(self.dir, 'tables', f"response_similarity_pairwise_comparisons_{axis}_{hemi_label}.tex")
                stats_helpers.generate_latex_pairwise_similarity_table_with_hemisphere(results_df, output_path,include_hemi=split_hemi)
                    
    def compute_all_overlap2(self,load=False,plot=True,selection_type='top_percent',pvalue=None,regions=[],label='',file_tag=''):
        from matplotlib import colors
        print('computing overlap between voxel groups:')

        if(selection_type=='top_percent'):
            suffix = '_binary'
            folder = '/localizer_masks/'
            overlap_folder = '/localizer_overlap_maps/'

        cmap = colors.ListedColormap(['white','mediumblue','gold','green','white','mediumblue','gold','green'])
        names = regions
        selected_features = []
        for selections in [[(name1,name1_model_type,model1,mask1),(name2,name2_model_type,model2,mask2)] for (name1,name1_model_type,model1,mask1) in names for (name2,name2_model_type,model2,mask2) in names]:
            selections.sort()
            selected_features.append((selections[0],selections[1]))

        selected_features = list(set(selected_features))
        selected_features.sort()

        if(not load):
            def process(subject,name1,name1_type,model1,mask1,name2,name2_type,model2,mask2,hemi,plot):
                def get_info(subject,data1_filepath,data2_filepath,name1,name2,name1_measure_label,name2_measure_label,model1,model2,name1_model,name2_model,mask1,mask2,hemi,plot_filepath):
                    try:
                        data1_img = nibabel.load(data1_filepath)
                        data2_img = nibabel.load(data2_filepath)

                        data1 = (data1_img.get_fdata()>0)*1.0
                        data2 = (data2_img.get_fdata()>0)*2.0
                        
                        overlap = data1+data2
                        overlap_img = nibabel.Nifti1Image(overlap.astype('int32'),data1_img.affine)
                        if(plot_filepath!='None'):
                            title = name1+': blue, '+name2+': yellow, overlap: green'
                            plotting_helpers.plot_img_volume(overlap_img,plot_filepath,threshold=0.99,cmap=cmap,title=title,vmax=3)
                        
                        #### save a map of the overlap only
                        overlap_only = (overlap==3)*1
                        overlap_only_img = nibabel.Nifti1Image(overlap_only.astype('int32'),data1_img.affine)
                        filepath = filepath1.split(subject)[0]+subject+'_overlap-'+name1+'_'+name2+'_binary.nii.gz'
                        nibabel.save(overlap_only_img, filepath)  
                        ####

                        total_voxels1_all = len(data1[data1==1])
                        total_voxels2_all = len(data2[data2==2])
                        total_voxels_all = total_voxels1_all+total_voxels2_all
                        voxels_overlap_all = len(overlap[overlap==3])

                        fullway = overlap.shape[0]
                        halfway = int(fullway/2)

                        if(hemi=='left'):
                            overlap=overlap[0:halfway]
                            data1= data1[0:halfway]
                            data2= data2[0:halfway]
                        elif(hemi=='right'):
                            overlap=overlap[halfway+1:fullway]
                            data1 = data1[halfway+1:fullway]
                            data2 = data2[halfway+1:fullway]

                        total_voxels1 = len(data1[data1==1])
                        total_voxels2 = len(data2[data2==2])
                        total_voxels = total_voxels1+total_voxels2

                        voxels1_only = len(overlap[overlap==1])
                        voxels2_only = len(overlap[overlap==2])
                        voxels_overlap = len(overlap[overlap==3])

                        if(total_voxels==0):
                            total_voxels = np.nan
                        if(total_voxels_all==0):
                            total_voxels_all = np.nan
                        if(voxels_overlap_all==0):
                            voxels_overlap_all = np.nan

                        DICE_coef = total_voxels and (2*voxels_overlap)/total_voxels
                        proportion_of_voxels1_in_this_hemi = total_voxels1_all and total_voxels1/total_voxels1_all
                        proportion_of_voxels2_in_this_hemi = total_voxels2_all and total_voxels2/total_voxels2_all
                        proportion_of_all_voxels_in_this_hemi = total_voxels_all and total_voxels/total_voxels_all
                        proportion_of_overlap_in_this_hemi = voxels_overlap_all and voxels_overlap/voxels_overlap_all

                        proportion_of_1_that_is_also_2 = total_voxels1 and voxels_overlap/total_voxels1
                        proportion_of_2_that_is_also_1 = total_voxels2 and voxels_overlap/total_voxels2

                        proportion_of_voxels_that_is_voxels1 = total_voxels and voxels1_only/total_voxels
                        proportion_of_voxels_that_is_voxels2 =  total_voxels and voxels2_only/total_voxels
                        proportion_of_voxels_that_is_overlap =  total_voxels and voxels_overlap/total_voxels

                        names = [(name1,name1_measure_label),(name2,name2_measure_label)]
                        names.sort()
                        names_label = names[0][0]+'-'+names[0][1]+'_'+names[1][0]+'-'+names[1][1]
                    except Exception as e:
                        print(e)
                        results = (np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)
                        return results

                    results = (subject, name1,name2,names_label,name1_measure_label,name2_measure_label,name1_model,name2_model,mask1,mask2,hemi,total_voxels1,total_voxels2,total_voxels,voxels1_only, voxels2_only, voxels_overlap,DICE_coef,
                        proportion_of_voxels1_in_this_hemi,proportion_of_voxels2_in_this_hemi,proportion_of_all_voxels_in_this_hemi,proportion_of_overlap_in_this_hemi, 
                        proportion_of_1_that_is_also_2, proportion_of_2_that_is_also_1,
                        proportion_of_voxels_that_is_voxels1, proportion_of_voxels_that_is_voxels2,proportion_of_voxels_that_is_overlap)
                    return results

                enc_file_label1 = subject+'_encoding_model-'+model1+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen) #+ '_mask-'+mask_name
                enc_file_label2 = subject+'_encoding_model-'+model2+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                glm_file_label = subject+'_smoothingfwhm-'+str(self.smoothing_fwhm) #+ '_mask-'+mask_name


                name1_model = name1_type.split('-')[0]
                name1_measure_label = name1_type.split('-')[1]
                name2_model = name2_type.split('-')[0]
                name2_measure_label = name2_type.split('-')[1]

                if(name1_model=='encoding'):
                    filepath1 = self.out_dir + folder+enc_file_label1+ '_mask-'+mask1+'_measure-'+name1_measure_label+'_enc_feature_loc-'+name1+suffix+'.nii.gz'
                elif(name1_model=='glm'):
                    filepath1 = self.out_dir + folder+glm_file_label+ '_mask-'+mask1+'_glm_loc-'+name1+'_run-all'+suffix+'.nii.gz'
                elif(name1_model=='overlap'):
                    filepath1 = self.out_dir + folder+subject+ '_overlap-'+name1+suffix+'.nii.gz'

                if(name2_model=='encoding'):
                    filepath2 = self.out_dir + folder+enc_file_label2+ '_mask-'+mask2+'_measure-'+name2_measure_label+'_enc_feature_loc-'+name2+suffix+'.nii.gz'
                elif(name2_model=='glm'):
                    filepath2 = self.out_dir + folder+glm_file_label+ '_mask-'+mask2+'_glm_loc-'+name2+'_run-all'+suffix+'.nii.gz'
                elif(name2_model=='overlap'):
                    filepath2 = self.out_dir + folder+subject+ '_overlap-'+name2+suffix+'.nii.gz'
                
                if(plot):
                    plot_filepath = self.figure_dir + overlap_folder+self.enc_file_label+'_'+name1_type+'-'+name1+'_'+name2_type+'-'+name2+'_'+selection_type+'.pdf'

                #check if subject has all of the localizers, only some subjects have the language localizer
                subject_has_localizers = True
                if (name1=='intact-degraded')|(name2=='intact-degraded') :
                    if(subject not in self.subjects['language']):
                        subject_has_localizers = False
                if subject_has_localizers:
                    results = get_info(subject,filepath1,filepath2, name1,name2,name1_measure_label,name2_measure_label,model1,model2,name1_model,name2_model,mask1,mask2,hemi,plot_filepath)
                else:
                    results = [np.nan]*27
                # print(subject+'_name1-'+name1+'_measure-'+name1_measure_label+'_name2-'+name2+'_measure-'+name2_measure_label)

                return results


            results = Parallel(n_jobs=-1)(delayed(process)(subject,name1,name1_type,model1,mask1,name2,name2_type,model2,mask2,hemi,plot) for subject in tqdm(self.subjects['SIpointlights']) for ((name1,name1_type,model1,mask1),(name2,name2_type,model2,mask2)) in selected_features for hemi in ['all','left','right'])
            results = np.array(results)
            results = pd.DataFrame(results,columns = ['subject', 'name1','name2','names','name1_measure','name2_measure','name1_model','name2_model','mask1','mask2','hemi','total_voxels1','total_voxels2','total_voxels','voxels1_only','voxels2_only','voxels_overlap','DICE_coef',
                        'proportion_of_voxels1_in_this_hemi','proportion_of_voxels2_in_this_hemi','proportion_of_all_voxels_in_this_hemi','proportion_of_overlap_in_this_hemi', 
                        'proportion_of_1_that_is_also_2', 'proportion_of_2_that_is_also_1',
                        'proportion_of_voxels_that_is_voxels1', 'proportion_of_voxels_that_is_voxels2','proportion_of_voxels_that_is_overlap'])
            file_label = self.sid+self.enc_file_label+'_'+selection_type+'_overlap2'
            results.to_csv(self.out_dir+'/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_'+label+'_enc_feature_localizer_results_'+file_tag+'.csv')
        
        file_label = self.sid+self.enc_file_label+'_'+selection_type+'_overlap2'
        #reload the csv so that all dtypes are automatically correct
        results = pd.read_csv(self.out_dir+'/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_'+label+'_enc_feature_localizer_results_'+file_tag+'.csv')        
        file_label = 'overlap/'+file_label

        overlap_matrix = np.zeros(())

        def get_overlap_matrix(names,hemi):
            overlap_matrix = np.zeros((len(names),len(names)))
            std_matrix=overlap_matrix.copy()

            ind_tracker = []

            for (ind1,name1) in enumerate(names):
                for (ind2,name2) in enumerate(names):
                    if(set((ind1,ind2)) not in ind_tracker):
                        temp_names = [name1,name2]
                        temp_names.sort()
                        # print(temp_names)
                        names_label = temp_names[0][0]+'-'+temp_names[0][1].split('-')[1]+'_'+temp_names[1][0]+'-'+temp_names[1][1].split('-')[1]
                        # print(names_label)
                        temp_results = results[(results.hemi==hemi)&(results.names==names_label)]#&(results.mask_name==mask_name)]
                        # stat_result = stats.ttest_1samp(temp_results.DICE_coef.dropna(),0)
                        # print(temp_results.mask_name)

                        value1 = np.nanmean(temp_results.DICE_coef)#proportion_of_1_that_is_also_2)
                        value2 = np.nanmean(temp_results.DICE_coef)#proportion_of_2_that_is_also_1)
                        std = np.nanstd(temp_results.DICE_coef)

                        overlap_matrix[ind1][ind2]=value1
                        overlap_matrix[ind2][ind1]=value2
                        std_matrix[ind1][ind2]=std
                        std_matrix[ind2][ind1]=std
                        ind_tracker.append(set((ind1,ind2)))
                        if (not np.isnan(value1)) & (name1!=name2):
                            print(f"[{hemi.upper()}] {name1} vs {name2}: "
                                f"mean overlap = {value1:.3f}, std = {std:.3f}, n = {len(temp_results)}")

                        
            overlap_matrix[np.triu_indices(overlap_matrix.shape[0],0)] = np.nan
            overlap_matrix[0,0] =np.nan
            overlap_matrix[overlap_matrix.shape[0]-1,overlap_matrix.shape[0]-1]=np.nan
            fig = plt.figure(figsize=(len(names)*2,len(names)*1.25)) #figure size depends on how many names we have
            ax = fig.add_subplot(111)
            cax = ax.imshow(overlap_matrix,cmap='Greens',vmin=0.0,vmax=0.5,interpolation='nearest')

            sns.despine(left=True,bottom=True)
            fig.colorbar(cax)
            plt.title(hemi+' '+' and '.join(label.split('_')))
            ax.set_xticks(range(0,len(names)-1))
            ax.set_yticks(range(1,len(names)))
                

            for i in range(len(names)):
                for j in range((len(names))):
                    c = overlap_matrix[j,i]
                    std = std_matrix[j,i]
                    # pvalue = pvalue_matrix[j,i]
                    if(~np.isnan(c)):
                        ax.text(i, j, str(np.round(c,2)) +','+str(np.round(std,2)), va='center', ha='center',color='black')
                    #     if(pvalue<0.001):
                    #         ax.text(i, j, str(np.round(c,4)), va='center', ha='center',fontweight='heavy')
                    #     elif(pvalue<0.05):
                    #         ax.text(i, j, str(np.round(c,4)), va='center', ha='center')
                    #     else:
                    #         ax.text(i, j, str(np.round(c,4)), va='center', ha='center',color='gray')

            name_dict = {'interact-no_interact':' controlled \n  social   \ninteraction',
                         'intact-degraded':'controlled\nlanguage',
                         'social':'movie\nsocial',
                         'sbert+word2vec':'movie\nsBERT+\nwordvec',
                         'sbert':'movie\nsBERT',
                         'face':'movie\nface',
                         'speaking':'movie\nspeaking',
                         'alexnet':'movie\nalexnet',
                         'hubert': 'movie\nHuBERT',
                         'motion': 'movie\nmotion',
                         'word2vec': 'movie\nword2vec',
                         }
            for feature in self.feature_names:
                name_dict[feature] = 'movie\n'+feature
            ax.set_xticklabels([name_dict[name[0]] for name in names[:-1]])
            ax.set_yticklabels([name_dict[name[0]] for name in names[1:]])
            plt.savefig(self.figure_dir + '/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_overlap_matrix_'+label+'_'+hemi+'_'+file_tag+'.png',bbox_inches='tight',dpi=300)
            plt.close()
            return overlap_matrix

        overlap_matrix_left = get_overlap_matrix(names,'left')
        overlap_matrix_right = get_overlap_matrix(names,'right')   
        return overlap_matrix_left,overlap_matrix_right
    def plot_localizer(self,task,p_threshold=1,vmin=None,vmax=None,symmetric_cbar=True,cmap='yellow_hot',plot_outlines = False):
        label = 'zscore'
        for contrast in self.localizer_contrasts[task]:
            subject_tqdm = tqdm(self.subjects[task])
            for subject in subject_tqdm:
                subject_tqdm.set_description(contrast + ': ' +subject)
                try:
                    # print(subject)
                    filepath = os.path.join(self.glm_dir,subject, subject+ "_task-"+ task+'_space-'+self.space+ "_run-"+ self.all_runs[task]+ "_measure-"+label + "_contrast-"+contrast+".nii.gz")
                    nii = nibabel.load(filepath)
                    # plot individual subject too
                    ROI_niis = []
                    filename = subject+self.glm_file_label+'_measure-'+label+'_contrast-'+contrast
                    if plot_outlines:
                        glm_file_label_ = '_smoothingfwhm-'+str(self.smoothing_fwhm)
                        file_label_ = subject+glm_file_label_+'_mask-'+'_'.join(self.localizer_masks[contrast])#mask
                        ROI_file = self.out_dir + '/localizer_masks/'+file_label_+'_glm_loc-'+contrast+'_run-all_binary.nii.gz'
                        try:
                            ROI_niis.append(nibabel.load(ROI_file))
                            filename = filename + '_loc-' + contrast
                        except Exception as e:
                            print(e)
                    ind_filepath = os.path.join(self.figure_dir,'glm_zscores',filename)
                    cmap_ = cmap
                    threshold_dict = {0.05: 1.72,
                                  0.01: 2.5,
                                  0.001: 3.6,
                                  1:None}
                    thresholded_map, threshold_corrected = threshold_stats_img(
                        nii, alpha=threshold_dict[p_threshold], height_control=None,two_sided=False #only care about positive direction?
                            )
                    plotting_helpers.plot_surface(thresholded_map,ind_filepath,threshold=0.01,title='',symmetric_cbar=symmetric_cbar,vmin=vmin,vmax=vmax,cmap=cmap_,colorbar_label='z-score', ROIs = [contrast], ROI_niis = ROI_niis, ROI_colors = ['white'])
                except Exception as e:
                    print(e)
                    pass
    def plot_map(self,feature='',measure='ind_feature_performance',localizers=[],threshold=0,vmin=None,vmax=None,cmap='yellow_hot',png_dpi=None):
        #png_dpi: also save a .png of each map at this dpi (for figures composed from pngs)
        fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')

        def _build_labels(model_name):
            old = '_encoding_model-'+model_name+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
            new = '_encoding_model-'+model_name+'_smoothingfwhm-'+str(self.smoothing_fwhm)
            if(self.mask_name!=None):
                old = old + '_mask-'+self.mask_name
                new = new + '_mask-'+self.mask_name + '_chunklen-'+str(self.chunklen)
            else:
                new = new + '_chunklen-'+str(self.chunklen)
            return new, old

        def _load_with_fallback(folder, subject, model_name, label):
            new_label, old_label = _build_labels(model_name)
            new_path = self.enc_dir+'/'+folder+'/'+subject+new_label+'_measure-'+label+'_raw.nii.gz'
            old_path = self.enc_dir+'/'+folder+'/'+subject+old_label+'_measure-'+label+'_raw.nii.gz'
            if os.path.exists(new_path):
                return nibabel.load(new_path), new_label
            if os.path.exists(old_path):
                return nibabel.load(old_path), old_label
            raise FileNotFoundError(f"Could not find file with either ordering:\n  New: {new_path}\n  Old: {old_path}")

        if(measure=='performance'):
            folder = 'performance'
            label = 'perf'
        if(measure=='ind_feature_performance'):
            folder = 'ind_feature_performance'
            label = 'ind_perf'
        if(measure=='ind_product_measure'):
            folder = 'ind_product_measure'
            label = 'ind_product_measure'
        elif(measure=='added_variance'):
            folder = 'performance'
            label = 'perf'
        plot_label = measure + ': ' + feature
        for subject in tqdm(self.subjects['SIpointlights']):
            # try:
                if(measure=='performance'):
                    nii1, enc_file_label = _load_with_fallback(folder, subject, self.model, label)
                    data1 = nii1.get_fdata()
                    data1[data1<0]=0 #clip negatives to 0 (negatives are not meaningful)
                elif(measure=='added_variance'):
                    base_model,subtract_model = helpers.get_added_variance_models(feature)

                    nii1, _ = _load_with_fallback(folder, subject, subtract_model, label)
                    subtract_model = nii1.get_fdata()
                    subtract_model[subtract_model<0] = 0

                    nii1, enc_file_label = _load_with_fallback(folder, subject, base_model, label)
                    base_model = nii1.get_fdata()
                    base_model[base_model<0] = 0

                    data1 = base_model-subtract_model
                else:
                    nii1, enc_file_label = _load_with_fallback(folder, subject, self.model, label)
                    data1 = nii1.get_fdata()
                    if self.clip_negative_product_measure:
                        data1[data1<0] = 0 #clip response values to 0
                    if(self.scale_by=='total_variance'):
                        if self.clip_negative_product_measure:
                            data1[data1<0] = 0 #clip response values to 0
                        data1 = data1/data1.sum(axis=0,keepdims=1)
                    if(feature in self.combined_features):
                        for (ind,sub_feature_name) in enumerate(self.model_features_dict[feature]):
                            feature_ind = self.get_feature_index(sub_feature_name)
                            sub_data = data1[feature_ind]
                            if(ind==0):
                                overall = sub_data
                            else:
                                overall = overall+sub_data
                        data1 = overall
                    else:
                        feature_index = self.get_feature_index(feature)
                        data1 = data1[feature_index]

                #project to surface

                #mask the with statistically significant performance maps for both SLIP and SimCLR

                ## TODO mask with statistically significant unique variance maps!!!

                #load the fdr corrected p value map and threshold it at p<0.05
                # threshold = threshold
                # enc_file_label = '_encoding_model-'+feature1+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                # if(self.mask_name!=None):
                #   enc_file_label = enc_file_label + '_mask-'+self.mask_name
                # filepath = self.enc_dir+'/unique_variance/'+subject+enc_file_label+'_feature-'+feature+'_measure-unique_var_p_fdr.nii.gz'
                # nii = nibabel.load(filepath)
                # p_fdr1 = nii.get_fdata()
                # p_fdr1_mask = p_fdr1*0
                # self.mask = helpers.load_mask(self,self.mask_name)
                # self.mask = nilearn.image.resample_img(self.mask, target_affine=nii.affine, target_shape=p_fdr1.shape,interpolation='nearest')
                # p_fdr1_mask[p_fdr1>threshold]=0
                # p_fdr1_mask[p_fdr1<threshold]=1
                # p_fdr1_mask[(self.mask.get_fdata()==0)]=0

                

                diff_map = data1
                # diff_map[(p_fdr1_mask==0)]= 0 #zero out any voxel that isn't significantly predicted by either feature space
                # diff_map = np.log10(data1/data2)
                diff_map_nii = nibabel.Nifti1Image(diff_map, nii1.affine)

                #plot difference map
                # enc_file_label = '_encoding_model-'+self.model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                # if(self.mask_name!=None):
                #   enc_file_label = enc_file_label + '_mask-'+self.mask_name
                # filename = self.figure_dir + '/map/'+subject+enc_file_label+'_measure-'+measure+'_feature-'+feature+'.png'
                # from matplotlib import colors
                # cmap = 'Greens'
                # helpers.plot_img_volume(diff_map_nii,filename,symmetric_cbar=False,threshold=0.000001,cmap=cmap,vmin=0,vmax=0.1,title=subject+', '+feature)

                glm_file_label_ = '_smoothingfwhm-'+str(self.smoothing_fwhm)
                # enc_file_label_ = '_encoding_model-full_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                enc_file_label_ = '_encoding_model-SimCLR_SLIP_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                # if(localizer!=None):
                #     if(localizer=='intact-degraded'):
                #         if(subject in self.subjects['language']):
                #             localizer_contrasts = [localizer]
                #         else:
                #             localizer_contrasts = []
                #     else:
                #         localizer_contrasts = [localizer]
                # else:
                #     localizer_contrasts = []
                # else:
                #   localizer_contrasts = []
                # localizer_contrasts = []
                # localizer_contrasts = ['social']
                # localizer_contrasts = ['SimCLR']

                all_ROI_data = np.zeros(diff_map.shape)
                ROI_niis = []
                filename = self.figure_dir + '/map/'+subject+enc_file_label+'_measure-'+measure
                if(measure!='performance'):
                    filename=filename+'_feature-'+feature
                for ind,localizer_contrast in enumerate(localizers):
                    file_label_ = subject+glm_file_label_+'_mask-'+'_'.join(self.localizer_masks[localizer_contrast])#mask
                    ROI_file = self.out_dir + '/localizer_masks/'+file_label_+'_glm_loc-'+localizer_contrast+'_run-all_binary.nii.gz'
                    try:
                        ROI_niis.append(nibabel.load(ROI_file))
                        filename = filename + '_loc-' + localizer_contrast
                    except Exception as e:
                        print(e)

                # all_ROI_nii = nibabel.Nifti1Image(all_ROI_data, nii1.affine)

                cmap_ = cmap
                # helpers.plot_lateral_surface(diff_map_nii,filename,symmetric_cbar=True,threshold=0.000000001,cmap='PuOr',vmax=None,title=subject+', '+feature1+'-'+feature2)
                plotting_helpers.plot_surface(diff_map_nii,filename,ROI_niis=ROI_niis,symmetric_cbar=False,threshold=threshold,cmap=cmap_,vmin=vmin,vmax=vmax,title='', colorbar_label='Explained variance $R^2$',ROIs=localizers,ROI_colors=['white' for localizer in localizers],png_dpi=png_dpi)#self.colors_dict[localizer])

            # except Exception as e:
            #   print(e)
            #   pass
class VoxelSelectionManager:
    def __init__(self, parent):
        """
        VoxelSelectionManager handles config building and file loading.
        'parent' is the main class (your big class) where self.models, self.dir, self.subjects, etc., live.
        """
        self.parent = parent

    def build_config(self, response_label):
        """Return the config for the given response label."""
        RESPONSE_CONFIG = {
            'performance': {
                'folder': 'performance',
                'resp_label': 'perf_raw',
                'feature_names': ['NA'],
                # cross-subject encoding is loaded separately (include_cross_subject) and added to the summary on save
                'models': [m for m in self.parent.models if m != 'cross-subject encoding'] or [self.parent.model]
            },
            'added_variance': {
                'folder': 'performance',
                'resp_label': 'perf_raw',
                'feature_names': self.parent.plot_features,
                'models': [self.parent.model]
            },
            'unique_variance': {
                'folder': 'performance',
                'resp_label': 'perf_raw',
                'feature_names': self.parent.plot_features,
                'models': [self.parent.model]
            },
            'ind_feature_performance': {
                'folder': 'ind_feature_performance',
                'resp_label': 'ind_perf_raw',
                'feature_names': self.parent.plot_features,
                'models': [self.parent.model]
            },
            'ind_product_measure': {
                'folder': 'ind_product_measure',
                'resp_label': 'ind_product_measure_raw',
                'feature_names': self.parent.plot_features,
                'models': [self.parent.model]
            },
            'features_preferred_delay': {
                'folder': 'features_preferred_delay',
                'resp_label': 'features_preferred_delay',
                'feature_names': self.parent.plot_features,
                'models': [self.parent.model]
            },
        }
        if response_label not in RESPONSE_CONFIG:
            raise ValueError(f"Unknown response_label: {response_label}")
        return RESPONSE_CONFIG[response_label]

    def load_files(self, config, response_label, load=False, include_cross_subject=False):
        """
        Load all necessary NIfTI files (GLM, encoding, cross-subject).
        """
        if load:
            return None  # Optionally, load precomputed things if needed

        files = {}

        # --- Load cross-subject files (only if requested)
        if include_cross_subject:
            for subject in self.parent.subjects['SIpointlights']:
                files[('cross-subject encoding', subject)] = nibabel.load(
                    os.path.join(self.parent.dir, 'analysis', 'Brain2Brain', 'performance',
                                 f"{subject}_brain2brain_encoding_smoothingfwhm-{self.parent.smoothing_fwhm}_chunklen-{self.parent.chunklen}_mask-ISC_measure-perf_raw.nii.gz")
                )

        # --- Load GLM files (localizers + responses)
        for glm_task in self.parent.glm_task:
            for loc_run, resp_run in self.parent.run_groups[glm_task] + [(self.parent.all_runs[glm_task], self.parent.all_runs[glm_task])]:
                for subject in self.parent.subjects[glm_task]:
                    for localizer in self.parent.localizer_contrasts[glm_task]:
                        files[(glm_task, loc_run, localizer, subject)] = nibabel.load(
                            os.path.join(self.parent.glm_dir, subject,
                                         f"{subject}_task-{glm_task}_space-{self.parent.space}_run-{loc_run}_measure-zscore_contrast-{localizer}.nii.gz")
                        )
                    for response_contrast in self.parent.response_contrasts[glm_task]:
                        files[(glm_task, resp_run, response_contrast, subject)] = nibabel.load(
                            os.path.join(self.parent.glm_dir, subject,
                                         f"{subject}_task-{glm_task}_space-{self.parent.space}_run-{resp_run}_measure-weights_contrast-{response_contrast}.nii.gz")
                        )

        # --- Load Encoding Model files
        models = config['models']
        feature_names = config['feature_names']
        response_folder = config['folder']
        resp_label = config['resp_label']

        for subject in self.parent.subjects['SIpointlights']:
            if response_label in ['added_variance', 'unique_variance']:
                models_to_load = self._get_special_models(response_label, feature_names)
            else:
                models_to_load = models

            for model in models_to_load:
                # Handle measure_suffix: insert before '_raw' suffix
                # e.g., 'perf_raw' + '_speech_only_eval' -> 'perf_speech_only_eval_raw'
                if resp_label.endswith('_raw'):
                    measure_label = resp_label[:-4] + self.parent.measure_suffix + '_raw'
                else:
                    measure_label = resp_label + self.parent.measure_suffix

                # Try both filename orderings (new: mask_chunklen, old: chunklen_mask)
                filepath_new = os.path.join(self.parent.enc_dir, response_folder,
                                           f"{subject}_encoding_model-{model}_smoothingfwhm-{self.parent.smoothing_fwhm}_mask-{self.parent.mask_name}_chunklen-{self.parent.chunklen}_measure-{measure_label}.nii.gz")
                filepath_old = os.path.join(self.parent.enc_dir, response_folder,
                                           f"{subject}_encoding_model-{model}_smoothingfwhm-{self.parent.smoothing_fwhm}_chunklen-{self.parent.chunklen}_mask-{self.parent.mask_name}_measure-{measure_label}.nii.gz")

                if os.path.exists(filepath_new):
                    files[(model, subject)] = nibabel.load(filepath_new)
                elif os.path.exists(filepath_old):
                    files[(model, subject)] = nibabel.load(filepath_old)
                else:
                    raise FileNotFoundError(f"Could not find file with either ordering:\n  New: {filepath_new}\n  Old: {filepath_old}")

                if response_label == 'ind_product_measure':
                    # For performance file: insert measure_suffix before '_raw'
                    perf_measure_label = 'perf' + self.parent.measure_suffix + '_raw'
                    perf_filepath_new = os.path.join(self.parent.enc_dir, 'performance',
                                                    f"{subject}_encoding_model-{model}_smoothingfwhm-{self.parent.smoothing_fwhm}_mask-{self.parent.mask_name}_chunklen-{self.parent.chunklen}_measure-{perf_measure_label}.nii.gz")
                    perf_filepath_old = os.path.join(self.parent.enc_dir, 'performance',
                                                    f"{subject}_encoding_model-{model}_smoothingfwhm-{self.parent.smoothing_fwhm}_chunklen-{self.parent.chunklen}_mask-{self.parent.mask_name}_measure-{perf_measure_label}.nii.gz")

                    if os.path.exists(perf_filepath_new):
                        files[(model, subject, 'performance')] = nibabel.load(perf_filepath_new)
                    elif os.path.exists(perf_filepath_old):
                        files[(model, subject, 'performance')] = nibabel.load(perf_filepath_old)
                    else:
                        raise FileNotFoundError(f"Could not find performance file with either ordering:\n  New: {perf_filepath_new}\n  Old: {perf_filepath_old}")

                if response_label == 'features_preferred_delay':
                    # For ind_product_measure file: insert measure_suffix before '_raw'
                    prod_measure_label = 'ind_product_measure' + self.parent.measure_suffix + '_raw'
                    prod_filepath_new = os.path.join(self.parent.enc_dir, 'ind_product_measure',
                                                    f"{subject}_encoding_model-{model}_smoothingfwhm-{self.parent.smoothing_fwhm}_mask-{self.parent.mask_name}_chunklen-{self.parent.chunklen}_measure-{prod_measure_label}.nii.gz")
                    prod_filepath_old = os.path.join(self.parent.enc_dir, 'ind_product_measure',
                                                    f"{subject}_encoding_model-{model}_smoothingfwhm-{self.parent.smoothing_fwhm}_chunklen-{self.parent.chunklen}_mask-{self.parent.mask_name}_measure-{prod_measure_label}.nii.gz")

                    if os.path.exists(prod_filepath_new):
                        files[(model, subject, 'ind_product_measure')] = nibabel.load(prod_filepath_new)
                    elif os.path.exists(prod_filepath_old):
                        files[(model, subject, 'ind_product_measure')] = nibabel.load(prod_filepath_old)
                    else:
                        raise FileNotFoundError(f"Could not find product measure file with either ordering:\n  New: {prod_filepath_new}\n  Old: {prod_filepath_old}")

        return files

    def _get_special_models(self, response_label, feature_names):
        """Helper to get list of models for added/unique variance."""
        models = []
        if response_label == 'added_variance':
            for feature in feature_names:
                base, subtract = helpers.get_added_variance_models(feature)
                models.extend([base] + subtract)
        elif response_label == 'unique_variance':
            for feature in feature_names:
                base, subtract = helpers.get_unique_variance_models(feature)
                models.extend([base, subtract])
        return list(set(models))
    def build_glm_params_list(self, config):
        """
        Build list of parameter combinations to parallelize.
        """
        params = []
        models = config['models']
        feature_names = config['feature_names']
        print(feature_names)

        for glm_task1 in self.parent.glm_task:
            for glm_task2 in self.parent.glm_task:
                if glm_task1 != glm_task2:
                    run_group_sets = [[(self.parent.all_runs[glm_task1], self.parent.all_runs[glm_task2])]]
                else:
                    # cross-validated splits give independent glm responses; the all-runs localizer gives the
                    # encoding responses, so every participant with this localizer (not only those who also did
                    # the other task) has an all-runs row. Its glm response is not independent and is not plotted.
                    run_group_sets = [self.parent.run_groups[glm_task1],
                                      [(self.parent.all_runs[glm_task1], self.parent.all_runs[glm_task1])]]
                for subject in self.parent.subjects[glm_task1]:
                    for localizer_contrast in self.parent.localizer_contrasts[glm_task1]:
                        for response_contrast in self.parent.response_contrasts[glm_task2]:
                            for hemi in ['left', 'right']:
                                for mask_name in self.parent.localizer_masks[localizer_contrast]:
                                    for feature_name in feature_names:
                                        for model_name in models:
                                            for run_groups in run_group_sets:
                                                params.append((
                                                    subject, glm_task1, glm_task2,
                                                    localizer_contrast, response_contrast,
                                                    run_groups, hemi, feature_name,
                                                    mask_name, model_name
                                                ))
        return params
    def process_voxel_selection(self, subject, mask_name, response_label, feature_name, model, glm_task1,
                             glm_task2, localizer_contrast, response_contrast,
                             run_groups, files, hemi, extraction_threshold=0):
        glm_responses = []
        enc_responses = []
        cross_subject_responses = []
        num_voxels = []
        prop_voxels = []
        localizer_runs = []
        response_runs = []

        all_glm_voxelwise = []
        all_enc_voxelwise = []
        all_cross_subject_voxelwise = []

        for localize_run, response_run in run_groups:
            try:
                z_scores_img = files[(glm_task1, localize_run, localizer_contrast, subject)]
                localizer_label = f"{subject}_smoothingfwhm-{self.parent.smoothing_fwhm}_mask-{mask_name}_glm_loc-{localizer_contrast}_run-{localize_run}"
                localizer_mask = self.parent.mm.get_subject_mask_glm(localizer_label, mask_name, z_scores_img)

                glm_weights_img = files[(glm_task2, response_run, response_contrast, subject)]
                glm_weights = glm_weights_img.get_fdata()
            except Exception as e:
                #print(e)
                continue

            # Load cross-subject data only if available
            cross_subject = files[('cross-subject encoding', subject)].get_fdata() if ('cross-subject encoding', subject) in files else None

            if response_label == 'added_variance':
                base_model, subtract_models = helpers.get_added_variance_models(feature_name)
                all_subtract_models = [files[(model_name, subject)].get_fdata() for model_name in subtract_models]
                subtract_model_data = np.max(all_subtract_models, axis=0)
                base_model_data = np.clip(files[(base_model, subject)].get_fdata(), a_min=0, a_max=None)
                subtract_model_data = np.clip(subtract_model_data, a_min=0, a_max=None)
                enc_response = base_model_data - subtract_model_data

            elif response_label == 'unique_variance':
                base_model, subtract_model = helpers.get_unique_variance_models(feature_name)
                base_model_data = np.clip(files[(base_model, subject)].get_fdata(), a_min=0, a_max=None)
                subtract_model_data = np.clip(files[(subtract_model, subject)].get_fdata(), a_min=0, a_max=None)
                enc_response = base_model_data - subtract_model_data

            else:
                enc_response = files[(model, subject)].get_fdata()
                if response_label == 'ind_product_measure':
                    constraining_enc_response = files[(model, subject, 'performance')].get_fdata()
                    if self.parent.mask_by_performance:
                        enc_response[:, constraining_enc_response < 0] = 0
                    if self.parent.clip_negative_product_measure:
                        enc_response = np.clip(enc_response, a_min=0, a_max=None)
                if response_label == 'performance':
                    if self.parent.clip_negative_performance:
                        enc_response = np.clip(enc_response, a_min=0, a_max=None)

            mask_name_hemi = f"{hemi}-{mask_name}"
            hemi_mask = self.parent.mm.get_resampled_mask(mask_name_hemi, z_scores_img)

            if response_label in ['ind_feature_performance', 'ind_product_measure', 'features_preferred_delay']:
                if response_label == 'features_preferred_delay':
                    product_measure_response = files[(model, subject, 'ind_product_measure')].get_fdata()
                if feature_name in self.parent.combined_features:
                    indices = [self.parent.get_feature_index(sub_feature_name) for sub_feature_name in self.parent.model_features_dict[feature_name]]
                    if response_label == 'features_preferred_delay':
                        enc_response = enc_response / len(self.parent.model_features_dict[feature_name])
                        product_measure_response = product_measure_response[indices].sum(axis=0, keepdims=0)
                        hemi_mask[product_measure_response < extraction_threshold] = 0
                    enc_response = enc_response[indices].sum(axis=0)
                else:
                    indices = [self.parent.get_feature_index(feature_name)]
                    if response_label == 'features_preferred_delay':
                        hemi_mask[product_measure_response[indices][0] < extraction_threshold] = 0
                    enc_response = enc_response[indices][0]

            combined_mask = (hemi_mask == 1) & localizer_mask

            glm_data = glm_weights[combined_mask]
            enc_data = enc_response[combined_mask]
            cross_subject_data = cross_subject[combined_mask] if cross_subject is not None else np.array([])

            if self.parent.scale_by == 'total_variance':
                enc_data = np.clip(enc_data, a_min=0, a_max=None)
                enc_data = enc_data / enc_data.sum(axis=0, keepdims=True)

            #only save the first loc run, resp run combo so we have responses for all regions
            if ((localize_run == self.parent.all_runs[glm_task1])|(localize_run == self.parent.run_groups[glm_task1][1][0])):
                all_glm_voxelwise.append(glm_data)
                all_enc_voxelwise.append(enc_data)
                all_cross_subject_voxelwise.append(cross_subject_data)

            num_voxel = len(glm_data)
            glm_responses.append(np.nanmean(glm_data) if not np.isnan(glm_data).all() else np.nan)
            enc_responses.append(np.nanmean(enc_data) if not np.isnan(enc_data).all() else np.nan)
            cross_subject_responses.append(np.nanmean(cross_subject_data) if len(cross_subject_data) > 0 and not np.isnan(cross_subject_data).all() else np.nan)
            num_voxels.append(num_voxel)
            prop_voxels.append(num_voxel / self.parent.n_voxels_all[self.parent.perc_top_voxels][mask_name])
            localizer_runs.append(str(localize_run))
            response_runs.append(str(response_run))

        def safe_mean(x):
            x = np.asarray(x)
            return np.nan if x.size == 0 else np.nanmean(x)

        return {'subject': subject,
                'glm_task_localizer': glm_task1,
                'glm_task_response': glm_task2,
                'localizer_contrast': localizer_contrast,
                'glm_response_contrast': response_contrast,
                'hemisphere': hemi,
                'mask': mask_name,
                'model': model,
                'enc_feature_name': feature_name,
                'glm_weight': safe_mean(glm_responses),
                'encoding_response': safe_mean(enc_responses),
                'cross-subject encoding': safe_mean(cross_subject_responses),
                'num_voxels': np.mean(num_voxels),
                'proportion_voxels': np.mean(prop_voxels),
                'averaged_localizer_runs': ",".join(localizer_runs),
                'averaged_response_runs': ",".join(response_runs),
                'glm_voxelwise': np.concatenate(all_glm_voxelwise) if all_glm_voxelwise else np.array([]),
                'enc_voxelwise': np.concatenate(all_enc_voxelwise) if all_enc_voxelwise else np.array([]),
                'cross_subject_voxelwise': np.concatenate(all_cross_subject_voxelwise) if all_cross_subject_voxelwise else np.array([]),
            }
    def save_voxel_selection_results(self, results_list, output_basename, response_label):
        """
        Save both summary (mean) and voxelwise results.
        """
        summary_rows = []
        voxelwise_rows = []

        for res in results_list:
            summary_row = {k: v for k, v in res.items() if not k.endswith('_voxelwise')}
            voxelwise_row = {k: v for k, v in res.items() }
            summary_rows.append(summary_row)
            voxelwise_rows.append(voxelwise_row)

        summary_df = pd.DataFrame(summary_rows)
        if(response_label=='performance'):
            add_this = 'cross-subject encoding'
            temp_results = summary_df.copy()
            temp_results['model'] = [add_this for item in summary_df['model']]
            temp_results['encoding_response'] = temp_results[add_this]

            summary_df = pd.concat([summary_df,temp_results],ignore_index=True)
        
        #take out subjects with no data
        summary_df = summary_df[summary_df['num_voxels']!=np.nan]
        summary_path = self.parent.save_data(summary_df, os.path.join(self.parent.out_dir, output_basename + '_summary'), save_type='csv')
        voxelwise_path = self.parent.save_data(voxelwise_rows, os.path.join(self.parent.out_dir, output_basename + '_voxelwise'), save_type='pkl')

        return summary_path, voxelwise_path

    def collect_glm_voxel_selection(self, response_label, filepath_tag='', extraction_threshold=0, load=False, include_cross_subject=False):
        """
        Full pipeline to collect GLM voxel selection results.
        """
        config = self.build_config(response_label)
        files = self.load_files(config, response_label, load=load, include_cross_subject=include_cross_subject)
        if load:
            return None  # You can add loading-from-csv here later if you want

        params_list = self.build_glm_params_list(config)

        results = Parallel(n_jobs=-1)(
            delayed(self.process_voxel_selection)(
                subject, mask_name, response_label, feature_name, model_name,
                glm_task_localizer, glm_task_response,
                localizer_contrast, response_contrast,
                run_groups, files, hemi, extraction_threshold
            ) for (subject, glm_task_localizer, glm_task_response,
                   localizer_contrast, response_contrast,
                   run_groups, hemi, feature_name,
                   mask_name, model_name) in tqdm(params_list)
        )
        
        base_filename = f"{self.parent.sid}{self.parent.enc_file_label}_model-{self.parent.model}_perc_top_voxels-{self.parent.perc_top_voxels}_glm_localizer_{response_label}_{filepath_tag}"
        self.save_voxel_selection_results(results, output_basename=base_filename, response_label=response_label)

        return results
    def collect_encoding_voxel_selection(self, selection_label, response_label,
                                      selection_model, localizers_to_plot, filepath_tag='',
                                      extraction_threshold=0, load=False):
        """
        Full pipeline to collect voxel selection results for encoding model-based selection.
        """

        config = self.build_config(response_label)
        files = self.load_encoding_files(config, selection_label, selection_model, localizers_to_plot, load=load)

        if load:
            return None  # You could also add loading from CSV here later if you want

        params_list = self.build_encoding_params_list(config, localizers_to_plot, selection_model)

        results = Parallel(n_jobs=-1)(
            delayed(self.process_encoding_voxel_selection)(
                subject, mask_name, feature_name_loc, feature_name_resp,
                glm_task, response_contrast,
                run_groups, files, hemi, selection_model,
                selection_label, extraction_threshold
            ) for (subject, glm_task, response_contrast,
                run_groups, hemi, feature_name_loc,
                feature_name_resp, mask_name) in tqdm(params_list)
        )

        base_filename = (f"{self.parent.sid}{self.parent.enc_file_label}_model-{self.parent.model}"
                        f"_measure-{config['response_label']}_perc_top_voxels-{self.parent.perc_top_voxels}"
                        f"_encoding_localizer_{filepath_tag}")

        self.save_voxel_selection_results(results, output_basename=base_filename)

        return results
    
class MaskManager:
    def __init__(self, parent):
        """
        MaskManager handles mask loading, resampling, and caching.
        'parent' is the main SecondLevelIndividual object.
        """
        self.parent = parent
        self.cached_masks = {}           # (mask_name) -> mask array
        self.cached_subject_masks = {}   # (localizer_label) -> mask array
    def get_resampled_mask(self, mask_name, reference_img):
        """Resample the base mask to match the reference image's shape and affine."""
        if mask_name not in self.cached_masks:
            base_mask_img = helpers.load_mask(self.parent, mask_name)
            resampled_img = nilearn.image.resample_img(
                base_mask_img,
                target_affine=reference_img.affine,
                target_shape=reference_img.shape,
                interpolation='nearest'
            )
            self.cached_masks[mask_name] = resampled_img.get_fdata()
        return self.cached_masks[mask_name]

    def get_subject_mask_glm(self, localizer_label, mask_name, zscore_img):
        """
        Get the top-N% voxel mask based on a GLM localizer z-score image.
        """
        if localizer_label not in self.cached_subject_masks:
            localizer_mask = self.get_resampled_mask(mask_name, zscore_img)
            n_voxels = self.parent.n_voxels_all[self.parent.perc_top_voxels][mask_name]
            zscores = zscore_img.get_fdata()

            # Threshold z-scores within the mask
            threshold, _ = helpers.get_top_n(zscores[localizer_mask == 1], int(n_voxels))
            subject_mask = (localizer_mask == 1) & (zscores > threshold)

            # Save to cache
            self.cached_subject_masks[localizer_label] = subject_mask

            # Save mask file (optional)
            img = nibabel.Nifti1Image(subject_mask.astype('uint8'), zscore_img.affine)
            filepath = os.path.join(self.parent.out_dir, 'localizer_masks', f"{localizer_label}.nii.gz")
            nibabel.save(img, filepath)

        return self.cached_subject_masks[localizer_label]

    def get_subject_mask_encoding(self, localizer_label, mask_name, feature_img, subject, selection_model, feature, data_type='ind_product_measure'):
        """
        Get the top-N% voxel mask based on an encoding performance map (e.g., product measure).
        """
        if (localizer_label, data_type) not in self.cached_subject_masks:
            brain_data = feature_img.get_fdata()

            if data_type != 'performance':
                # Sum features if necessary
                if feature in self.parent.combined_features:
                    indices = [self.parent.get_feature_index(sub_feature, selection_model=selection_model)
                               for sub_feature in self.parent.model_features_dict[feature]]
                    brain_data = brain_data[indices].sum(axis=0)
                else:
                    index = self.parent.get_feature_index( feature, selection_model=selection_model)
                    brain_data = brain_data[index]

            brain_data_img = nibabel.Nifti1Image(brain_data, feature_img.affine)
            localizer_mask = self.get_resampled_mask(mask_name, brain_data_img)
            n_voxels = self.parent.n_voxels_all[self.parent.perc_top_voxels][mask_name]

            # Threshold brain_data within mask
            threshold, _ = helpers.get_top_n(brain_data[localizer_mask == 1], int(n_voxels))
            subject_mask = (localizer_mask == 1) & (brain_data > threshold)

            # Save to cache
            self.cached_subject_masks[(localizer_label, data_type)] = subject_mask

            # Save mask file (optional)
            img = nibabel.Nifti1Image(subject_mask.astype('uint8'), brain_data_img.affine)
            filepath = os.path.join(self.parent.out_dir, 'localizer_masks', f"{localizer_label}.nii.gz")
            nibabel.save(img, filepath)

        return self.cached_subject_masks[(localizer_label, data_type)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model','-model',type=str,default='full')
    parser.add_argument('--task','-task',type=str,default='sherlock')
    parser.add_argument('--mask','-mask',type=str, default='ISC') #the mask that contains all masks of interest (overarching mask )
    parser.add_argument('--perc-top-voxels','-perc-top-voxels',type=int,default=None)
    parser.add_argument('--space','-space',type=str,default='MNI152NLin2009cAsym')
    parser.add_argument('--smoothing-fwhm','-smoothing-fwhm',type=float,default=0.0)
    parser.add_argument('--chunklen','-chunklen',type=int,default=17)
    parser.add_argument('--population','-population',type=str,default='NT')

    parser.add_argument('--dir', '-dir', type=str,
                        default=helpers.PROJECT_DIR)
    parser.add_argument('--out_dir', '-output', type=str,
                        default=os.path.join(helpers.PROJECT_DIR, 'analysis'))
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default=os.path.join(helpers.PROJECT_DIR, 'figures'))
    args = parser.parse_args()
    SecondLevelIndividual(args).run()

if __name__ == '__main__':
    main()
    
