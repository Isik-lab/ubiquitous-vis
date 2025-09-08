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

    def __init__(self, args):
        self.process = 'SecondLevelIndividual'
        self.dir = args.dir
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
        self.localizer_contrasts = {'SIpointlights':{'interact&no_interact','interact-no_interact'},'language':{'intact-degraded','degraded-intact'}}
        self.MT = ['MT']
        self.ISC = ['ISC']
        self.STS = ['pSTS','aSTS']
        self.language =['aTemp','pTemp','frontal']
        self.language_ROI_names = ['aTemp','pTemp','frontal']
        self.localizer_masks = {'interact&no_interact':self.MT,'interact-no_interact':self.STS,'intact-degraded':self.language, 'degraded-intact':self.STS,
                                'motion pointlights':self.MT,'SI pointlights':self.STS, 'language':self.language, 'DMN': self.STS,
                                'motion':self.MT,'num_agents':self.STS, 'alexnet':self.ISC,
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
        Path(f'{self.out_dir}/{"all_significant_voxels"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.out_dir}/{"preference_map"}').mkdir(exist_ok=True, parents=True)

        Path(f'{self.figure_dir}/{"localizer_masks"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"glm_zscores"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"scatter"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"localizer_overlap_maps"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"difference_maps"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"response_similarity"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"preference_map"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"map"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"overlap"}').mkdir(exist_ok=True, parents=True)
        Path(f'{self.figure_dir}/{"features_preferred_delay"}').mkdir(exist_ok=True, parents=True)

        self.enc_file_label = '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen) 
        self.glm_file_label = '_smoothingfwhm-'+str(self.smoothing_fwhm)

        if(self.mask_name!=None):
            self.enc_file_label = self.enc_file_label + '_mask-'+self.mask_name
            # self.glm_file_label = self.glm_file_label + '_mask-' +self.mask_name
            self.mask = helpers.load_mask(self,self.mask_name)

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
            'language':'HuBERT+word2vec+sBERT',
            'vision_transformers':'SimCLR+motion',
            'language_transformers':'HuBERT+word2vec+GPT2',
            'alexnet': 'Vision Model',
            'motion':'Motion Model',
            'hubert':'Speech Model',
            'word2vec':'Word Model',
            'sbert':'Sentence Model'
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

    def get_feature_index(self, feature, weight=False, selection_model=''):
        if(selection_model==''):
            file_label = 'encoding_model-'+self.model+self.enc_file_label
        else:
            file_label = 'encoding_model-'+selection_model+self.enc_file_label

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
    
    def generate_preference_maps(self,load=False,measure='ind_product_measure',restricted=False,threshold=0,features=[],color_dict=None,file_tag='',views=['lateral','ventral']):
        fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')
        enc_file_label = '_encoding_model-'+self.model + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
        if(self.mask_name!=None):
            enc_file_label = enc_file_label + '_mask-'+self.mask_name

        for subject_ind,subject in enumerate(self.subjects['sherlock']):
            #load the feature product measure data
            if(measure=='ind_product_measure'):
                enc_response_path = os.path.join(self.enc_dir,'ind_product_measure',subject+enc_file_label+'_measure-ind_product_measure_raw.nii.gz')
                enc_response_img = nibabel.load(enc_response_path)
                enc_response_data = enc_response_img.get_fdata()
                
                enc_performance_path = os.path.join(self.enc_dir,'performance',subject+enc_file_label+'_measure-perf_raw.nii.gz')
                enc_performance_img = nibabel.load(enc_performance_path)
                enc_performance_data = enc_performance_img.get_fdata()
            elif(measure=='structured_variance'):
                data = [] #first entry is the full model (assumes that self.model is the full model)
                for feature in features:
                    enc_file_label = '_encoding_model-'+feature + '_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                    if(self.mask_name!=None):
                        enc_file_label = enc_file_label + '_mask-'+self.mask_name
                    enc_response_path = os.path.join(self.enc_dir,'performance',subject+enc_file_label+'_measure-perf_raw.nii.gz')
                    enc_response_img = nibabel.load(enc_response_path)
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
                    filename = filename + '_loc-' + localizer_contrast
                except Exception as e:
                    print(e)
            plotting_helpers.plot_preference_surf(preference1_map_surf,os.path.join(self.figure_dir,"preference_map",file_label+'_measure-'+measure+'_preference1_map_'+file_tag),ROI_niis=ROI_niis,ROIs=ROIs,ROI_colors=ROI_colors,color_dict=color_dict,cmap=surf_cmap,threshold=0.001,title=title,vmax = len(features),views=views)
    
    def glm_voxel_selection(self, load=False, plot_ind=True, plot=True, plot_stacked=True,
                         response_label='ind_feature_performance', localizers_to_plot=[],
                         localizer_label_dict={}, plot_noise_ceiling=False, stats_to_do=None,
                         parametric=True, filepath_tag='', extraction_threshold=0,
                         average_posterior_anterior=False, average_left_right=False,
                         figure_tag='', restrict_legend=False,
                         plot_lines=False, label_region=True,
                         legend_below=True):
        """
        For GLM, select voxels and collect responses using VoxelSelectionManager.
        """
        print('Running GLM voxel selection...')

        if not load:
            results = self.vsm.collect_glm_voxel_selection(
                response_label=response_label,
                filepath_tag=filepath_tag,
                extraction_threshold=extraction_threshold
            )
            
        base_filename = f"{self.sid}{self.enc_file_label}_model-{self.model}_perc_top_voxels-{self.perc_top_voxels}_glm_localizer_{response_label}_{filepath_tag}"
        summary_path = os.path.join(self.out_dir, base_filename + '_summary.csv')
        results = pd.read_csv(summary_path)
        # Drop rows where 'num_voxels' is missing (NaN)
        results = results.dropna(subset=['num_voxels'])
        # Optional: reset index
        results = results.reset_index(drop=True)
        pkl_path = os.path.join(self.out_dir, base_filename +'_voxelwise.pkl')


        # --- Plotting ---
        if plot:
            localizer_contrasts = [loc + ' (' + localizer_label_dict[loc] + ')' for loc in localizers_to_plot]

            width_ratios = [1 if (average_posterior_anterior) else len(self.localizer_masks[loc]) for loc in localizers_to_plot]
            if ('language' in localizers_to_plot):
                if('frontal' in self.localizer_masks['language']):
                    width_ratios[2]=2
            
            self.plot_glm_response(results, average_posterior_anterior, column_group='localizer_contrast_label',
                                col_order=localizer_contrasts, label_dict=localizer_label_dict, width_ratios=width_ratios,
                                loc_name='localizer_contrast', model_label='model', selection_model='',
                                file_label='glm_localizer_glm_response', filepath_tag=filepath_tag,label_region=label_region,
                                legend_below=legend_below)

            print(results['averaged_localizer_runs'])
            results['cross_validated'] = [len(runs.split(',')) > 1 for runs in results['averaged_localizer_runs']]
            results = results[~results['cross_validated']].copy()

            hue = 'enc_feature_name'
            if response_label == 'performance':
                results['encoding_response'][results['encoding_response'] < 0] = 0
                hue = 'model'
            self.plot_enc_response(results, average_posterior_anterior, average_left_right, column_group='localizer_contrast_label',
                                    col_order=localizer_contrasts, label_dict=localizer_label_dict, width_ratios=width_ratios,
                                    response_label=response_label, hue=hue, loc_name='localizer_contrast',
                                    model_label='model', file_label='glm_localizer_enc_response-'+response_label, plot_stacked=plot_stacked,
                                    plot_noise_ceiling=plot_noise_ceiling, stats_to_do=stats_to_do,
                                    parametric=parametric, filepath_tag=filepath_tag, figure_tag=figure_tag,
                                    restrict_legend=restrict_legend, label_region=label_region)

        return summary_path,pkl_path
        
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

            average_these = ['glm_weight', 'proportion_voxels', 'encoding_response',
                            'ISC', 'cross-subject encoding (MT+STS+language)',
                            'cross-subject encoding (ISC)', 'num_voxels']
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

            average_these = ['glm_weight', 'encoding_response', 'proportion_voxels',
                            'ISC', 'cross-subject encoding (MT+STS+language)', 'cross-subject encoding (ISC)']
            columns = [col for col in results.columns if col not in average_these]

            # Replace all-NaN index columns with 0 to avoid pivot issues
            for col in columns:
                if results[col].isna().all():
                    results[col] = 0

            results = pd.pivot_table(data=results, values=average_these, index=columns, aggfunc='mean').reset_index()

        if average_left_right:
            results['hemisphere'] = 'both' #replace all hemisphere info with both
            average_these = ['glm_weight', 'encoding_response', 'proportion_voxels',
                            'ISC', 'cross-subject encoding (MT+STS+language)', 'cross-subject encoding (ISC)']
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
        legend_below=True
    ):
        # 1. Prepare Data
        df = self.prepare_glm_dataframe(
            results=results,
            average_posterior_anterior=average_posterior_anterior,
            loc_name=loc_name,
            label_dict=label_dict
        )

        # 2. Set plot params
        params = {
            'x': 'hemi_mask',
            'y': 'glm_weight',
            'hue': 'Condition',
            'hue_order': ['interacting pointlights', 'non-interacting pointlights', 'intact speech', 'degraded speech'],
            'palette': {'interacting pointlights':self.colors_dict['social'],'non-interacting pointlights':self.colors_dict['non-social'], 'intact speech':self.colors_dict['sbert'],'degraded speech':self.colors_dict['hubert']}#'plasma'
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
            aspect=1.2
        )

        # 4. Define regions
        MT = self.MT
        ISC = self.ISC
        STS = ['STS'] if average_posterior_anterior else self.STS
        language = ['temporal','frontal'] if average_posterior_anterior else self.language_ROI_names

        region_sets = {'point': point_fig, 'bar': bar_fig}

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
                    comparison_pairs = [
                        ('interacting pointlights', 'non-interacting pointlights'),
                        ('intact speech', 'degraded speech')
                    ]

                    pairs, pvals = stats_helpers.compute_pairwise_stats(
                        data=panel_df,
                        masks=masks,
                        condition_var='Condition',
                        value_var='glm_weight',
                        parametric=parametric,
                        comparison_pairs=comparison_pairs 
                    )

                    plotting_helpers.apply_stat_annotations(
                        ax=ax,
                        pairs=pairs,
                        pvals=pvals,
                        data=panel_df,
                        params=params
                    )

                    # 6. Cosmetic
                    ax.set_xticklabels([f"{t.get_text().split(' ')[1]}\n{t.get_text().split(' ')[0]}" for t in ax.get_xticklabels()])
                    ax.set_title(localizer_label)
                    # for side, xpos in zip(['left', 'right'], [0.25, 0.75]):
                    #     ax.text(xpos, -0.17, side, transform=ax.transAxes, ha='center', va='top',fontsize=20)
                    
                    # Draw a rounded rectangle with text
                    if(label_region):
                        region_name = ax.get_title().split('(')[0] +'regions'
                        label_text = region_name
                        text_x = 0.5
                        text_y = 1.08
                    
                        # Add rectangle and text
                        bbox = dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='none', linewidth=1.5)
                        ax.text(text_x, text_y, label_text, fontsize=20, fontweight='bold',
                                ha='center', va='center', transform=ax.transAxes, bbox=bbox)

            fig.set_axis_labels("", "beta weight")
            fig.set_titles("")
            fig.savefig(
                os.path.join(
                    self.figure_dir,
                    f"{self.sid}{self.enc_file_label}_model-{selection_model}_{file_label}_perc_top_voxels-{self.perc_top_voxels}_glm_response_{label}.png"
                ),
                bbox_inches='tight',
                dpi=300
            )
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
            csv_path = os.path.join(self.out_dir, f"{self.sid}{self.enc_file_label}_model-{self.model}_{file_label}_perc_top_voxels-{self.perc_top_voxels}_glm_response_{label}_{filepath_tag}_social_interaction_selectivity.csv")
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
            csv_path = os.path.join(self.out_dir, f"{self.sid}{self.enc_file_label}_model-{self.model}_{file_label}_perc_top_voxels-{self.perc_top_voxels}_glm_response_{label}_{filepath_tag}_language_selectivity.csv")
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
                csv_path = os.path.join(self.out_dir, f"{self.sid}{self.enc_file_label}_model-{self.model}_{file_label}_perc_top_voxels-{self.perc_top_voxels}_glm_response_{label}_{filepath_tag}_social_interaction_selectivity_in_p_Vs_a.csv")
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
                csv_path = os.path.join(self.out_dir, f"{self.sid}{self.enc_file_label}_model-{self.model}_{file_label}_perc_top_voxels-{self.perc_top_voxels}_glm_response_{label}_{filepath_tag}_language_selectivity_in_p_Vs_a.csv")
                fit.to_csv(csv_path)
            
            # Plot hemispheric distribution of selective voxels
            plotting_helpers.plot_hemispheric_distribution(
                self=self,
                results=df,
                column_group=column_group,
                col_order=col_order,
                loc_name=loc_name,
                hue_key='Condition',
                value_key='proportion_voxels',
                file_tag=f"glm_{file_label}",
                restrict_to_feature='interacting pointlights'  # or None to show all
            )
            
            region_stats_df = stats_helpers.compare_hemispheric_distribution(
                data=df,
                region_column='hemi_mask',
                hemisphere_column='hemisphere',
                value_column='proportion_voxels',
                subject_column='subject'
            )
            csv_path = os.path.join(self.out_dir, f"{self.sid}{self.enc_file_label}_model-{self.model}_{file_label}_perc_top_voxels-{self.perc_top_voxels}_glm_response_{label}_{filepath_tag}_ROI_hemispheric_distribution.csv")
            region_stats_df.to_csv(csv_path)
            
            plotting_helpers.plot_hemispheric_distribution_correlations(
                self=self,
                results=df,
                file_label=file_label,
                # subset_regions=MT+STS+language
                )  
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
        label_region=True
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

        bar_fig, point_fig = plotting_helpers.plot_bar_and_strip(
            data=temp_df,
            column_group=column_group,
            col_order=col_order,
            width_ratios=width_ratios,
            params=params,
            height=5,
            aspect=1
        )

        region_sets = {'bar': bar_fig, 'point': point_fig}
        MT = self.MT
        ISC = self.ISC
        STS = ['STS'] if average_posterior_anterior else self.STS
        language = ['temporal','frontal'] if average_posterior_anterior else self.language_ROI_names

        if average_left_right:
            hemis = ['both']
        else:
            hemis = ['left','right']
        pvalue_results = []

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
                        pairs, pvals = stats_helpers.run_statistical_comparisons(
                            data=panel_df,
                            masks=masks,
                            mode=stats_to_do,
                            condition_levels=params['hue_order'],
                            condition_var='enc_feature_name',
                            value_var='encoding_response',
                            correction='bonferroni',
                            parametric=parametric
                        )
                        
                        plotting_helpers.apply_stat_annotations(ax, pairs, pvals, panel_df, params)
                        pvalue_results.append(dict(zip(pairs, pvals)))
                    print(ax.get_xticklabels())
                    ax.set_xticklabels([f"{t.get_text().split(' ')[1]}\n{t.get_text().split(' ')[0]}" for t in ax.get_xticklabels()])
                    # for hemi_text, xpos in zip(['left', 'right'], [0.25, 0.75]):
                    #     ax.text(xpos, -0.1, hemi_text, transform=ax.transAxes, ha='center', va='top')
                    
                    # Draw a rounded rectangle with text
                    if(label_region):
                        region_name = title_text.split('(')[0] + 'regions'
                        label_text = region_name
                        text_x = 0.5
                        text_y = 1.1
                    
                        # Add rectangle and text
                        bbox = dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='none', linewidth=1.5)
                        ax.text(text_x, text_y, label_text, fontsize=20, fontweight='bold',
                                ha='center', va='center', transform=ax.transAxes, bbox=bbox)

            fig.fig.subplots_adjust(top=0.85)
            fig.set_axis_labels("", "explained variance $R^2$")
            fig.set_titles("")
            fig.savefig(
                os.path.join(
                    self.figure_dir,
                    f"{self.sid}{self.enc_file_label}_model-{self.model}_{file_label}_perc_top_voxels-{self.perc_top_voxels}_{label}_{filepath_tag}_{figure_tag}.png"
                ),
                bbox_inches='tight',
                dpi=300
            )
            plt.close(fig.fig)

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
                file_suffix=f"{filepath_tag}_{figure_tag}"
            )
            
        # if len(set(temp_df['Feature_Space'])) > 1:
        #     anova_df = plotting_helpers.run_anova_per_region_pair_per_hemisphere(
        #         data=temp_df,
        #         plot_features=[self.labels_dict.get(feature,feature) for feature in self.plot_features]
        #     )
        #     csv_path = os.path.join(self.out_dir, f"{self.sid}{self.enc_file_label}_model-{self.model}_{file_label}_perc_top_voxels-{self.perc_top_voxels}_enc_response-{response_label}_{label}_{filepath_tag}_{figure_tag}_anova_interaction_test.csv")            
        #     anova_df.to_csv(csv_path, index=False)
            
        #     output_path = os.path.join(self.dir,'tables',f"pairwise_anova_{self.model}_{filepath_tag}_{figure_tag}.tex")
        #     plotting_helpers.generate_latex_anova_pairwise_table(anova_df,output_path)
    def generate_probability_map_glm(self,plot=True,glm_task='SIpointlights',vmax=None):
        glm_file_label = '_smoothingfwhm-'+str(self.smoothing_fwhm)
        
        for localizer_contrast in self.localizer_contrasts[glm_task]:
            combined_masks = np.zeros(self.brain_shape)
            for mask in self.localizer_masks[localizer_contrast]:
                for subject in tqdm(self.subjects[glm_task],desc=localizer_contrast): #only include subjects with both languge and SI for now (localizer maps using all runs are not generated for subjects with data from only one experiment)
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
                
            combined_masks = combined_masks/(len(self.subjects[glm_task])*len(self.localizer_masks[localizer_contrast]))
            file_label = self.sid+glm_file_label+'_mask-'+'_'.join(self.localizer_masks[localizer_contrast])#'sub-all_encoding_model-'+self.model
            combined_masks_map_img = nibabel.Nifti1Image(combined_masks,localizer_map_img.affine)
            map_img_filename = self.out_dir + '/localizer_masks/'+file_label+'_glm_loc-'+localizer_contrast+'_probability_map.nii.gz'
            nibabel.save(combined_masks_map_img,map_img_filename)
            if(plot):
                cmap = 'matrix_green'
                map_filename = self.figure_dir + '/localizer_masks/'+file_label+'_glm_loc-'+localizer_contrast+'_probability_map'
                plotting_helpers.plot_surface(nii=combined_masks_map_img,filename=map_filename,views=['lateral'],threshold=0,vmin=0,vmax=vmax,cmap=cmap,colorbar_label='proportion of overlap')

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
        print(label)
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
    def compute_response_similarity_across_subjects(self,load=False,plot=True,selection_type='top_percent',pvalue=None,response='weights',split_hemi=True,axes=[],filepath_tag=''):#or 'predicted_time_series' or 'weights_raw'):
        print('computing response similarity of voxel groups:')
        if(selection_type=='top_percent'):
            suffix = '_binary'
            folder = '/localizer_masks/'
            overlap_folder = '/localizer_overlap_maps/'
        elif(selection_type=='all_significant'):
            suffix = '_sig-'+str(pvalue)
            folder = '/all_significant_voxels/'
            overlap_folder = '/all_significant_overlap_maps/'
            
        localizer_contrasts_social = ['interact-no_interact']#,'interact&no_interact']
        localizer_contrasts_lang = ['intact-degraded']
        localizer_contrasts_motion = ['interact&no_interact']
        localizer_names_social = [(name,'glm-SIpointlights-'+ROI) for name in localizer_contrasts_social for ROI in self.localizer_masks['SI pointlights']]
        localizer_names_lang = [(name,'glm-language-'+ROI) for name in localizer_contrasts_lang for ROI in self.localizer_masks['language']]
        localizer_names_motion = [(name,'glm-SIpointlights-'+ROI) for name in localizer_contrasts_motion for ROI in self.localizer_masks['motion pointlights']]
        names = localizer_names_motion+localizer_names_social+localizer_names_lang

        if(split_hemi):
            selected_features = [[(name,name_model_type),(hemi)] for (name,name_model_type) in names for hemi in ['left','right']]
        else:
            selected_features = [[(name,name_model_type),('both')] for (name,name_model_type) in names]
        selected_features.sort()
        
        pairwise_subjects = []
        subjects = self.subjects['sherlock']
        for i in range(len(subjects)):
            for j in range(i + 1, len(subjects)):
                pairwise_subjects.append((subjects[i], subjects[j]))

        if(not load):
            ## preload/precompute all necessary files
            files = {}
            for subject in self.subjects['sherlock']:
                if(response=='weights'):
                    data_filepath = self.enc_dir + '/'+response+'/'+subject+'_encoding_model-'+self.model+self.enc_file_label+'_measure-'+response+'_raw.h5'
                    # f = h5py.File(data_filepath,'r+')
                    # data = f['weights'][()]
                    data = data_filepath
                    files[('response',subject)] = data
                elif(response=='str_variance'): #different loading scheme for unique variance
                    unique_variance_models = []
                    for ax in axes:
                        ax_data = []
                        for feature in ax: #get list of all models needed to get the unique variance for these features
                            unique_variance_models = helpers.get_unique_variance_models(feature)
                            base_model = unique_variance_models[0]
                            subtract_model = unique_variance_models[1]
                            base_model_data = files[(base_model,subject)].get_fdata()
                            subtract_model_data = files[(subtract_model,subject)].get_fdata()
                            
                            #clip negative values to 0
                            base_model_data = np.clip(base_model_data,a_min=0,a_max=None)
                            subtract_model_data = np.clip(subtract_model_data,a_min=0,a_max=None)
                            
                            data = base_model_data-subtract_model_data
                            ax_data.append(data)
                        files[('response',subject)] = data
                else:
                    if(response=='ind_product_measure'):
                        data_filepath = self.enc_dir + '/'+response+'/'+subject+'_encoding_model-'+self.model+self.enc_file_label+'_measure-ind_product_measure_raw.nii.gz'
                    else:
                        data_filepath = self.enc_dir + '/'+response+'/'+subject+self.enc_file_label+'_measure-'+response+'.nii.gz'
                    data = nibabel.load(data_filepath,mmap=True).get_fdata()
                    files[('response',subject)] = data
                
                for ((mask_name,mask_name_type),(hemi)) in selected_features:
                    encoding_masks = [(mask,mask_type) for mask,mask_type in [(mask_name,mask_name_type)] if 'encoding' in mask_type]
                    glm_masks = [(mask,mask_type) for mask,mask_type in [(mask_name,mask_name_type)] if 'glm' in mask_type]
                    
                    for mask, mask_type in encoding_masks:
                        ROI = mask_type.split('-')[-1]
                        localizer_file = f"{subject}_encoding_model-{self.model}_smoothingfwhm-{self.smoothing_fwhm}_chunklen-{self.chunklen}_mask-{ROI}_measure-ind_product_measure_raw_enc_feature_loc-{mask}.nii.gz"
                        localizer = nibabel.load(os.path.join(self.out_dir,'localizer_masks',localizer_file)).get_fdata()    
                        files[('localizer',mask,mask_type,subject)] = localizer
                
                    for mask, mask_type in glm_masks:
                        ROI = mask_type.split('-')[-1]
                        task = mask_type.split('-')[1]
                        if subject in self.subjects[task]:
                            localizer_file = f"{subject}_smoothingfwhm-{self.smoothing_fwhm}_mask-{ROI}_glm_loc-{mask}_run-{self.all_runs[task]}.nii.gz"
                            localizer = nibabel.load(os.path.join(self.out_dir,'localizer_masks',localizer_file)).get_fdata()    
                            files[('localizer',mask,mask_type,subject)] = localizer
                        
            def process(subject1,subject2):
                def get_info(subject1,subject2,response):
                    
                    data1 = files[('response',subject1)]
                    data2 = files[('response',subject2)]
                    
                    if(response=='weights'): 
                        #load the data here to preserve memory (it is not preloaded)
                        f = h5py.File(data1,'r+')
                        data1 = f['weights'][()]
                        f = h5py.File(data2,'r+')
                        data2 = f['weights'][()]
                    results = []
                    for ((mask_name,mask_name_type),(hemi)) in selected_features:
                        try:
                            mask_name_type_split = mask_name_type.split('-')
                            name_measure_label = mask_name_type_split[1]
                            mask_ROI = mask_name_type_split[2]

                            mask1_one_hemi = files[('localizer',mask_name,mask_name_type,subject1)] > 0 
                            mask2_one_hemi = files[('localizer',mask_name,mask_name_type,subject2)] > 0
                            if(split_hemi):
                                self.mask = helpers.load_mask(self,hemi+'-'+self.mask_name)
                            else:
                                self.mask = helpers.load_mask(self,self.mask_name)
                            brain_mask = self.mask.get_fdata()#np.reshape(self.mask.get_fdata(),(-1)) #so we need to mask the masks with the brain mask
                            mask1_one_hemi[(brain_mask==0)] = 0
                            mask2_one_hemi[(brain_mask==0)] = 0
                            
                            for axis in axes.keys():
                                features1=axes[axis]
                                if(response =='weights'): #the weights data is already masked (not in brain shape, in voxels x TR shape)
                                    self.mask = helpers.load_mask(self,self.mask_name)
                                    brain_mask = self.mask.get_fdata()#np.reshape(self.mask.get_fdata(),(-1))
                                    mask1 = mask1_one_hemi[(brain_mask)==1]
                                    mask2 = mask2_one_hemi[(brain_mask)==1]

                                    features1_indices = np.concatenate([ self.get_feature_index(feature,weight=response=='weights') for feature in features1 ])
                                else:
                                    mask1 = mask1_one_hemi
                                    mask2 = mask2_one_hemi
                                    features1_indices = [ self.get_feature_index(feature,weight=response=='weights') for feature in features1 ]
                                data1_masked = data1[:, mask1]
                                data2_masked = data2[:, mask2]
                                data1_masked = np.nanmean(data1_masked,axis=1)
                                data2_masked = np.nanmean(data2_masked,axis=1)

                                # features1_indices = [ self.get_feature_index(feature,weight=response=='weights') for feature in features1 ]
                                # features2_indices = [ self.get_feature_index(feature,weight=weight) for feature in features2 ]
                                if len(data1) > 0:# and len(data2) > 0:
                                    corr1_selected_data1 = data1_masked[features1_indices]
                                    corr1_selected_data2 = data2_masked[features1_indices]
                                    
                                    # print('shape',corr1_selected_data1.shape)

                                    nas1 = np.logical_or(np.isnan(corr1_selected_data1), np.isnan(corr1_selected_data2))
                                    
                                    pearsoncorr_results = scipy.stats.pearsonr(corr1_selected_data1[~nas1], corr1_selected_data2[~nas1])# if axis is None, ravel both arrays before computing
                                    pearsoncorr = pearsoncorr_results[0]
                                    pearsonpvalue = pearsoncorr_results[1]
                                    
                                    spearmancorr_results = scipy.stats.spearmanr(corr1_selected_data1[~nas1], corr1_selected_data2[~nas1])# if axis is None, ravel both arrays before computing
                                    spearmancorr = spearmancorr_results[0]
                                    spearmanpvalue = spearmancorr_results[1]
                                else:
                                    pearsoncorr=pearsonpvalue=spearmancorr=spearmanpvalue = np.nan
                                mask_name_label = mask_name+'-'+mask_ROI
                                results.append((subject1,subject2,axis,mask_name_label,name_measure_label,hemi,pearsoncorr,pearsonpvalue,spearmancorr,spearmanpvalue))
                        except Exception as e:
                            pass
                            # print(f"Exception occurred: {e}")
                    #     return (np.nan,) * 11
                    
                    # mask_name = mask_name+'-'+mask_ROI
                    # results = (subject1,subject2,axis,mask_name,name_measure_label,hemi,corr,corr1,corr2,pvalue1,pvalue2)
                    # print(results)
                    return results

                results = get_info(subject1,subject2,response)
                return results

            # for (feature_name1,feature_name2,localizer_contrast1,localizer_contrast2) in selected_features
            # for subject in self.subjects:
            results = Parallel(n_jobs=1)(delayed(process)(subject1,subject2) 
                for subject1,subject2 in tqdm(pairwise_subjects))
                # for ((mask_name,mask_name_type),(hemi)) in selected_features 
                # for axis in axes.keys())
            # results = pd.DataFrame(zip(subjects,localizer_contrasts,num_voxels_glm,proportion_glm,overlap_types),columns=['subject','localizer_contrast','glm_num_voxels','proportion_glm','hemi'])
            results = np.vstack(results)
            results = pd.DataFrame(results,columns = ['subject1','subject2','axis','mask_name','mask_measure_label','hemi','pearsoncorr','pearsonpvalue','spearmancorr','spearmanpvalue'])
            file_label = self.sid+self.enc_file_label+'_model-'+self.model+'_'+response+'_response_similarity_across_subjects'#'sub-all_encoding_model-'+self.model
            #delete all repeat rows and save
            results.to_csv(self.out_dir+'/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_enc_feature_localizer_results_'+filepath_tag+'.csv')
        #for each subject select voxels and average their time series data, then correlate that across hemispheres and across regions
        #average all individual matrices together
        file_label = self.sid+self.enc_file_label+'_model-'+self.model+'_'+response+'_response_similarity_across_subjects'#'sub-all_encoding_model-'+self.model
        results=pd.read_csv(self.out_dir+'/'+file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_enc_feature_localizer_results_'+filepath_tag+'.csv')
        results = results.drop(columns=['pearsonpvalue','spearmanpvalue'])
        results['clean_mask_name'] = [item.split('-')[-1] for item in results['mask_name']]
        
        #averaging over posterior and anterior!!!
        for prev_mask in self.STS:
            results['clean_mask_name'].replace(prev_mask, 'STS', inplace=True) 
        for prev_mask in self.language:
            results['clean_mask_name'].replace(prev_mask, 'temporal', inplace=True)
        #average across the masks 
        average_these = ['spearmancorr','pearsoncorr']
        # Identify columns to use as indices (i.e., not the ones to average)
        columns = [col for col in results.columns if col not in average_these]
        # Replace all-NaN index columns with 0
        for col in columns:
            if results[col].isna().all():
                results[col] = 0
        results = pd.pivot_table(data=results,values=average_these, index = columns, aggfunc='mean').reset_index()
        
        results = results.replace('MT','motion')
        results = results.replace('STS','social interaction')
        results = results.replace('temporal','language')
        
        hue_order = ['interact&no_interact-MT','interact-no_interact-pSTS','interact-no_interact-aSTS','intact-degraded-pTemp','intact-degraded-aTemp']
        # col_order = ['alexnet','motion','word2vec','sbert'] 
        col_order = axes.keys()
        results['DNN'] = results['axis']
        
        fig = sns.catplot(data=results,x='hemi',y='spearmancorr',hue='DNN',col='clean_mask_name',
                    kind='bar',edgecolor="black",linewidth=2,errorbar='se', errcolor="black",errwidth=2,
                    palette=self.colors_dict,hue_order=col_order,col_order=None,height=4,aspect=0.5)
        fig.set_titles("{col_name}")
        fig.set_axis_labels("",'Spearman correlation')
        
        plt.savefig(os.path.join(self.figure_dir,'response_similarity',file_label+'_perc_top_voxels-' + self.perc_top_voxels+'_enc_feature_localizer_results_spearman'+filepath_tag+'.png'),dpi=300)
    def compute_response_similarity(self, load=False, plot=True, selection_type='top_percent', pvalue=None, 
                                 response='weights', split_hemi=True, axes={}, filepath_tag='', average_posterior_anterior=False):

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
                similarity_matrix, _ = plotting_helpers.plot_similarity_matrix(
                    results_df=all_results,
                    names=names,
                    axis=axis,
                    split_hemi=split_hemi,
                    label_dict=self.labels_dict,
                    output_path=os.path.join(self.figure_dir, 'response_similarity',
                        f"{file_label}_perc_top_voxels-{self.perc_top_voxels}-{axis}_similarity_matrix_{filepath_tag}.png"),
                    plot_cbar=True
                )
                fit, results_df = stats_helpers.run_similarity_superregion_LME(all_results, axis,split_hemi=split_hemi,average_posterior_anterior=average_posterior_anterior)
                csv_path = os.path.join(self.out_dir, f"{self.sid}{self.enc_file_label}_model-{self.model}_{file_label}_perc_top_voxels-{self.perc_top_voxels}_{filepath_tag}_{axis}_response_similarity_stats.csv")            
                fit.to_csv(csv_path, index=False)
                
                csv_path = os.path.join(self.out_dir, f"{self.sid}{self.enc_file_label}_model-{self.model}_{file_label}_perc_top_voxels-{self.perc_top_voxels}_{filepath_tag}_{axis}_response_similarity_pairwise_comparisons.csv")            
                results_df.to_csv(csv_path, index=False)
                
                output_path = os.path.join(self.dir, 'tables', f"response_similarity_pairwise_comparisons_{axis}.tex")
                stats_helpers.generate_latex_pairwise_similarity_table_with_hemisphere(results_df, output_path)
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
    def searchlight_analysis(self,feature1_name,feature1,feature2_dict):
        from nilearn.image import new_img_like
        def run_searchlight_sum_vs_layergroups(data, mask_dict, feature1_indices, feature2_group_indices, radius=4):
            """
            Parameters:
                data: 4D array of shape (F, X, Y, Z)
                mask_dict: dict mapping group names to 3D boolean masks (X, Y, Z)
                feature1_indices: list of indices for summing feature 1 (e.g. AlexNet)
                feature2_group_indices: dict of group name -> list of feature 2 indices (e.g. sBERT groups)
                radius: radius of the spherical searchlight

            Returns:
                output_maps: dict of group name -> 3D array (X, Y, Z) of Pearson r correlations
            """
            import numpy as np
            import scipy.stats
            from scipy.ndimage import generate_binary_structure, iterate_structure

            # Transpose to (X, Y, Z, F)
            data = np.transpose(data, (1, 2, 3, 0))
            shape = data.shape[:3]

            # Sum feature 1 (AlexNet)
            feature1_sum = np.sum(data[..., feature1_indices], axis=-1)

            # Define spherical neighborhood
            sphere = iterate_structure(generate_binary_structure(3, 1), radius)

            output_maps = []

            for group_name, group_indices in feature2_group_indices.items():
                mask = mask_dict[group_name]
                r_map = np.full(shape, np.nan)

                for x in range(shape[0]):
                    for y in range(shape[1]):
                        for z in range(shape[2]):
                            if not mask[x, y, z]:
                                continue

                            neighborhood = []
                            for dx in range(-radius, radius + 1):
                                for dy in range(-radius, radius + 1):
                                    for dz in range(-radius, radius + 1):
                                        if not sphere[radius + dx, radius + dy, radius + dz]:
                                            continue
                                        xi, yi, zi = x + dx, y + dy, z + dz
                                        if (0 <= xi < shape[0]) and (0 <= yi < shape[1]) and (0 <= zi < shape[2]) and mask[xi, yi, zi]:
                                            neighborhood.append((xi, yi, zi))

                            if len(neighborhood) < 3:
                                continue

                            indices = tuple(zip(*neighborhood))
                            feature1_values = feature1_sum[indices]

                            voxel_data = data[indices]  # (N, F)
                            feature2_sum = voxel_data[:, group_indices].sum(axis=1)

                            if np.std(feature2_sum) > 0 and np.std(feature1_values) > 0:
                                r, _ = scipy.stats.pearsonr(feature1_values, feature2_sum)
                                r_map[x, y, z] = r

                output_maps.append(r_map)

            return output_maps
        enc_file_label = '_encoding_model-'+self.model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
        if(self.mask_name!=None):
            enc_file_label = enc_file_label + '_mask-'+self.mask_name
        for subject in self.subjects['sherlock']:
            folder = 'ind_product_measure'
            label = 'ind_product_measure'
            filepath = self.enc_dir+'/'+folder+'/'+subject+enc_file_label+'_measure-'+label+'_raw.nii.gz'
            nii = nibabel.load(filepath)
            data = nii.get_fdata()
            
            folder = 'performance'
            label = 'perf'
            filepath = self.enc_dir+'/'+folder+'/'+subject+enc_file_label+'_measure-'+label+'_raw.nii.gz'
            nii_perf = nibabel.load(filepath)
            data_perf = nii_perf.get_fdata()
            
            data[:,data_perf<0]=0
            
            feature1_indices = [self.get_feature_index(layer) for layer in feature1]
            feature2_group_indices = {
                group_name: [self.get_feature_index(layer) for layer in layer_names]
                for group_name, layer_names in feature2_dict.items()
            }        
            # mask = (helpers.load_mask(self,'ISC').get_fdata()==1) #boolean
            mask_dict = {}
            percentile = 0  # top 50% of nonzero voxels
            
            # Transpose for easier indexing: (F, X, Y, Z) -> (X, Y, Z, F)
            data_T = np.transpose(data, (1, 2, 3, 0))

            # Sum of feature1 
            feature1_map = np.sum(data_T[..., feature1_indices], axis=-1)
            feature1_mask = (feature1_map > 0)&(data_perf>0)

            # Now loop over each group in feature2_dict
            for group_name, layer_names in feature2_dict.items():
                group_indices = [self.get_feature_index(layer) for layer in layer_names]
                feature2_map = np.sum(data_T[..., group_indices], axis=-1)
                feature2_mask = (feature2_map > 0)&(data_perf>0)

                # Joint mask: valid only where both feature1 and feature2 have top 50% activation
                joint_mask = np.logical_and(feature1_mask, feature2_mask)

                # Optionally restrict to a base anatomical mask
                anatomical_mask = helpers.load_mask(self, 'ISC').get_fdata() == 1
                joint_mask = np.logical_and(joint_mask, anatomical_mask)

                mask_dict[group_name] = joint_mask
            
            r_maps = run_searchlight_sum_vs_layergroups(data,mask_dict,feature1_indices, feature2_group_indices)
            r_img = new_img_like(nii, np.array(r_maps))
            #saving results
            filename = self.out_dir + '/spatial_correlation/individual_maps/'+subject+enc_file_label+'_measure-voxelwise_correlation_'+feature1_name+'-.nii.gz'
            nibabel.save(r_img,filename)
            #save feature comparisons
            comparison_df = pd.DataFrame({'comparison': list(feature2_dict.keys())})
            csv_filename = self.out_dir + '/spatial_correlation/individual_maps/'+enc_file_label+'_measure-voxelwise_correlation_'+feature1_name+'-.csv'
            comparison_df.to_csv(csv_filename, index=False)

            
            
            for r_map,label in zip(r_maps,feature2_dict.keys()):
                nii = nibabel.Nifti1Image(r_map, nii.affine)
                filename = self.figure_dir + '/spatial_correlation/individual_maps/'+subject+enc_file_label+'_measure-voxelwise_correlation_'+feature1_name+'-'+label
                ROI_niis = []
                cmap = 'blue_neg_yellow_pos'
                vmax=1
                vmin=-1
                plotting_helpers.plot_surface(nii,filename,ROI_niis=ROI_niis,symmetric_cbar=False,cmap=cmap,title='', vmax=vmax, vmin=vmin, colorbar_label='r')#self.colors_dict[localizer])
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
    
    def plot_map(self,feature='',measure='ind_feature_performance',localizers=[],threshold=0,vmin=None,vmax=None,cmap='yellow_hot'):
        fsaverage = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')

        enc_file_label = '_encoding_model-'+self.model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
        if(self.mask_name!=None):
            enc_file_label = enc_file_label + '_mask-'+self.mask_name

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
                    filepath = self.enc_dir+'/'+folder+'/'+subject+enc_file_label+'_measure-'+label+'_raw.nii.gz'
                    nii1 = nibabel.load(filepath)
                    data1 = nii1.get_fdata()
                    data1[data1<0]=0 #clip negatives to 0 (negatives are not meaningful)
                elif(measure=='added_variance'):
                    base_model,subtract_model = helpers.get_added_variance_models(feature)
                    
                    enc_file_label = '_encoding_model-'+subtract_model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                    if(self.mask_name!=None):
                        enc_file_label = enc_file_label + '_mask-'+self.mask_name
                    filepath = self.enc_dir+'/'+folder+'/'+subject+enc_file_label+'_measure-'+label+'_raw.nii.gz'
                    nii1 = nibabel.load(filepath)
                    subtract_model = nii1.get_fdata()
                    subtract_model[subtract_model<0] = 0
                    
                    enc_file_label = '_encoding_model-'+base_model+'_smoothingfwhm-'+str(self.smoothing_fwhm)+'_chunklen-'+str(self.chunklen)
                    if(self.mask_name!=None):
                        enc_file_label = enc_file_label + '_mask-'+self.mask_name
                    filepath = self.enc_dir+'/'+folder+'/'+subject+enc_file_label+'_measure-'+label+'_raw.nii.gz'
                    nii1 = nibabel.load(filepath)
                    base_model = nii1.get_fdata()
                    base_model[base_model<0] = 0
                    
                    data1 = base_model-subtract_model
                else:
                    filepath = self.enc_dir+'/'+folder+'/'+subject+enc_file_label+'_measure-'+label+'_raw.nii.gz'
                    nii1 = nibabel.load(filepath)
                    data1 = nii1.get_fdata()
                    data1[data1<0] = 0 #clip response values to 0
                    if(self.scale_by=='total_variance'):
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
                plotting_helpers.plot_surface(diff_map_nii,filename,ROI_niis=ROI_niis,symmetric_cbar=False,threshold=threshold,cmap=cmap_,vmin=vmin,vmax=vmax,title='', colorbar_label='Explained Variance $R^2$',ROIs=localizers,ROI_colors=['white' for localizer in localizers])#self.colors_dict[localizer])

            # except Exception as e:
            #   print(e)
            #   pass
    def highest_beta_weights_correlate_with_annotated_features_multi(
        self,
        feature_name,
        mask_names,
        top_percent=1.0,
        zscore_inputs=True,
        output_dir=None,
    ):
        from src import helpers
        from scipy.stats import pearsonr, zscore 
        import os
        import numpy as np
        import pandas as pd
        import h5py
        from joblib import Parallel, delayed

        output_dir = output_dir or os.path.join(self.out_dir, 'top_unit_correlations')
        os.makedirs(output_dir, exist_ok=True)

        # Load all annotated features
        annotated_feature_names = helpers.get_models_dict()['annotated']
        all_annos = []
        for anno_name in annotated_feature_names:
            anno_path = os.path.join(self.dir, 'features', self.features_dict[anno_name].lower() + '.csv')
            anno_data = pd.read_csv(anno_path, header=None)
            all_annos.append(anno_data)
        annotations_df = pd.concat(all_annos, axis=1)
        annotations = annotations_df.values

        # Load DNN feature matrix
        feature_csv_path = os.path.join(self.dir, 'features', self.features_dict[feature_name].lower() + '.csv')
        features = np.loadtxt(feature_csv_path, delimiter=',', dtype=np.float32)
        feature_idx = self.get_feature_index(feature_name, weight=True)

        # Cache localizer data
        config = self.vsm.build_config('performance')
        files = self.vsm.load_files(config=config, response_label='performance', load=False)
        main_mask = helpers.load_mask(self, self.mask_name).get_fdata()

        # Mappings
        mask_localizer_dict = {
            'MT':'interact&no_interact',
            'pSTS':'interact-no_interact',
            'aSTS':'interact-no_interact',
            'pTemp':'intact-degraded',
            'aTemp':'intact-degraded',
            'frontal':'intact-degraded'
        }
        mask_glm_task_dict = {
            'MT':'SIpointlights',
            'pSTS':'SIpointlights',
            'aSTS':'SIpointlights',
            'pTemp':'language',
            'aTemp':'language',
            'frontal':'language'
        }

        # Combined mask logic
        merged_masks = {
            "STS": ["pSTS", "aSTS"],
            "temporal": ["pTemp", "aTemp"]
        }

        def compute_subject_mask(subject, mask_name):
            results = []
            unitwise = []
            try:
                mask_group = merged_masks.get(mask_name, [mask_name])

                weights_path = os.path.join(
                    self.enc_dir, 'weights',
                    f"{subject}_encoding_model-{self.model}{self.enc_file_label}_measure-weights_raw.h5"
                )
                with h5py.File(weights_path, 'r') as f:
                    weights = f['weights'][()]  # [n_units, n_voxels]

                for hemi in ['left', 'right']:
                    combined_voxel_mask = []

                    for submask in mask_group:
                        try:
                            zscore_img = files[
                                (mask_glm_task_dict[submask],
                                self.all_runs[mask_glm_task_dict[submask]],
                                mask_localizer_dict[submask],
                                subject)
                            ]
                            localizer_label = (
                                f"{subject}_smoothingfwhm-{self.smoothing_fwhm}_mask-{submask}"
                                f"_glm_loc-{mask_localizer_dict[submask]}_run-{self.all_runs[mask_glm_task_dict[submask]]}"
                            )
                            localizer_mask = self.mm.get_subject_mask_glm(localizer_label, submask, zscore_img)
                            hemi_mask = helpers.load_mask(self, f"{hemi}-{submask}").get_fdata()
                            combined_mask = ((hemi_mask == 1) & (localizer_mask == 1))[main_mask == 1]
                            combined_voxel_mask.append(combined_mask)
                        except Exception:
                            continue

                    if not combined_voxel_mask:
                        continue

                    final_mask = np.logical_or.reduce(combined_voxel_mask)
                    masked_weights = weights[:, final_mask]
                    unit_scores = np.mean(masked_weights, axis=1)
                    unit_scores = unit_scores[feature_idx]
                    top_n = int(np.ceil(len(unit_scores) * top_percent / 100))
                    top_units = np.argsort(unit_scores)[-top_n:]
                    unit_ranks = (-unit_scores).argsort().argsort()
                    unit_timecourses = features[:, top_units]

                    annos = zscore(annotations, axis=0) if zscore_inputs else annotations
                    if zscore_inputs:
                        unit_timecourses = zscore(unit_timecourses, axis=0)

                    for j, ann_name in enumerate(annotated_feature_names):
                        ann = annos[:, j]
                        unit_corrs = [pearsonr(unit_timecourses[:, i], ann)[0] for i in range(top_units.size)]
                        for u, r in zip(top_units, unit_corrs):
                            unitwise.append({
                                "subject": subject,
                                "hemi": hemi,
                                "mask": mask_name,
                                "annotated_feature": ann_name,
                                "unit_index": u,
                                "correlation": r,
                                "rank": unit_ranks[u]
                            })
                        results.append({
                            "subject": subject, "hemi": hemi, "mask": mask_name,
                            "annotated_feature": ann_name,
                            "correlation": np.mean(unit_corrs)
                        })
            except Exception as e:
                pass

            return results, unitwise

        # Run in parallel
        from itertools import product
        all_tasks = product(self.subjects['sherlock'], mask_names)
        parallel = Parallel(n_jobs=1, verbose=5)
        out = parallel(delayed(compute_subject_mask)(subj, mask) for subj, mask in all_tasks)

        subj_results = [item for r, _ in out for item in r]
        unitwise_results = [item for _, u in out for item in u]

        df_subjectwise = pd.DataFrame(subj_results)
        df_unitwise = pd.DataFrame(unitwise_results)

        df_subjectwise.to_csv(os.path.join(output_dir, f"subjectwise_annotation_corrs_{self.model}_{feature_name}.csv"), index=False)
        df_unitwise.to_csv(os.path.join(output_dir, f"subjectwise_unitwise_annotation_corrs_{self.model}_{feature_name}.csv"), index=False)

        return df_subjectwise, df_unitwise

    def get_top_units_per_region(
        self,
        csv_dir,
        model,
        layer,
        annotated_feature,
        method="average",        # "average" or "frequency"
        direction="most",        # "most" or "least"
        top_n=10,
        split_by_hemi=False,
        plot=True,
        return_unit_list=True,
    ):
        """
        Visualize and return AlexNet units most/least correlated with a given feature,
        separately per region (and optionally per hemisphere).

        Parameters:
        - df_unitwise: DataFrame with ["subject", "unit_index", "mask", "hemi", "annotated_feature", "correlation"]
        - annotated_feature: str
        - method: "average" or "frequency"
        - direction: "most" or "least"
        - top_n: int
        - split_by_hemi: bool, whether to analyze each (hemi, region) separately
        - plot: bool
        - return_unit_list: bool

        Returns:
        - DataFrame of selected units
        - Dictionary mapping (region or hemi_region) → list of unit indices (for Lucent)
        """
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        assert method in ["average", "frequency"]
        assert direction in ["most", "least"]

        ascending = direction == "least"

        # Filter to desired feature
        
        df_unitwise = pd.read_csv(f"{csv_dir}/subjectwise_unitwise_annotation_corrs_{model}_{layer}.csv")
        df = df_unitwise.copy() if method == "frequency" else df_unitwise[df_unitwise["annotated_feature"] == annotated_feature].copy()

        group_keys = ["mask"]
        if split_by_hemi:
            group_keys = ["hemi", "mask"]

        unit_lists = {}
        selected_units_all = []

        for group_name, group_df in df.groupby(group_keys):
            group_label = ":".join(group_name) if isinstance(group_name, tuple) else group_name

            if method == "average":
                unit_stats = (
                    group_df.groupby("unit_index")["correlation"]
                    .mean()
                    .sort_values(ascending=ascending)
                    .head(top_n)
                    .reset_index()
                )
                unit_stats["group"] = group_label

                selected_units_all.append(unit_stats)
                unit_lists[group_label] = unit_stats["unit_index"].tolist()

                if plot:
                    plot_df = group_df[group_df["unit_index"].isin(unit_stats["unit_index"])]
                    sns.boxplot(data=plot_df, x="unit_index", y="correlation")
                    plt.title(f"{direction.capitalize()} {top_n} Units - {group_label} (Avg Corr)")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.show()

            elif method == "frequency":
                # Filter to just this region (and hemisphere if split_by_hemi)
                freq_group = group_df.copy()

                # Count unit occurrences across subjects
                freq_counts = (
                    freq_group.groupby("unit_index")["subject"]
                    .nunique()
                    .sort_values(ascending=ascending)
                    .head(top_n)
                    .reset_index()
                    .rename(columns={"subject": "count"})
                )
                freq_counts["group"] = group_label

                selected_units_all.append(freq_counts)
                unit_lists[group_label] = freq_counts["unit_index"].tolist()

                if plot:
                    sns.barplot(data=freq_counts, x="unit_index", y="count", color="dodgerblue")
                    plt.title(f"{direction.capitalize()} {top_n} Units - {group_label} (Freq)")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.show()
                    
                    # Plot average correlation with each annotated feature for selected units
                    selected_units = freq_counts["unit_index"].tolist()
                    feature_corrs = group_df[group_df["unit_index"].isin(selected_units)]

                    # Group by feature and average across units and subjects
                    avg_corrs = (
                        feature_corrs.groupby(["unit_index","annotated_feature"])["correlation"]
                        .mean()
                        .reset_index()
                    )

                    if plot:
                        sns.barplot(
                            data=avg_corrs,
                            estimator=np.mean,
                            x="annotated_feature",
                            y="correlation",
                            errorbar="se", 
                            palette=self.colors_dict
                        )
                        sns.stripplot(
                            data=avg_corrs,
                            x="annotated_feature",
                            y="correlation",
                            color="black",
                            alpha=0.5,
                            jitter=True
                        )
                        plt.xticks(rotation=90)
                        plt.title(f"Avg Correlation of Selected Units - {group_label}")
                        plt.tight_layout()
                        plt.show()


        selected_df = pd.concat(selected_units_all, ignore_index=True)

        if return_unit_list:
            return selected_df, unit_lists
        else:
            return selected_df

    def select_most_frequent_units_per_region(
        self,
        csv_path: str,
        model: str,
        layer: str,
        top_n: int = 10,
        split_by_hemi: bool = False,
        split_posterior_anterior: bool = False,
        plot: bool = True,
        plot_points: bool=False,
        plot_units: list=[]
    ):
        """
        Select and visualize the most frequently selected units across subjects for each region (and optionally hemisphere),
        and plot their average correlation with all annotated features.

        Returns:
            freq_df: DataFrame of unit frequencies with group labels
            unit_lists: dict mapping region or hemi-region → list of top unit indices
        """
        def combine_regions(df, split_by_hemi, split_posterior_anterior):
            if not split_posterior_anterior:
                def relabel(mask):
                    if mask in ['MT']:
                        return 'motion regions'
                    elif mask in ['pSTS','aSTS','STS']:
                        return 'social interaction regions'
                    elif mask in ["pTemp", "aTemp","temporal","frontal"]:
                        return "language regions"
                    else:
                        return mask
                df["mask"] = df["mask"].apply(relabel)

            if split_by_hemi:
                df["group"] = df["hemi"] + " " + df["mask"]
            else:
                df["group"] = df["mask"]
            return df
        df = pd.read_csv(f"{csv_path}/subjectwise_unitwise_annotation_corrs_{model}_{layer}.csv")
        df = combine_regions(df, split_by_hemi=split_by_hemi, split_posterior_anterior=split_posterior_anterior)


        # group_keys = ["mask"]
        # if split_by_hemi:
        #     group_keys = ["hemi", "mask"]

        unit_lists = {}
        selected_units_all = []

        for group_name, group_df in df.groupby("group"):
            group_label = " ".join(group_name) if isinstance(group_name, tuple) else group_name

            # Count frequency of units across subjects
            freq_counts = (
                group_df.groupby("unit_index")["subject"]
                .nunique()
                .sort_values(ascending=False)
                .head(top_n)
                .reset_index()
                .rename(columns={"subject": "count"})
            )
            freq_counts["group"] = group_label
            selected_units_all.append(freq_counts)
            unit_lists[group_label] = freq_counts["unit_index"].tolist()

        freq_df = pd.concat(selected_units_all, ignore_index=True)

        if plot:
            if(split_posterior_anterior):
                col_order = ['left MT', 'left pSTS','left aSTS','left pTemp','left aTemp','left frontal',
                                'right MT', 'right pSTS','right aSTS','right pTemp','right aTemp','right frontal']
            else:
                col_order = ['left MT', 'left STS','left temporal','left frontal',
                            'right MT', 'right STS','right temporal','right frontal']
                col_order = ['left MT','right MT',
                             'left STS','right STS',
                             'left temporal','right temporal',
                             'left frontal','right frontal']
                
            if(split_by_hemi==False):
                col_order=['motion regions','social interaction regions','language regions']#'temporal','frontal']
            g = sns.catplot(
                data=freq_df,
                x="unit_index",
                y="count",
                col="group",
                col_order = col_order,
                # col_wrap=2,
                kind="bar",
                height=4,
                aspect=1.2,
                color="gray", 
                sharex=False,
                sharey=True
            )
            g.set_axis_labels("unit index","count")
            g.set_titles("{col_name}")
            for ax in g.axes.flat:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            # plt.subplots_adjust(top=0.85)
            plt.suptitle(f"Top {top_n} Most Shared Units per Region", fontsize=16)
            os.makedirs(os.path.join(self.figure_dir,'unit_interpretations'),exist_ok=True)
            plt.savefig(os.path.join(self.figure_dir,'unit_interpretations',f"{model}_{layer}_unit_frequencies.png"),dpi=300)


        # Correlations for selected units
        all_selected_units = freq_df["unit_index"].unique()
        
        # corr_df = df[df["unit_index"].isin(all_selected_units)].copy()
        groupwise_unit_avg = []

        # Loop through each region and its selected units
        for group_label, unit_subset in freq_df.groupby("group"):
            top_units = unit_subset["unit_index"].unique()

            # Get correlation data for these units in this group
            subset_df = df[(df["group"] == group_label) & (df["unit_index"].isin(top_units))]
            if(len(plot_units)>0):
                subset_df = subset_df[subset_df['unit_index'].isin(plot_units[group_label])]
            # Compute average correlation per unit and feature
            avg = (
                subset_df.groupby(["unit_index","annotated_feature"])["correlation"]
                .mean()
                .reset_index()
            )
            avg["group"] = group_label  # add group info back
            groupwise_unit_avg.append(avg)

        # Concatenate into a single DataFrame
        unit_avg = pd.concat(groupwise_unit_avg, ignore_index=True)
        
        if plot:
            if(split_posterior_anterior):
                col_order = ['left MT','right MT',
                             'left pSTS','right pSTS',
                             'left aSTS','right aSTS',
                             'left pTemp','right pTemp',
                             'left aTemp','right aTemp',
                             'left frontal','right frontal']
            else:
                col_order = ['right MT',
                             'right STS',
                             'left temporal',
                             'left frontal',]
            if(split_by_hemi==False):
                col_order=['motion regions','social interaction regions','language regions']#'temporal','frontal']
            
            unit_avg.replace(self.labels_dict,inplace=True)

            g = sns.catplot(
                data=unit_avg,
                x="unit_index",
                hue = "annotated_feature",
                hue_order = [self.labels_dict.get(item,item) for item in self.model_features_dict['annotated']],
                y="correlation",
                col="group",
                col_order = col_order,
                col_wrap = 1,#int(len(col_order)/2),
                kind="bar",
                errorbar="se",
                palette=self.colors_dict,
                height=5.15,
                aspect=0.8,
                sharey=True,
                sharex=False,
                legend=False
                )
            g.fig.subplots_adjust(right=0.8)  # or try 0.8 if legend still overlaps

            g.set_titles("{col_name}")#{col_name}
            g.set_axis_labels("","correlation")
            for ax in g.axes:
                title_text = ax.get_title()
                region_type = title_text
                
                label_text = region_type
                text_x = 0.5
                text_y = 1.15
            
                # Add rectangle and text
                bbox = dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='none', linewidth=1.5)
                ax.text(text_x, text_y, label_text, fontsize=20, fontweight='bold',
                        ha='center', va='center', transform=ax.transAxes, bbox=bbox)
            g.set_titles("")
            os.makedirs(os.path.join(self.figure_dir,'unit_interpretations'),exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.figure_dir,'unit_interpretations',f"{model}_{layer}.png"),bbox_inches='tight',dpi=300)
            plt.close()
        return freq_df, unit_lists

    def get_top_and_bottom_images_per_unit(
            self,
            feature,
            selected_units,
            top_n=5,
            return_indices=False,
            image_dir=None
        ):
        """
        Get top and bottom image frames for each selected unit individually.

        Parameters
        ----------
        feature : str
            Name of the feature to load.
        selected_units : list[int]
            List of unit indices to retrieve timepoints for.
        top_n : int
            Number of top/bottom timepoints to retrieve per unit.
        return_indices : bool
            Whether to return frame indices instead of file paths.
        image_dir : str
            Directory containing image frames, assumed to be sorted in temporal order.

        Returns
        -------
        results : dict
            Dictionary mapping unit index to a dict with 'top' and 'bottom' lists.
        """
        import os
        import numpy as np
        import re

        # Load feature time series
        feature_csv_path = os.path.join(self.dir, 'features', self.features_dict[feature].lower() + '.csv')
        features = np.loadtxt(feature_csv_path, delimiter=',', dtype=np.float32)
        n_timepoints = features.shape[0]

        # Sort image filenames if not returning indices
        if not return_indices:
            def extract_frame_number(filename):
                match = re.search(r'_(\d+)\.png$', filename)
                return int(match.group(1)) if match else -1  # fallback for unexpected formats

            image_files = sorted(
                [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')],
                key=lambda x: extract_frame_number(os.path.basename(x))
            )
        # Gather results
        results = {}
        for unit in selected_units:
            unit_activity = features[:, unit]
            top_indices = np.argsort(unit_activity)[-top_n:][::-1]
            bottom_indices = np.argsort(unit_activity)[:top_n]

            if return_indices:
                results[unit] = {
                    'top': top_indices.tolist(),
                    'bottom': bottom_indices.tolist()
                }
            else:
                results[unit] = {
                    'top': [image_files[i] for i in top_indices],
                    'bottom': [image_files[i] for i in bottom_indices]
                }

        return results

    def compile_correlated_gif(
        self,
        layer,
        unit,
        annotated_feature,
        image_dir,
        top_k=3,
        window_size=20,
        output_path="",
        fps=0.5
    ):
        """
        Create a GIF of the most correlated windows between a unit and an annotation.
        """
        from PIL import Image
        import imageio
        from PIL import ImageDraw, ImageFont
        import re
        from scipy.stats import zscore
        
        os.makedirs(output_path, exist_ok=True)

        def sliding_window_correlation(unit_ts, anno_ts, window_size):
            """
            Compute sliding window Pearson correlation between a unit time series and an annotated feature time series.
            """
            corrs = []
            for start in range(len(unit_ts) - window_size + 1):
                u_win = unit_ts[start:start + window_size]
                a_win = anno_ts[start:start + window_size]
                if np.std(u_win) == 0 or np.std(a_win) == 0:
                    corrs.append(0)
                else:
                    corrs.append(np.corrcoef(u_win, a_win)[0, 1])
            return np.array(corrs)
        
        layer_path = os.path.join(self.dir, 'features', self.features_dict[layer].lower() + '.csv')
        layer_ts = pd.read_csv(layer_path, header=None)       
        unit_ts = layer_ts.iloc[:, unit].values
        
        anno_path = os.path.join(self.dir, 'features', self.features_dict[annotated_feature].lower() + '.csv')
        anno_ts = pd.read_csv(anno_path, header=None).values.squeeze()

        corrs = sliding_window_correlation(unit_ts, anno_ts, window_size)
        print(corrs)
        top_idxs = np.argsort(corrs)[-top_k:]

        def extract_frame_number(filename):
            match = re.search(r'_(\d+)\.png$', filename)
            return int(match.group(1)) if match else -1  # fallback for unexpected formats

        frame_files = sorted(
            [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')],
            key=lambda x: extract_frame_number(os.path.basename(x))
        )
        

        # Compile images from each top correlated window
        
        for idx in top_idxs:
            # Extract data for time series plot
            u_win = unit_ts[idx:idx + window_size]
            a_win = anno_ts[idx:idx + window_size]
            
            # === Plot time series ===
            plt.figure(figsize=(6, 3))
            #zscore for easier visualization
            u_win = zscore(u_win)
            a_win = zscore(a_win)
            plt.plot(u_win, label=f"Unit {unit}", color="C0", linewidth=2)
            plt.plot(a_win, label=f"{annotated_feature}", color="C1", linewidth=2)
            plt.xlabel("Time (TRs)")
            plt.ylabel("Feature Value (z-scored)")
            plt.title(f"r = {corrs[idx]:.2f}")
            plt.legend(frameon=False)
            plt.tight_layout()
            
            # Save plot
            timeseries_path = f"{output_path}_{layer}_{unit}_{annotated_feature}_top{idx}_corr{corrs[idx]:.2f}_timeseries.png"
            plt.savefig(timeseries_path)
            plt.close()
            gif_frames = []
            window_imgs = frame_files[idx:idx + window_size]
            font = ImageFont.load_default()  # Or specify a .ttf font with ImageFont.truetype

            for t, img_path in enumerate(window_imgs):
                try:
                    img = Image.open(img_path).convert("RGB")
                    draw = ImageDraw.Draw(img)

                    tr_number = t #+ idx
                    tr_label = f"TR {tr_number}"

                    # Use getbbox to compute text size
                    bbox = font.getbbox(tr_label)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    x_offset = 10
                    y_offset = 5
                    position = (img.width - text_width - x_offset, y_offset)

                    # Optional: draw a translucent rectangle behind the text
                    draw.rectangle(
                        [position, (position[0] + text_width, position[1] + text_height)],
                        fill=(0, 0, 0)
                    )

                    draw.text(position, tr_label, fill="white", font=font)
                    gif_frames.append(img)
                except Exception as e:
                    print(f"Skipping {img_path}: {e}")

        # Save as GIF
            if gif_frames:

                corr_val = corrs[idx]
                corr_str = f"{corr_val:.2f}"
                filename = f"{output_path}_{layer}_{unit}_{annotated_feature}_TR-{idx}_corr-{corr_str}.gif"
                
                imageio.mimsave(filename, gif_frames, duration=500)
                
                
            else:
                print("No frames collected; skipping GIF creation.")
    def compare_units_to_features(self, 
                               unit_csv_path, 
                               model,
                               layer_name, 
                               load = False,
                               top_n = None,
                               method='cca', 
                               regions = None,
                               split_hemi=True,
                               ):
        import os
        import pandas as pd
        import numpy as np
        def run_single_cca_or_pls(
            unit_data,
            anno_ts,
            method="cca",
            chunklen=30
        ):
            from cca_zoo.model_selection import GridSearchCV
            from cca_zoo.linear import rCCA, PLS
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import GroupKFold
            import numpy as np

            scaler_X = StandardScaler()
            scaler_Y = StandardScaler()
            if(len(anno_ts.shape)<2):
                anno_ts = anno_ts.reshape(-1, 1)
            unit_data = scaler_X.fit_transform(unit_data)
            anno_data = scaler_Y.fit_transform(anno_ts)
            n_samples = len(anno_ts)
            
            cv_outer = GroupKFold(n_splits=5)
            n_chunks = int(n_samples/self.chunklen)
            #set groups so that it chunks the data according to chunk len, and then the chunks are split into training and test
            groups = [str(ind) for ind in range(0,n_chunks) for x in range(0,self.chunklen)]
            if(len(groups)!=n_samples): #add final group that didn't divide evenly into chunklen if necessary
                diff = n_samples-len(groups)
                groups.extend([str(n_chunks) for x in range(0,diff)]) 
            
            scores = []

            for train_idx, test_idx in cv_outer.split(unit_data, groups=groups):
                X_train, Y_train = unit_data[train_idx], anno_data[train_idx]
                X_test, Y_test = unit_data[test_idx], anno_data[test_idx]
                
                scaler_X = StandardScaler()
                scaler_Y = StandardScaler()
                
                X_train = scaler_X.fit_transform(X_train)
                Y_train = scaler_Y.fit_transform(Y_train)
                X_train = np.nan_to_num(X_train)
                Y_train = np.nan_to_num(Y_train)
                
                X_test = scaler_X.transform(X_test)
                Y_test = scaler_Y.transform(Y_test)
                X_test = np.nan_to_num(X_test)
                Y_test = np.nan_to_num(Y_test)
                
                cv_inner = GroupKFold(n_splits=5)
                n_samples = X_train.shape[0]
                n_chunks = int(n_samples/self.chunklen)
                groups = [str(ind) for ind in range(0,n_chunks) for x in range(0,self.chunklen)]
                if(len(groups)!=n_samples): #add final group that didn't divide evenly into chunklen if necessary
                    diff = n_samples-len(groups)
                    groups.extend([str(n_chunks) for x in range(0,diff)])
                reg_params = np.logspace(-5,0,5)
                if method == "cca":
                    model = GridSearchCV(
                        estimator=rCCA(latent_dimensions=1),
                        param_grid={"c": [reg_params, reg_params]},
                        cv=cv_inner,
                        error_score="raise",n_jobs=-1
                    ).fit((X_train, Y_train), groups=groups)
                    score = model.best_estimator_.average_pairwise_correlations((X_test, Y_test)).mean()
                elif method == "pls":
                    model = PLS(latent_dimensions=1).fit((X_train, Y_train))
                    score = model.average_pairwise_correlations((X_test, Y_test)).mean()
                else:
                    raise ValueError("Method must be 'cca' or 'pls'")

                scores.append(score)

            return np.mean(scores)

        if not load:
            # Load unit time courses
            layer_path = os.path.join(self.dir, 'features', self.features_dict[layer_name].lower() + '.csv')
            layer_ts = pd.read_csv(layer_path, header=None)

            # Load unit selections
            unit_df = pd.read_csv(f"{unit_csv_path}/top_unit_correlations/subjectwise_unitwise_annotation_corrs_{model}_{layer_name}.csv")
            unit_df["rank"] = unit_df["rank"].astype(int)

            annotated_features = self.model_features_dict['annotated'] #+ ['motion','word2vec'] + ['sbert_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]]
            if regions is None:
                regions = unit_df['mask'].unique()

            results = []

            for subject in tqdm(self.subjects['sherlock'], desc="Running CCA/PLS per subject"):
                subject_units = unit_df[unit_df['subject'] == subject]
                for annotated_feature in annotated_features:
                    for region in regions:
                        if split_hemi:
                            hemi_groups = subject_units[subject_units['mask'] == region]['hemi'].unique()
                        else:
                            hemi_groups = [None]
                        for hemi in hemi_groups:
                            if split_hemi:
                                region_units = subject_units[(subject_units['mask'] == region) & (subject_units['hemi'] == hemi)]
                            else:
                                region_units = subject_units[subject_units['mask'] == region]

                            if region_units.empty:
                                continue
                            
                            if top_n is not None:
                                region_units = region_units.nsmallest(top_n, 'rank')
                            
                            selected_units = region_units['unit_index'].unique()
                            if len(selected_units) == 0:
                                continue

                            try:
                                anno_path = os.path.join(self.dir, 'features', self.features_dict[annotated_feature].lower() + '.csv')
                                anno_ts = pd.read_csv(anno_path, header=None).values.squeeze()
                                unit_timecourses = layer_ts.iloc[:, selected_units].values

                                score = run_single_cca_or_pls(unit_timecourses, anno_ts, method=method)

                                result = {
                                    'subject': subject,
                                    'annotated_feature': annotated_feature,
                                    'region': region,
                                    'method': method,
                                    'score': score
                                }
                                if split_hemi:
                                    result['hemi'] = hemi
                                results.append(result)
                            except Exception as e:
                                print(f"Error processing {subject}, {annotated_feature}, {region}, hemi={hemi}: {e}")
                                continue

            results_df = pd.DataFrame(results)
            results_df.replace(self.labels_dict,inplace=True)
            results_df.to_csv((os.path.join(self.out_dir, f"subjectwise_unit_feature_CCA_{self.model}_{layer_name}.csv")))
        
        
        results_path = os.path.join(self.out_dir)
        hue_order = [ self.labels_dict.get(item, item) for item in self.model_features_dict['annotated'] ]
        
        bar_fig,point_fig = plotting_helpers.plot_top_unit_scores(results_path,model='vislang',layer_to_plot=layer_name,col_wrap=None,hue_order=hue_order)

        region_sets = {'bar': bar_fig, 'point': point_fig}
        MT = self.MT
        ISC = self.ISC
        STS = ['STS'] 
        language = ['temporal','frontal'] 
        
        pvalue_results = []

        for label, fig in region_sets.items():
            for ax_row in fig.axes:
                for ax in ax_row:
                    title_text = ax.get_title().split(' = ')[1]
                    region_type = title_text
                    regions = (
                        MT if region_type == 'MT'
                        else ISC if region_type == 'ISC'
                        else language if region_type == 'language'
                        else STS
                    )
                    if((region_type!='motion')):
                        sns.despine(ax=ax, left=True)
                        ax.tick_params(left=False, labelleft=False, )
                    
                    # ax.set_xticklabels([f"{t.get_text().split(' ')[1]}\n{t.get_text().split(' ')[0]}" for t in ax.get_xticklabels()])
                    # for hemi_text, xpos in zip(['left', 'right'], [0.25, 0.75]):
                    #     ax.text(xpos, -0.1, hemi_text, transform=ax.transAxes, ha='center', va='top')
                
                    region_name = region_type + ' regions'
                    label_text = region_name
                    text_x = 0.5
                    text_y = 1.01
                
                    # Add rectangle and text
                    bbox = dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='none', linewidth=1.5)
                    ax.text(text_x, text_y, label_text, fontsize=20, fontweight='bold',
                            ha='center', va='center', transform=ax.transAxes, bbox=bbox)

            fig.set_axis_labels("", "Canonical correlation coefficient")
            fig.set_titles("")
        

        bar_fig.savefig(
                os.path.join(
                    f"{self.figure_dir}/{layer_name}_cca.png"),
                bbox_inches='tight',
                dpi=300
            )
        plt.close()
        
        results_df = pd.read_csv(os.path.join(self.out_dir, f"subjectwise_unit_feature_CCA_{self.model}_{layer_name}.csv"))
        results_df['hemisphere'] = 'both'
        results_df['Feature_Space'] = results_df["annotated_feature"]
        results_df['encoding_response'] = results_df['score']
        results_df['hemi_mask'] = [hemi + ' ' + mask for hemi,mask in zip(results_df['hemisphere'],results_df['region'])]
        anova_df = stats_helpers.run_anova_per_region_pair_per_hemisphere(
            data=results_df,
            plot_features=[self.labels_dict.get(feature,feature) for feature in self.model_features_dict['annotated']]
        )
        csv_path = os.path.join(self.out_dir, f"{self.sid}{self.enc_file_label}_model-{self.model}_{label}_unit_feature_comparison_anova_interaction_test.csv")            
        anova_df.to_csv(csv_path, index=False)
        
        output_path = os.path.join(self.dir,'tables',f"pairwise_anova_{self.model}_unit_feature_comparisons.tex")
        stats_helpers.generate_latex_anova_pairwise_table(anova_df,output_path)
        
        
        return results_df
    def plot_unit_feature_highlights(
        self,
        layer,
        unit,
        annotated_features,
        image_dir,
        title = '',
        y_min = -15,
        window_size=20,
        top_k=3,
        output_path="",
    ):
        import os
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from PIL import Image
        from PIL import ImageEnhance
        import re
        from scipy.stats import zscore
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox

        os.makedirs(output_path, exist_ok=True)
        plt.rcParams.update({'font.size': 50, 'font.family': 'Arial'})


        def sliding_window_correlation(unit_ts, anno_ts, window_size):
            corrs = []
            for start in range(len(unit_ts) - window_size + 1):
                u_win = unit_ts[start:start + window_size]
                a_win = anno_ts[start:start + window_size]
                if np.std(u_win) == 0 or np.std(a_win) == 0:
                    corrs.append(0)
                else:
                    corrs.append(np.corrcoef(u_win, a_win)[0, 1])
            return np.array(corrs)

        def extract_frame_number(filename):
            match = re.search(r'_(\d+)\.png$', filename)
            return int(match.group(1)) if match else -1

        layer_path = os.path.join(self.dir, 'features', self.features_dict[layer].lower() + '.csv')
        unit_ts = pd.read_csv(layer_path, header=None).iloc[:, unit].values
        # unit_ts = zscore(unit_ts)

        frame_files = sorted(
            [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.png')],
            key=lambda x: extract_frame_number(os.path.basename(x))
        )

        fig, ax = plt.subplots(figsize=(60, 15))
        ax.plot(unit_ts, color="black", label=f"Unit {unit}",lw=5)
        img_y_offset = unit_ts.min() - 1  # vertical offset for image display
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(y_min, 10)
        ax.set_xlim(-10, 1931)
        ylim = ax.get_ylim()
        sns.despine(ax=ax,top=True,right=True,left=False,bottom=True)
        ax.set_xticks([])
        
        used_image_paths = set()


        img_coords = []
        used_label_slots = []  # (x_center, y_level)
        for feat_idx, feat_name in enumerate(annotated_features):
            anno_path = os.path.join(self.dir, 'features', self.features_dict[feat_name].lower() + '.csv')
            anno_ts = pd.read_csv(anno_path, header=None).values.squeeze()
            # anno_ts = zscore(anno_ts)

            corrs = sliding_window_correlation(unit_ts, anno_ts, window_size)
            # top_idxs = np.argsort(corrs)[-top_k:]
            sorted_idxs = np.argsort(corrs)[::-1]

            # Greedily select non-overlapping windows
            top_idxs = []
            for idx in sorted_idxs:
                if all(abs(idx - chosen) >= window_size for chosen in top_idxs):
                    top_idxs.append(idx)
                if len(top_idxs) == top_k:
                    break
            label_spacing = 110    # how close labels can be horizontally
            label_y_step = 0.27*ylim[1]   # vertical offset between stacked labels

            for i, idx in enumerate(top_idxs):
                color = self.colors_dict.get(feat_name, f"C{feat_idx}")
                x_center = idx + window_size / 2

                # Highlight region
                ax.add_patch(Rectangle(
                    (idx, 0), window_size, ylim[1],
                    color=color, alpha=0.3, label=feat_name if i == 0 else None))

                # Annotate with correlation value
                corr_value = corrs[idx]

                # Stack labels vertically to avoid x-collisions
                y_level = 0
                for (existing_x, existing_level) in used_label_slots:
                    if abs(x_center - existing_x) < label_spacing:
                        y_level = max(y_level, existing_level + 1)
                used_label_slots.append((x_center, y_level))

                y_position = ylim[1] - 0.3 * ylim[1] + y_level * label_y_step

                ax.text(
                    x_center, y_position,
                    f"{self.labels_dict.get(feat_name, feat_name)}\nr = {corr_value:.2f}",
                    ha='center', va='top',
                    fontsize=40, color=color, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=5, alpha=0.8),
                    zorder=20
                )

                # Get representative frame from that window
                window_imgs = frame_files[idx:idx + window_size]
                best_img_idx = np.argmax(zscore(unit_ts[idx:idx + window_size]))# + zscore(anno_ts[idx:idx + window_size])) 
                best_img_path = window_imgs[best_img_idx] if best_img_idx < len(window_imgs) else None

                if best_img_path and os.path.exists(best_img_path) and best_img_path not in used_image_paths:
                    img = Image.open(best_img_path)
                    # Brighten the image
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(1.5)  # Factor > 1 brightens, < 1 darkens (e.g., 1.5 = 50% brighter)
                    img.thumbnail((485, 485), Image.Resampling.LANCZOS)
                    x_img = x_center - (img.width / fig.dpi / 2)
                    img_coords.append((x_img, img_y_offset, img, color))
                    used_image_paths.add(best_img_path)

        used_slots = []  # keep track of existing image x-ranges and their y-offset levels
        img_spacing = 220  # how far apart images need to be on x-axis
        y_offset_step = 4  # how much to lower each new row

        for x_img, y_img, img, color in sorted(img_coords, key=lambda x: x[0]):
            # Determine the vertical offset to avoid overlap
            y_level = 0
            for (existing_x, existing_level) in used_slots:
                if abs(x_img - existing_x) < img_spacing:
                    y_level = max(y_level, existing_level + 1)
            
            # Apply vertical offset based on level
            new_y_img = img_y_offset - (y_offset_step * y_level)
            used_slots.append((x_img, y_level))

            # Draw line
            ax.plot([x_img, x_img], [0, new_y_img], linestyle="--", color=color,zorder=10,lw=5 )

            # Draw image
            imgbox = OffsetImage(img, zoom=1)
            ab = AnnotationBbox(imgbox, (x_img, new_y_img), frameon=False, box_alignment=(0.5, 1.0))
            ax.add_artist(ab)
            
        if len(title)>0:
            bbox = dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='none', linewidth=5)
            ax.text(0.01, 1, title, fontsize=60, fontweight='bold',
                    ha='left', va='center', transform=ax.transAxes, bbox=bbox)

        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel(f"Unit {unit} activation")
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys())

        plt.tight_layout()
        save_path = os.path.join(output_path, f"highlighted_{layer}_unit-{unit}.png")
        plt.savefig(save_path)
        plt.close()

    

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
                'models': self.parent.models
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

    def load_files(self, config, response_label, load=False):
        """
        Load all necessary NIfTI files (GLM, encoding, ISC, cross-subject).
        """
        if load:
            return None  # Optionally, load precomputed things if needed

        files = {}

        # --- Load ISC and cross-subject files
        files[('ISC')] = nibabel.load(
            os.path.join(self.parent.dir, 'analysis', 'SecondLevelGroup', 'performance',
                         f"sub-NT_brain2brain_correlation_smoothingfwhm-{self.parent.smoothing_fwhm}_chunklen-{self.parent.chunklen}_measure-perf_raw.nii.gz")
        )

        for subject in self.parent.subjects['SIpointlights']:
            files[('cross-subject encoding (ISC)', subject)] = nibabel.load(
                os.path.join(self.parent.dir, 'analysis', 'Brain2Brain', 'performance',
                             f"{subject}_brain2brain_encoding_smoothingfwhm-{self.parent.smoothing_fwhm}_chunklen-{self.parent.chunklen}_mask-ISC_measure-perf_raw.nii.gz")
            )
            files[('cross-subject encoding (MT+STS+language)', subject)] = nibabel.load(
                os.path.join(self.parent.dir, 'analysis', 'Brain2Brain', 'performance',
                             f"{subject}_brain2brain_encoding_smoothingfwhm-{self.parent.smoothing_fwhm}_chunklen-{self.parent.chunklen}_mask-combined_parcels_measure-perf_raw.nii.gz")
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
                files[(model, subject)] = nibabel.load(
                    os.path.join(self.parent.enc_dir, response_folder,
                                 f"{subject}_encoding_model-{model}_smoothingfwhm-{self.parent.smoothing_fwhm}_chunklen-{self.parent.chunklen}_mask-{self.parent.mask_name}_measure-{resp_label}.nii.gz")
                )

                if response_label == 'ind_product_measure':
                    files[(model, subject, 'performance')] = nibabel.load(
                        os.path.join(self.parent.enc_dir, 'performance',
                                     f"{subject}_encoding_model-{model}_smoothingfwhm-{self.parent.smoothing_fwhm}_chunklen-{self.parent.chunklen}_mask-{self.parent.mask_name}_measure-perf_raw.nii.gz")
                    )
                if response_label == 'features_preferred_delay':
                    files[(model, subject, 'ind_product_measure')] = nibabel.load(
                        os.path.join(self.parent.enc_dir, 'ind_product_measure',
                                     f"{subject}_encoding_model-{model}_smoothingfwhm-{self.parent.smoothing_fwhm}_chunklen-{self.parent.chunklen}_mask-{self.parent.mask_name}_measure-ind_product_measure_raw.nii.gz")
                    )

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

        for glm_task1 in self.parent.glm_task:
            for glm_task2 in self.parent.glm_task:
                run_groups = [(self.parent.all_runs[glm_task1], self.parent.all_runs[glm_task2])] if glm_task1 != glm_task2 else self.parent.run_groups[glm_task1]
                for subject in self.parent.subjects[glm_task1]:
                    for localizer_contrast in self.parent.localizer_contrasts[glm_task1]:
                        for response_contrast in self.parent.response_contrasts[glm_task2]:
                            for hemi in ['left', 'right']:
                                for mask_name in self.parent.localizer_masks[localizer_contrast]:
                                    for feature_name in feature_names:
                                        for model_name in models:
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
        ISC_responses = []
        cross_subject_responses_anat = []
        cross_subject_responses_ISC = []
        num_voxels = []
        prop_voxels = []
        localizer_runs = []
        response_runs = []

        all_glm_voxelwise = []
        all_enc_voxelwise = []
        all_ISC_voxelwise = []
        all_cross_subject_anat_voxelwise = []
        all_cross_subject_ISC_voxelwise = []

        for localize_run, response_run in run_groups:
            try:
                z_scores_img = files[(glm_task1, localize_run, localizer_contrast, subject)]
                localizer_label = f"{subject}_smoothingfwhm-{self.parent.smoothing_fwhm}_mask-{mask_name}_glm_loc-{localizer_contrast}_run-{localize_run}"
                localizer_mask = self.parent.mm.get_subject_mask_glm(localizer_label, mask_name, z_scores_img)

                glm_weights_img = files[(glm_task2, response_run, response_contrast, subject)]
                glm_weights = glm_weights_img.get_fdata()
            except Exception as e:
                print(e)
                continue

            ISC = files[('ISC')].get_fdata()
            cross_subject_anat = files[('cross-subject encoding (MT+STS+language)', subject)].get_fdata()
            cross_subject_ISC = files[('cross-subject encoding (ISC)', subject)].get_fdata()

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
                    enc_response[:, constraining_enc_response < 0] = 0

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
            ISC_data = ISC[combined_mask]
            cross_subject_data_anat = cross_subject_anat[combined_mask]
            cross_subject_data_ISC = cross_subject_ISC[combined_mask]

            if self.parent.scale_by == 'total_variance':
                enc_data = np.clip(enc_data, a_min=0, a_max=None)
                enc_data = enc_data / enc_data.sum(axis=0, keepdims=True)

            #only save the first loc run, resp run combo so we have responses for all regions
            if ((localize_run == self.parent.all_runs[glm_task1])|(localize_run == self.parent.run_groups[glm_task1][1][0])): 
                all_glm_voxelwise.append(glm_data)
                all_enc_voxelwise.append(enc_data)
                all_ISC_voxelwise.append(ISC_data)
                all_cross_subject_anat_voxelwise.append(cross_subject_data_anat)
                all_cross_subject_ISC_voxelwise.append(cross_subject_data_ISC)

            num_voxel = len(glm_data)
            glm_responses.append(np.nanmean(glm_data) if not np.isnan(glm_data).all() else np.nan)
            enc_responses.append(np.nanmean(enc_data) if not np.isnan(enc_data).all() else np.nan)
            ISC_responses.append(np.nanmean(ISC_data) if not np.isnan(ISC_data).all() else np.nan)
            cross_subject_responses_anat.append(np.nanmean(cross_subject_data_anat) if not np.isnan(cross_subject_data_anat).all() else np.nan)
            cross_subject_responses_ISC.append(np.nanmean(cross_subject_data_ISC) if not np.isnan(cross_subject_data_ISC).all() else np.nan)
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
                'ISC': safe_mean(ISC_responses),
                'cross-subject encoding (MT+STS+language)': safe_mean(cross_subject_responses_anat),
                'cross-subject encoding (ISC)': safe_mean(cross_subject_responses_ISC),
                'num_voxels': np.mean(num_voxels),
                'proportion_voxels': np.mean(prop_voxels),
                'averaged_localizer_runs': ",".join(localizer_runs),
                'averaged_response_runs': ",".join(response_runs),
                'glm_voxelwise': np.concatenate(all_glm_voxelwise) if all_glm_voxelwise else np.array([]),
                'enc_voxelwise': np.concatenate(all_enc_voxelwise) if all_enc_voxelwise else np.array([]),
                'ISC_voxelwise': np.concatenate(all_ISC_voxelwise) if all_ISC_voxelwise else np.array([]),
                'cross_subject_anat_voxelwise': np.concatenate(all_cross_subject_anat_voxelwise) if all_cross_subject_anat_voxelwise else np.array([]),
                'cross_subject_ISC_voxelwise': np.concatenate(all_cross_subject_ISC_voxelwise) if all_cross_subject_ISC_voxelwise else np.array([]),
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
            #put ISC and cross-subject encoding as models
            for add_this in ['ISC','cross-subject encoding (MT+STS+language)','cross-subject encoding (ISC)']:
                temp_results = summary_df.copy()
                temp_results['model'] = [add_this for item in summary_df['model']]
                temp_results['encoding_response'] = temp_results[add_this]

                summary_df = pd.concat([summary_df,temp_results],ignore_index=True)
        
        #take out subjects with no data
        summary_df = summary_df[summary_df['num_voxels']!=np.nan]
        summary_path = self.parent.save_data(summary_df, os.path.join(self.parent.out_dir, output_basename + '_summary'), save_type='csv')
        voxelwise_path = self.parent.save_data(voxelwise_rows, os.path.join(self.parent.out_dir, output_basename + '_voxelwise'), save_type='pkl')

        return summary_path, voxelwise_path

    def collect_glm_voxel_selection(self, response_label, filepath_tag='', extraction_threshold=0, load=False):
        """
        Full pipeline to collect GLM voxel selection results.
        """
        config = self.build_config(response_label)
        files = self.load_files(config, response_label, load=load)
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
        
        if(response_label=='performance'):
            #put ISC and cross-subject encoding as models
            for add_this in ['ISC','cross-subject encoding (MT+STS+language)','cross-subject encoding (ISC)']:
                temp_results = results.copy()
                temp_results['model'] = [add_this for item in results['model']]
                temp_results['encoding_response'] = temp_results[add_this]

                results = pd.concat([results,temp_results],ignore_index=True)

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
                        default='/Users/hsmall2/Documents/GitHub/deep_nat_lat')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/hsmall2/Documents/GitHub/deep_nat_lat/analysis')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/hsmall2/Documents/GitHub/deep_nat_lat/figures')
    args = parser.parse_args()
    SecondLevelIndividual(args).run()

if __name__ == '__main__':
    main()
    
