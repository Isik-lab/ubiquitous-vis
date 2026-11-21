import os
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import pandas as pd 
from pathlib import Path

from src import encoding, helpers
import argparse
class FeatureSpaceSimilarity(encoding.EncodingModel):
    def __init__(self, args):
        self.process = 'FeatureSpaceCorrelation'
        self.chunklen = args.chunklen
        self.features = args.features
        self.feature1 = args.features.split('-')[0]
        self.feature2 = args.features.split('-')[1]
        self.method = args.method
        self.latent_dim = args.latent_dim
        self.srp_matrices = {} #for feature space dim reduction
        self.dir = args.dir
        self.out_dir = args.out_dir + "/" + self.process
        self.figure_dir = args.figure_dir + "/" + self.process
        self.apply_speech_masking = getattr(args, 'apply_speech_masking', False) #whether to restrict data to speech periods only (default: False)
        Path(f'{self.out_dir}/').mkdir(exist_ok=True, parents=True)
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
                       'hue':'hue',
                       'alexnet_pca' : 'AlexNet_all_layers_pcs_363dims',
                        'motion_pca': 'motion_pcs_363dims',
                        'hubert_pca': 'HuBERT_all_layers_pcs_363dims',
                        'word2vec_pca': 'word2vec_pcs_363dims',
                        'GPT2_pca': 'GPT2_chunk30_all_layers_pcs_363dims',
                        'SimCLR_pca': 'SimCLR_attention_all_layers_pcs_363dims',
                        'alexnet_pca_90' : 'alexnet_pcs_1011dims',
                        'motion_pca_90': 'motion_pcs_81dims',
                        'hubert_chunk60_pca_90': 'hubert_chunk60_pcs_349dims',
                        'hubert_chunk60_layer1_pca_90': 'hubert_chunk60_layer1_pcs_54dims',
                        'hubert_chunk60_layer2_pca_90': 'hubert_chunk60_layer2_pcs_77dims',
                        'hubert_chunk60_layer3_pca_90': 'hubert_chunk60_layer3_pcs_109dims',
                        'hubert_chunk60_layer4_pca_90': 'hubert_chunk60_layer4_pcs_133dims',
                        'hubert_chunk60_layer5_pca_90': 'hubert_chunk60_layer5_pcs_148dims',
                        'hubert_chunk60_layer6_pca_90': 'hubert_chunk60_layer6_pcs_166dims',
                        'hubert_chunk60_layer7_pca_90': 'hubert_chunk60_layer7_pcs_188dims',
                        'hubert_chunk60_layer8_pca_90': 'hubert_chunk60_layer8_pcs_221dims',
                        'hubert_chunk60_layer9_pca_90': 'hubert_chunk60_layer9_pcs_229dims',
                        'hubert_chunk60_layer10_pca_90': 'hubert_chunk60_layer10_pcs_216dims',
                        'hubert_chunk60_layer11_pca_90': 'hubert_chunk60_layer11_pcs_140dims',
                        'hubert_chunk60_layer12_pca_90': 'hubert_chunk60_layer12_pcs_100dims',
                        'gpt2_chunk60_layer1_pca_90': 'gpt2_chunk60_layer1_pcs_194dims',
                        'gpt2_chunk60_layer2_pca_90': 'gpt2_chunk60_layer2_pcs_191dims',
                        'gpt2_chunk60_layer3_pca_90': 'gpt2_chunk60_layer3_pcs_203dims',
                        'gpt2_chunk60_layer4_pca_90': 'gpt2_chunk60_layer4_pcs_219dims',
                        'gpt2_chunk60_layer5_pca_90': 'gpt2_chunk60_layer5_pcs_229dims',
                        'gpt2_chunk60_layer6_pca_90': 'gpt2_chunk60_layer6_pcs_240dims',
                        'gpt2_chunk60_layer7_pca_90': 'gpt2_chunk60_layer7_pcs_245dims',
                        'gpt2_chunk60_layer8_pca_90': 'gpt2_chunk60_layer8_pcs_244dims',
                        'gpt2_chunk60_layer9_pca_90': 'gpt2_chunk60_layer9_pcs_240dims',
                        'gpt2_chunk60_layer10_pca_90': 'gpt2_chunk60_layer10_pcs_233dims',
                        'gpt2_chunk60_layer11_pca_90': 'gpt2_chunk60_layer11_pcs_225dims',
                        'gpt2_chunk60_layer12_pca_90': 'gpt2_chunk60_layer12_pcs_226dims',
                        'alexnet_layer1_pca_90': 'alexnet_layer1_raw_pcs_745dims',
                        'alexnet_layer2_pca_90': 'alexnet_layer2_raw_pcs_787dims',
                        'alexnet_layer3_pca_90': 'alexnet_layer3_raw_pcs_1058dims',
                        'alexnet_layer4_pca_90': 'alexnet_layer4_raw_pcs_1061dims',
                        'alexnet_layer5_pca_90': 'alexnet_layer5_raw_pcs_776dims',
                        'alexnet_layer6_pca_90': 'alexnet_layer6_raw_pcs_486dims',
                        'alexnet_layer7_pca_90': 'alexnet_layer7_raw_pcs_523dims',
                        'word2vec_chunk60_pca_90': 'word2vec_chunk60_pcs_71dims',
                        'gpt2_chunk60_pca_90': 'gpt2_chunk60_pcs_318dims',
                        'SimCLR_pca_90': 'SimCLR_attention_all_layers_pcs_745dims',
                        'motion_pcs_90pct': 'motion_pcs_90pct',
                        'word2vec_pcs_90pct': 'word2vec_pcs_90pct',
                        'word2vec_speech_masked_pc90':'word2vec_pcs_90pct_speech_masked',
                        'word2vec_speech_masked':'word2vec_speech_masked',
                        'word2vec_chunk20_speech_masked':'word2vec_chunklen20_speech_masked',
                        'word2vec_chunk60_speech_masked':'word2vec_chunklen60_speech_masked',
                        'word2vec_chunk20':'word2vec_chunklen20',
                        'word2vec_chunk60':'word2vec_chunklen60',
                        'word2vec_chunk80':'word2vec_chunklen80',
                        'word2vec_chunk120':'word2vec_chunklen120',
                        'scene_description':['scene_description_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                        'GPT2_chunk120': [f'GPT2_chunk120_layer{layer_num}' for layer_num in [1,2,3,4,5,6,7,8,9,10,11,12]], 
                        'hubert_chunk120': [f'hubert_chunk120_layer{layer_num}' for layer_num in [1,2,3,4,5,6,7,8,9,10,11,12]], 
                   'SimCLR_attention_eps0.3':['SimCLR_attention_eps0.3_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'SimCLR_embedding_eps0.3':['SimCLR_embedding_eps0.3_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],

                        'motion_speech_masked_pc90':'motion_pcs_90pct_speech_masked',
                        'motion_eps0.1':'pymoten',
                        'motion_eps0.2':'pymoten_srp_eps0.2',
                        'motion_eps0.4':'pymoten_srp_eps0.4',
                        'motion_eps0.6':'pymoten_srp_eps0.6',
                        'word2vec_chunk60_eps0.1':'word2vec_chunklen60',
                        'word2vec_chunk60_eps0.2':'word2vec_chunklen60',
                        'word2vec_chunk60_eps0.4':'word2vec_chunklen60',
                        'word2vec_chunk60_eps0.6':'word2vec_chunklen60_srp_eps0.6',
                        'alexnet_eps0.1_layer1':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-3_srp_eps0.1',
                        'alexnet_eps0.1_layer2':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-6_srp_eps0.1',
                        'alexnet_eps0.1_layer3':'torchvision_alexnet_imagenet1k_v1_ReLU-2-8_srp_eps0.1',
                        'alexnet_eps0.1_layer4':'torchvision_alexnet_imagenet1k_v1_ReLU-2-10_srp_eps0.1',
                        'alexnet_eps0.1_layer5':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-13_srp_eps0.1',
                        'alexnet_eps0.1_layer6':'torchvision_alexnet_imagenet1k_v1_ReLU-2-16',
                        'alexnet_eps0.1_layer7':'torchvision_alexnet_imagenet1k_v1_ReLU-2-19',
                        'alexnet_eps0.2_layer1':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-3_srp_eps0.2',
                        'alexnet_eps0.2_layer2':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-6_srp_eps0.2',
                        'alexnet_eps0.2_layer3':'torchvision_alexnet_imagenet1k_v1_ReLU-2-8_srp_eps0.2',
                        'alexnet_eps0.2_layer4':'torchvision_alexnet_imagenet1k_v1_ReLU-2-10_srp_eps0.2',
                        'alexnet_eps0.2_layer5':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-13_srp_eps0.2',
                        'alexnet_eps0.2_layer6':'torchvision_alexnet_imagenet1k_v1_ReLU-2-16_srp_eps0.2',
                        'alexnet_eps0.2_layer7':'torchvision_alexnet_imagenet1k_v1_ReLU-2-19_srp_eps0.2',
                        'alexnet_eps0.3_layer1':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-3_srp_eps0.3',
                        'alexnet_eps0.3_layer2':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-6_srp_eps0.3',
                        'alexnet_eps0.3_layer3':'torchvision_alexnet_imagenet1k_v1_ReLU-2-8_srp_eps0.3',
                        'alexnet_eps0.3_layer4':'torchvision_alexnet_imagenet1k_v1_ReLU-2-10_srp_eps0.3',
                        'alexnet_eps0.3_layer5':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-13_srp_eps0.3',
                        'alexnet_eps0.3_layer6':'torchvision_alexnet_imagenet1k_v1_ReLU-2-16_srp_eps0.3',
                        'alexnet_eps0.3_layer7':'torchvision_alexnet_imagenet1k_v1_ReLU-2-19_srp_eps0.3',
                        'alexnet_eps0.4_layer1':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-3_srp_eps0.4',
                        'alexnet_eps0.4_layer2':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-6_srp_eps0.4',
                        'alexnet_eps0.4_layer3':'torchvision_alexnet_imagenet1k_v1_ReLU-2-8_srp_eps0.4',
                        'alexnet_eps0.4_layer4':'torchvision_alexnet_imagenet1k_v1_ReLU-2-10_srp_eps0.4',
                        'alexnet_eps0.4_layer5':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-13_srp_eps0.4',
                        'alexnet_eps0.4_layer6':'torchvision_alexnet_imagenet1k_v1_ReLU-2-16_srp_eps0.4',
                        'alexnet_eps0.4_layer7':'torchvision_alexnet_imagenet1k_v1_ReLU-2-19_srp_eps0.4',
                        'alexnet_eps0.6_layer1':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-3_srp_eps0.6',
                        'alexnet_eps0.6_layer2':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-6_srp_eps0.6',
                        'alexnet_eps0.6_layer3':'torchvision_alexnet_imagenet1k_v1_ReLU-2-8_srp_eps0.6',
                        'alexnet_eps0.6_layer4':'torchvision_alexnet_imagenet1k_v1_ReLU-2-10_srp_eps0.6',
                        'alexnet_eps0.6_layer5':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-13_srp_eps0.6',
                        'alexnet_eps0.6_layer6':'torchvision_alexnet_imagenet1k_v1_ReLU-2-16_srp_eps0.6',
                        'alexnet_eps0.6_layer7':'torchvision_alexnet_imagenet1k_v1_ReLU-2-19_srp_eps0.6',
                       }
        
        for layer in self.features_dict['scene_description']:
            self.features_dict[layer] = layer

        # for layer in self.features_dict['sbert']:
        #     self.features_dict[layer]='downsampled_all-mpnet-base-v2_'+layer.split('_')[1]
        # for layer in self.features_dict['GPT2_1sent']:
        #     self.features_dict[layer]='downsampled_GPT2_'+layer.split('_')[2] + '_context-sent-1_embeddings'
        # for layer in self.features_dict['GPT2_3sent']:
        #     self.features_dict[layer]='downsampled_GPT2_'+layer.split('_')[2] + '_context-sent-3_embeddings'
        # for layer in self.features_dict['GPT2_1word']:
        #     self.features_dict[layer]='GPT2_'+layer.split('_')[2] + '_word'
        # for layer in self.features_dict['hubert']:
        #     self.features_dict[layer]='hubert-base-ls960-ft_'+layer.split('_')[1]+'_TR' #embeddings extracted each TR

        # for layer in self.features_dict['alexnet_pca_90']:
        #     self.features_dict[layer] = layer
        # for layer in self.features_dict['alexnet_speech_masked_pc90']:
        #     self.features_dict[layer] = layer
        # # for layer in self.features_dict['hubert_pca_90']:
        # #     self.features_dict[layer] = layer
        # for layer in self.features_dict['hubert_chunk60_speech_masked_pc90']:
        #     self.features_dict[layer] = layer
        # for layer in self.features_dict['GPT2_chunk30_pca_90']:
        #     self.features_dict[layer] = layer
        # for layer in self.features_dict['GPT2_chunk60_speech_masked_pc90']:
        #     self.features_dict[layer] = layer
        # for layer in self.features_dict['GPT2_chunk30']:
        #     self.features_dict[layer]='GPT2_'+layer.split('_')[2] + '_running_time_chunk-30'
        
        # for layer in self.features_dict['GPT2_chunk60_eps0.1']:
        #     self.features_dict[layer]='GPT2_'+layer.split('_')[3] + '_chunklen60_maxcontext30'
        # for layer in self.features_dict['GPT2_chunk60_eps0.2']:
        #     self.features_dict[layer]='GPT2_'+layer.split('_')[3] + '_chunklen60_maxcontext30'
        # for layer in self.features_dict['GPT2_chunk60_eps0.4']:
        #     self.features_dict[layer]='GPT2_'+layer.split('_')[3] + '_chunklen60_maxcontext30_srp_eps0.4'
        # for layer in self.features_dict['GPT2_chunk60_eps0.6']:
        #     self.features_dict[layer]='GPT2_'+layer.split('_')[3] + '_chunklen60_maxcontext30_srp_eps0.6'
        
        # for layer in self.features_dict['hubert_chunk60_eps0.1']:
        #     self.features_dict[layer]='hubert_'+layer.split('_')[3] + '_chunklen60'
        # for layer in self.features_dict['hubert_chunk60_eps0.2']:
        #     self.features_dict[layer]='hubert_'+layer.split('_')[3] + '_chunklen60'
        # for layer in self.features_dict['hubert_chunk60_eps0.4']:
        #     self.features_dict[layer]='hubert_'+layer.split('_')[3] + '_chunklen60_srp_eps0.4'
        # for layer in self.features_dict['hubert_chunk60_eps0.6']:
        #     self.features_dict[layer]='hubert_'+layer.split('_')[3] + '_chunklen60_srp_eps0.6'
        

        # for layer in self.features_dict['GPT2_chunk20']:
        #     self.features_dict[layer]='GPT2_'+layer.split('_')[2] + '_chunklen20_maxcontext30'
        # for layer in self.features_dict['GPT2_chunk40']:
        #     self.features_dict[layer]='GPT2_'+layer.split('_')[2] + '_chunklen40_maxcontext30'
        # for layer in self.features_dict['GPT2_chunk60']:
        #     self.features_dict[layer]='GPT2_'+layer.split('_')[2] + '_chunklen60_maxcontext30'
        # for layer in self.features_dict['GPT2_chunk80']:
        #     self.features_dict[layer]='GPT2_'+layer.split('_')[2] + '_chunklen80_maxcontext60'
        for layer in self.features_dict['GPT2_chunk120']:
            self.features_dict[layer]='GPT2_'+layer.split('_')[2] + '_chunklen80_maxcontext60'
        
        # for layer in self.features_dict['hubert_chunk20']:
        #     self.features_dict[layer]='hubert_'+layer.split('_')[2] + '_chunklen20'
        # for layer in self.features_dict['hubert_chunk40']:
        #     self.features_dict[layer]='hubert_'+layer.split('_')[2] + '_chunklen40'
        # for layer in self.features_dict['hubert_chunk60']:
        #     self.features_dict[layer]='hubert_'+layer.split('_')[2] + '_chunklen60'
        # for layer in self.features_dict['hubert_chunk80']:
        #     self.features_dict[layer]='hubert_'+layer.split('_')[2] + '_chunklen80'
        for layer in self.features_dict['hubert_chunk120']:
            self.features_dict[layer]='hubert_'+layer.split('_')[2] + '_chunklen120'
        
        # for layer in self.features_dict['GPT2_chunk20_speech_masked']:
        #     self.features_dict[layer]='GPT2_'+layer.split('_')[4] + '_chunklen20_maxcontext30_speech_masked'
        # for layer in self.features_dict['GPT2_chunk40_speech_masked']:
        #     self.features_dict[layer]='GPT2_'+layer.split('_')[4] + '_chunklen40_maxcontext30_speech_masked'
        # for layer in self.features_dict['GPT2_chunk60_speech_masked']:
        #     self.features_dict[layer]='GPT2_'+layer.split('_')[4] + '_chunklen60_maxcontext30_speech_masked'
        
        # for layer in self.features_dict['hubert_chunk20_speech_masked']:
        #     self.features_dict[layer]='hubert_'+layer.split('_')[4] + '_chunklen20_speech_masked'
        # for layer in self.features_dict['hubert_chunk40_speech_masked']:
        #     self.features_dict[layer]='hubert_'+layer.split('_')[4] + '_chunklen40_speech_masked'
        # for layer in self.features_dict['hubert_chunk60_speech_masked']:
        #     self.features_dict[layer]='hubert_'+layer.split('_')[4] + '_chunklen60_speech_masked'
            
        # for time_chunk in [1,5,4,8,10,15,16,20]:
        #     for layer in self.features_dict['GPT2_'+str(time_chunk)+'s']:
        #         self.features_dict[layer]='GPT2_'+layer.split('_')[2] + '_running_time_chunk-'+str(time_chunk)

        # tracker=2
        # for layer in self.features_dict['SimCLR_attention_eps0.1']:
        #     self.features_dict[layer]='slip_vit_b_simclr_attention-3-'+str(tracker) +'_srp_eps0.1' #get the SRP reduced version
        #     tracker=tracker+8
        
        # tracker=2
        # for layer in self.features_dict['SimCLR_attention_eps0.2']:
        #     self.features_dict[layer]='slip_vit_b_simclr_attention-3-'+str(tracker) +'_srp_eps0.2' #get the SRP reduced version
        #     tracker=tracker+8
        
        tracker=2
        for layer in self.features_dict['SimCLR_attention_eps0.3']:
            self.features_dict[layer]='slip_vit_b_simclr_attention-3-'+str(tracker) +'_srp_eps0.3' #get the SRP reduced version
            tracker=tracker+8
        
        tracker=6
        for layer in self.features_dict['SimCLR_embedding_eps0.3']:
            self.features_dict[layer]='slip_vit_b_simclr_mlp-3-'+str(tracker) +'_srp_eps0.3' #get the SRP reduced version
            tracker=tracker+8
        
        # tracker=2
        # for layer in self.features_dict['SimCLR_attention_eps0.4']:
        #     self.features_dict[layer]='slip_vit_b_simclr_attention-3-'+str(tracker) +'_srp_eps0.4' #get the SRP reduced version
        #     tracker=tracker+8
        
        # tracker=2
        # for layer in self.features_dict['SimCLR_attention_eps0.6']:
        #     self.features_dict[layer]='slip_vit_b_simclr_attention-3-'+str(tracker) +'_srp_eps0.6' #get the SRP reduced version
        #     tracker=tracker+8

        print(self.features_dict)

    def create_speech_mask(self):
        """
        Create a speech-period mask from language features.
        Returns a boolean array indicating speech periods (True) and non-speech periods (False).
        """
        # Load word2vec to create the mask
        lang_feature_path = self.dir + '/features/word2vec.csv'
        lang_data = np.array(pd.read_csv(lang_feature_path, header=None)).astype(dtype="float32")

        # Create mask: speech periods are where language features are non-zero
        speech_mask = ~np.all(lang_data == 0, axis=1)

        n_samples_total = len(speech_mask)
        n_samples_speech = speech_mask.sum()
        n_excluded = n_samples_total - n_samples_speech

        print('...creating speech-period mask from language features...')
        print(f'  Total samples: {n_samples_total}')
        print(f'  Speech period samples: {n_samples_speech}')
        print(f'  Excluded non-speech samples: {n_excluded} ({n_excluded/n_samples_total*100:.1f}%)')

        return speech_mask

    def canonical_correlation_analysis(self,feature_names=[],latent_dimensions='auto',regularized=False,outer_folds=10,inner_folds=5):
        from cca_zoo.model_selection import GridSearchCV
        from cca_zoo.nonparametric import KCCA
        from cca_zoo.linear import CCA,rCCA

        #### load feature spaces
        ######## and do dimensionality reduction for any multidimensional feature spaces

        # Create speech mask if enabled
        if self.apply_speech_masking:
            speech_mask = self.create_speech_mask()

        loaded_features = {}
        dimensions = []
        for feature_space in feature_names:
            filepath = self.dir + '/features/'+self.features_dict[feature_space].lower()+'.csv'
            data = np.array(pd.read_csv(filepath,header=None))

            # Apply speech mask if enabled
            if self.apply_speech_masking:
                data = data[speech_mask]

            n_samples, n_features = data.shape
            loaded_features[feature_space] = data.astype(dtype="float32")
            dimensions.append(n_features)
        reg_params = np.logspace(-5,0,5) #np.logspace(-10,20,50)
        print(reg_params)
        # param_grid = {"kernel": ["linear"], "c": [reg_params, reg_params]}
        param_grid = {"c": [reg_params, reg_params]}
        n_splits_outer = outer_folds
        n_splits_inner = inner_folds
        
        
        correlations_train = []
        correlations_test = []
        cv_outer = GroupKFold(n_splits=n_splits_outer)
        n_chunks = int(n_samples/self.chunklen)
        #set groups so that it chunks the data according to chunk len, and then the chunks are split into training and test
        groups = [str(ind) for ind in range(0,n_chunks) for x in range(0,self.chunklen)]
        if(len(groups)!=n_samples): #add final group that didn't divide evenly into chunklen if necessary
            print('adding outer stragglers')
            diff = n_samples-len(groups)
            groups.extend([str(n_chunks) for x in range(0,diff)]) 
        splits = cv_outer.split(X=range(0,n_samples),groups=groups)
        for i, (train_outer, test_outer) in enumerate(splits):
            data1 = loaded_features[feature_names[0]]
            data2 = loaded_features[feature_names[1]]

            scaler_X = StandardScaler()
            scaler_Y = StandardScaler()

            train1 = data1[train_outer].astype(dtype="float32")
            train2 = data2[train_outer].astype(dtype="float32")
            train1 = scaler_X.fit_transform(train1)
            train2 = scaler_Y.fit_transform(train2)
            train1 = np.nan_to_num(train1)
            train2 = np.nan_to_num(train2)

            test1 = data1[test_outer].astype(dtype="float32")
            test2 = data2[test_outer].astype(dtype="float32")
            test1 = scaler_X.transform(test1)
            test2 = scaler_Y.transform(test2)
            test1 = np.nan_to_num(test1)
            test2 = np.nan_to_num(test2)


            if(regularized):
                #do temporal chunking for the inner loop as well
                cv_inner = GroupKFold(n_splits=n_splits_inner)
                n_chunks = int(n_samples/self.chunklen)
                n_samples = train1.shape[0]
                n_chunks = int(n_samples/self.chunklen)
                groups = [str(ind) for ind in range(0,n_chunks) for x in range(0,self.chunklen)]
                if(len(groups)!=n_samples): #add final group that didn't divide evenly into chunklen if necessary
                    diff = n_samples-len(groups)
                    groups.extend([str(n_chunks) for x in range(0,diff)])
                inner_splits = cv_inner.split(X=range(0,n_samples),groups=groups)
                # for split in inner_splits:
                #     dimensions.append(len(split[0]))
                # print(dimensions)
                # if(latent_dimensions=='auto'):
                #     latent_dimensions = np.min(dimensions) #take the maximum number of latent dimensions, which is the dimensions of the smallest feature space
                print(train1.shape)
                rCCA_model = rCCA(latent_dimensions=self.latent_dim)
                # Tuning hyperparameters using GridSearchCV for the linear kernel.
                model = GridSearchCV(rCCA_model,param_grid=param_grid,cv=cv_inner,verbose=4, error_score='raise').fit((train1,train2),groups=groups)
            else:
                dimensions.append(train1.shape[0])
                dimensions.append(train2.shape[0])
                # print(dimensions)
                # if(latent_dimensions=='auto'):
                #     latent_dimensions = np.min(dimensions) #take the maximum number of latent dimensions, which is the dimensions of the smallest feature space

                model = CCA(latent_dimensions=self.latent_dim).fit((train1,train2))
            # model.fit_transform((train1, train2))
            # print(rCCA_model.explained_variance((test1,test2)))
            # correlations_train.append(model.score((train1, train2)))
            
            
            # correlations_test.append(model.score((test1, test2))) #score is sum of correlations of latents
            correlations_test.append(
                model.best_estimator_.average_pairwise_correlations((test1, test2)).mean()
            )
            # correlations_test.append(np.sum(rCCA_model.explained_variance((test1,test2))))
        # correlation_train = np.mean(correlations_train)
        correlation_test = np.mean(correlations_test) #get the mean over the folds

        return correlation_test

    def run(self):

        print(self.features)
        if(self.method=='CCA'):
            result = self.canonical_correlation_analysis(feature_names = [self.feature1,self.feature2],regularized=True,outer_folds=5)
        #save the result in a csv
        file_label = self.features+'_latent_dim-'+str(self.latent_dim)
        if self.apply_speech_masking:
            file_label = file_label + '_speech-masked'
        filepath = self.out_dir + '/'+file_label+'.csv'
        with open(filepath, 'w') as file:
            file.write(str(result))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunklen','-chunklen',type=int,default=20)
    parser.add_argument('--features','-features',type=str,default='-') #formatted with features separated by '-' ex) 'social-speaking'
    parser.add_argument('--method','-method',type=str,default='regression') #CCA
    parser.add_argument('--latent_dim','-latent_dim',type=int,default=1)
    parser.add_argument('--apply-speech-masking', action='store_true',
                    help='Restrict analysis to speech periods only')

    parser.add_argument('--dir', '-dir', type=str,
                        default=helpers.PROJECT_DIR)
    parser.add_argument('--out_dir', '-output', type=str,
                        default=os.path.join(helpers.PROJECT_DIR, 'analysis'))
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default=os.path.join(helpers.PROJECT_DIR, 'figures'))
    args = parser.parse_args()
    FeatureSpaceSimilarity(args).run()

if __name__ == '__main__':
    main()

