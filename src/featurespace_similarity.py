import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import pandas as pd 
from pathlib import Path

from src import encoding
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
        Path(f'{self.out_dir}/').mkdir(exist_ok=True, parents=True)
        self.features_dict = {
                       'sbert':['sbert_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                       'GPT2_1sent':['GPT2_1sent_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                       'GPT2_3sent':['GPT2_3sent_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                       'GPT2_1word':['GPT2_1word_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                       'GPT2_4s':['GPT2_4s_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                       'GPT2_8s':['GPT2_8s_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                       'GPT2_16s':['GPT2_16s_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                       'GPT2_24s':['GPT2_24s_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                       'SLIP':'slip_vit_b_yfcc15m',
                       'SimCLR':'slip_vit_b_simclr_yfcc15m',
                       'CLIP':'slip_vit_b_clip_yfcc15m',
                       'CLIPtext':'clip_base_25ep_text_embeddings',
                       'SLIPtext':'slip_base_25ep_text_embeddings',
                       'SLIPtext_100ep':'downsampled_slip_base_100ep_embeddings',
                       'CLIP_ViT':'clip_vitb32',
                       'CLIP_RN':'clip_rn50',
                       'alexnet_layer1':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-3_srp',
                       'alexnet_layer2':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-6_srp',
                       'alexnet_layer3':'torchvision_alexnet_imagenet1k_v1_ReLU-2-8_srp',
                       'alexnet_layer4':'torchvision_alexnet_imagenet1k_v1_ReLU-2-10_srp',
                       'alexnet_layer5':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-13_srp',
                       'alexnet_layer6':'torchvision_alexnet_imagenet1k_v1_ReLU-2-16',
                       'alexnet_layer7':'torchvision_alexnet_imagenet1k_v1_ReLU-2-19',
                       'cochdnn':['cochdnn_layer'+str(layer) for layer in [0,1,2,3,4,5,6]],
                       'hubert':['hubert_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                       'annotated':['social','num_agents','face','valence','arousal','speaking','turn_taking','mentalization','written_text','music'],
                       'SimCLR_attention':['SimCLR_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                       'SLIP_attention':['SLIP_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                       'SimCLR_embedding':['SimCLR_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                       'SLIP_embedding':['SLIP_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                       'SLIP_100ep_attention':['SLIP_100ep_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                       'SLIP_100ep_embedding':['SLIP_100ep_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
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
                       'pixel':'pixel',
                       'hue':'hue',
                       'amplitude':'amplitude',
                       'pitch':'pitch',
                       'music':'music',
                       'glove':'glove',
                       'word2vec':'word2vec',
                       'speaking_turn_taking':'speaking_turn_taking',
                       'pitch_amplitude':'pitch_amplitude'}
        # for layer in [1,2,3,4,5,6,7,8,9,10,11,12]:
        #     self.features_dict['sbert_layer'+str(layer)]='downsampled_all-mpnet-base-v2_layer'+str(layer)
        for layer in self.features_dict['GPT2_1sent']:
            self.features_dict[layer]='downsampled_GPT2_'+layer.split('_')[2] + '_context-sent-1_embeddings'
        for layer in self.features_dict['GPT2_3sent']:
            self.features_dict[layer]='downsampled_GPT2_'+layer.split('_')[2] + '_context-sent-3_embeddings'
        for layer in self.features_dict['sbert']:
            self.features_dict[layer]='downsampled_all-mpnet-base-v2_'+layer.split('_')[1]
        for layer in self.features_dict['GPT2_1sent']:
            self.features_dict[layer]='downsampled_GPT2_'+layer.split('_')[2] + '_context-sent-1_embeddings'
        for layer in self.features_dict['GPT2_3sent']:
            self.features_dict[layer]='downsampled_GPT2_'+layer.split('_')[2] + '_context-sent-3_embeddings'
        for layer in self.features_dict['GPT2_1word']:
            self.features_dict[layer]='GPT2_'+layer.split('_')[2] + '_word'
        for layer in self.features_dict['hubert']:
            self.features_dict[layer]='hubert-base-ls960-ft_'+layer.split('_')[1]+'_tr'

        for time_chunk in [4,8,16,24]:
            for layer in self.features_dict['GPT2_'+str(time_chunk)+'s']:
                self.features_dict[layer]='GPT2_'+layer.split('_')[2] + '_time_chunk-'+str(time_chunk)
        
        tracker=2
        for layer in self.features_dict['SimCLR_attention']:
            self.features_dict[layer]='slip_vit_b_simclr_yfcc15m_attention-3-'+str(tracker) +'_srp' #get the SRP reduced version
            tracker=tracker+8

        tracker=6
        for layer in self.features_dict['SimCLR_embedding']:
            self.features_dict[layer]='slip_vit_b_simclr_yfcc15m_mlp-3-'+str(tracker) +'_srp' #get the SRP reduced version
            tracker=tracker+8
        
        tracker=2
        for layer in self.features_dict['SLIP_attention']:
            self.features_dict[layer]='slip_vit_b_yfcc15m_attention-3-'+str(tracker) +'_srp' #get the SRP reduced version
            tracker=tracker+8
        
        tracker=6
        for layer in self.features_dict['SLIP_embedding']:
            self.features_dict[layer]='slip_vit_b_yfcc15m_mlp-3-'+str(tracker) +'_srp' #get the SRP reduced version
            tracker=tracker+8

        tracker=2
        for layer in self.features_dict['SLIP_100ep_attention']:
            self.features_dict[layer]='slip_vit_b_max_yfcc15m_attention-3-'+str(tracker) +'_srp' #get the SRP reduced version
            tracker=tracker+8
        
        tracker=6
        for layer in self.features_dict['SLIP_100ep_embedding']:
            self.features_dict[layer]='slip_vit_b_max_yfcc15m_mlp-3-'+str(tracker) +'_srp' #get the SRP reduced version
            tracker=tracker+8

        print(self.features_dict)

    def canonical_correlation_analysis(self,feature_names=[],latent_dimensions='auto',regularized=False,outer_folds=10,inner_folds=5):
        from cca_zoo.model_selection import GridSearchCV
        from cca_zoo.nonparametric import KCCA
        from cca_zoo.linear import CCA,rCCA

        #### load feature spaces
        ######## and do dimensionality reduction for any multidimensional feature spaces

        loaded_features = {}
        dimensions = [] 
        for feature_space in feature_names:
            filepath = self.dir + '/features/'+self.features_dict[feature_space].lower()+'.csv'
            data = np.array(pd.read_csv(filepath,header=None))
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
        filepath = self.out_dir + '/'+self.features+'_latent_dim-'+str(self.latent_dim)+'.csv'
        with open(filepath, 'w') as file:
            file.write(str(result))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunklen','-chunklen',type=int,default=20)
    parser.add_argument('--features','-features',type=str,default='-') #formatted with features separated by '-' ex) 'social-speaking'
    parser.add_argument('--method','-method',type=str,default='regression') #CCA
    parser.add_argument('--latent_dim','-latent_dim',type=int,default=1)

    parser.add_argument('--dir', '-dir', type=str,
                        default='/Users/hsmall2/Documents/GitHub/deep_nat_lat')
    parser.add_argument('--out_dir', '-output', type=str,
                        default='/Users/hsmall2/Documents/GitHub/deep_nat_lat/analysis')
    parser.add_argument('--figure_dir', '-figures', type=str,
                        default='/Users/hsmall2/Documents/GitHub/deep_nat_lat/figures')
    args = parser.parse_args()
    FeatureSpaceSimilarity(args).run()

if __name__ == '__main__':
    main()

