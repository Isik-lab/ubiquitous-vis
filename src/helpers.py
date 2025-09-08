def get_subjects(population):
    if(population=='NT'):
        subjects_social = ['sub-05','sub-06','sub-07','sub-08','sub-09','sub-10','sub-11','sub-12','sub-13','sub-14','sub-15','sub-16','sub-19','sub-21','sub-23','sub-25','sub-26','sub-28','sub-32','sub-33','sub-35','sub-36','sub-38','sub-39','sub-41','sub-45','sub-46','sub-48','sub-50','sub-51','sub-53','sub-54','sub-55','sub-56','sub-57','sub-58','sub-60','sub-61','sub-62'] 
        subjects_language = ['sub-19','sub-21','sub-23','sub-25','sub-26','sub-28','sub-32','sub-33','sub-35','sub-36','sub-38','sub-39','sub-41','sub-45','sub-46','sub-48','sub-50','sub-51','sub-53','sub-54','sub-55','sub-56','sub-57','sub-58','sub-60','sub-61','sub-62']
        exclude_subjects = ['sub-11','sub-12','sub-13','sub-21','sub-48']

        subjects={'SIpointlights':[subject for subject in subjects_social if subject not in exclude_subjects],
                  'language':[subject for subject in subjects_language if subject not in exclude_subjects]}
        ## SUBJECT EXCLUSION EXPLANATIONS ##
        #sub 11 only had audio in the right ear
        #sub 12 only had audio in the right ear
        #sub-13 first run of sherlock is not correctly time locked <- check if we actually need to exclude this one
        #sub-21 fell asleep during localizers, scanning mistakes during localizers
        #sub-48 fell asleep during sherlock, unsure if they were paying attention during localizers
               
        ##TEMPORARY
        
    if(population=='ASD'):
        subjects_social = ['sub-04','sub-17','sub-18','sub-20','sub-22','sub-24','sub-27','sub-29','sub-34','sub-37','sub-40','sub-42','sub-43','sub-44','sub-47','sub-49','sub-52','sub-59']
        subjects_language = ['sub-20','sub-27','sub-29','sub-34','sub-37','sub-40','sub-42','sub-43','sub-44','sub-47','sub-49','sub-52','sub-59']
        bad_subjects = ['sub-24','sub-47']
        subjects={'SIpointlights':[subject for subject in subjects_social if subject not in bad_subjects],
                  'language':[subject for subject in subjects_language if subject not in bad_subjects]}
        ## SUBJECT EXCLUSION EXPLANATIONS ##
        # sub 43 more than 10% of sherlock are motion outliers (FD>0.9mm)

    subjects['sherlock'] = subjects['SIpointlights']
    return subjects

def get_models_dict():
    models_dict = {
                   'sbert':['sbert_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'GPT2_1sent':['GPT2_1sent_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'GPT2_3sent':['GPT2_3sent_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'GPT2_1word':['GPT2_1word_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'social':['social'],
                   'motion':['motion'],
                   'word2vec':['word2vec'],
                   'alexnet':['alexnet_layer'+str(layer) for layer in [1,2,3,4,5,6,7]],
                   'hubert':['hubert_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'annotated':['pixel','hue','face','num_agents','social','valence','arousal','pitch','amplitude','music','speaking','turn_taking','mentalization','written_text'],
                   'SimCLR_attention':['SimCLR_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'SLIP_attention':['SLIP_attention_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'SimCLR_embedding':['SimCLR_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'SLIP_embedding':['SLIP_embedding_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10,11,12]],
                   'SLIPtext':['SLIPtext']          }
        # self.model_features_dict['full']=self.model_features_dict['alexnet_layers']+self.model_features_dict['sbert_layers']+['social','num_agents','speaking','turn_taking','mentalization','word2vec','valence','arousal','motion','face','indoor_outdoor','written_text','pixel','hue','pitch','amplitude','music']
    models_dict['GPT2'] = models_dict['GPT2_1sent']
    models_dict['SLIP']=models_dict['SLIP_attention']+models_dict['SLIP_embedding']
    models_dict['SimCLR']=models_dict['SimCLR_attention']+models_dict['SimCLR_embedding']
    
    models_dict['alexnet+motion'] = models_dict['alexnet']+models_dict['motion']
    models_dict['sbert+word2vec'] = models_dict['sbert']+models_dict['word2vec']
    models_dict['hubert+sbert+word2vec'] = models_dict['hubert']+models_dict['sbert']+models_dict['word2vec']
    models_dict['GPT2+word2vec'] = models_dict['GPT2_1sent']+models_dict['word2vec']
        
    models_dict['joint'] = models_dict['alexnet']+ models_dict['motion'] + models_dict['hubert']+ models_dict['word2vec'] + models_dict['sbert'] + models_dict['annotated']

    
    models_dict['vislang_alexnet_1'] =   ['alexnet_layer'+str(layer) for layer in [1]]+ models_dict['motion'] + models_dict['hubert']+ models_dict['word2vec'] + models_dict['sbert']
    models_dict['vislang_alexnet_1-2'] = ['alexnet_layer'+str(layer) for layer in [1,2]]+ models_dict['motion'] + models_dict['hubert']+ models_dict['word2vec'] + models_dict['sbert']
    models_dict['vislang_alexnet_1-3'] = ['alexnet_layer'+str(layer) for layer in [1,2,3]]+ models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec'] + models_dict['sbert']
    models_dict['vislang_alexnet_1-4'] = ['alexnet_layer'+str(layer) for layer in [1,2,3,4]]+ models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec'] + models_dict['sbert']
    models_dict['vislang_alexnet_1-5'] = ['alexnet_layer'+str(layer) for layer in [1,2,3,4,5]]+ models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec'] + models_dict['sbert']
    models_dict['vislang_alexnet_1-6'] = ['alexnet_layer'+str(layer) for layer in [1,2,3,4,5,6]]+ models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec'] + models_dict['sbert']
    models_dict['vislang_alexnet_7'] =   ['alexnet_layer'+str(layer) for layer in [7]]+ models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec'] + models_dict['sbert']
    models_dict['vislang_alexnet_6-7'] = ['alexnet_layer'+str(layer) for layer in [6,7]]+ models_dict['motion'] + models_dict['hubert']+ models_dict['word2vec'] + models_dict['sbert']
    models_dict['vislang_alexnet_5-7'] = ['alexnet_layer'+str(layer) for layer in [5,6,7]]+ models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec'] + models_dict['sbert']
    models_dict['vislang_alexnet_4-7'] = ['alexnet_layer'+str(layer) for layer in [4,5,6,7]]+ models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec'] + models_dict['sbert']
    models_dict['vislang_alexnet_3-7'] = ['alexnet_layer'+str(layer) for layer in [3,4,5,6,7]]+ models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec'] + models_dict['sbert']
    models_dict['vislang_alexnet_2-7'] = ['alexnet_layer'+str(layer) for layer in [2,3,4,5,6,7]]+ models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec'] + models_dict['sbert']
    models_dict['vislang_alexnet_nolayer2'] = ['alexnet_layer'+str(layer) for layer in [1,3,4,5,6,7]]+ models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec'] + models_dict['sbert']
    models_dict['vislang_alexnet_nolayer3'] = ['alexnet_layer'+str(layer) for layer in [1,2,4,5,6,7]]+ models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec'] + models_dict['sbert']
    models_dict['vislang_alexnet_nolayer4'] = ['alexnet_layer'+str(layer) for layer in [1,2,3,5,6,7]]+ models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec'] + models_dict['sbert']
    models_dict['vislang_alexnet_nolayer5'] = ['alexnet_layer'+str(layer) for layer in [1,2,3,4,6,7]]+ models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec'] + models_dict['sbert']
    models_dict['vislang_alexnet_nolayer6'] = ['alexnet_layer'+str(layer) for layer in [1,2,3,4,5,7]]+ models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec'] + models_dict['sbert']
    models_dict['vislang_alexnet_nolayer7'] = ['alexnet_layer'+str(layer) for layer in [1,2,3,4,5,6]]+ models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec'] + models_dict['sbert']


    models_dict['vislang_sbert_11-12'] = ['sbert_layer'+str(layer) for layer in [11,12]] + models_dict['alexnet'] +models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec']
    models_dict['vislang_sbert_9-12'] = ['sbert_layer'+str(layer) for layer in [9,10,11,12]] + models_dict['alexnet'] +models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec']
    models_dict['vislang_sbert_7-12'] = ['sbert_layer'+str(layer) for layer in [7,8,9,10,11,12]] + models_dict['alexnet'] +models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec']
    models_dict['vislang_sbert_5-12'] = ['sbert_layer'+str(layer) for layer in [5,6,7,8,9,10,11,12]] + models_dict['alexnet'] +models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec']
    models_dict['vislang_sbert_3-12'] = ['sbert_layer'+str(layer) for layer in [3,4,5,6,7,8,9,10,11,12]] + models_dict['alexnet'] +models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec']
    models_dict['vislang_sbert_nolayer3-4'] = ['sbert_layer'+str(layer) for layer in [1,2,5,6,7,8,9,10,11,12]] + models_dict['alexnet'] +models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec']
    models_dict['vislang_sbert_nolayer5-6'] = ['sbert_layer'+str(layer) for layer in [1,2,3,4,7,8,9,10,11,12]] + models_dict['alexnet'] +models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec']
    models_dict['vislang_sbert_nolayer7-8'] = ['sbert_layer'+str(layer) for layer in [1,2,3,4,5,6,9,10,11,12]] + models_dict['alexnet'] +models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec']
    models_dict['vislang_sbert_nolayer9-10'] = ['sbert_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,11,12]] + models_dict['alexnet'] +models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec']
    models_dict['vislang_sbert_nolayer11-12'] = ['sbert_layer'+str(layer) for layer in [1,2,3,4,5,6,7,8,9,10]] + models_dict['alexnet'] +models_dict['motion']  + models_dict['hubert']+ models_dict['word2vec']

    models_dict['sbert_early'] = ['sbert_layer'+str(layer) for layer in [1,2,3,4]]
    # models_dict['sbert_early-mid'] = ['sbert_layer'+str(layer) for layer in [4,5,6]]
    models_dict['sbert_mid'] = ['sbert_layer'+str(layer) for layer in [5,6,7,8]]

    # models_dict['sbert_mid-late'] = ['sbert_layer'+str(layer) for layer in [7,8,9]]
    models_dict['sbert_late'] = ['sbert_layer'+str(layer) for layer in [9,10,11,12]]
    
    models_dict['vislang'] = models_dict['alexnet']+ models_dict['motion'] + models_dict['hubert']+ models_dict['word2vec'] + models_dict['sbert']
    models_dict['vislang-alexnet']= models_dict['motion'] + models_dict['hubert']+ models_dict['word2vec'] + models_dict['sbert']
    models_dict['vislang-sbert'] = models_dict['alexnet']+ models_dict['motion'] + models_dict['hubert']+ models_dict['word2vec']
    models_dict['vislang-alexnet-sbert'] = models_dict['motion'] + models_dict['hubert']+ models_dict['word2vec'] 

    models_dict['vision'] = models_dict['alexnet']+ models_dict['motion']
    models_dict['language'] = models_dict['hubert']+ models_dict['word2vec'] + models_dict['sbert']
    
    models_dict['vision_transformers'] = models_dict['SimCLR']+ models_dict['motion']
    models_dict['language_transformers'] = models_dict['hubert']+ models_dict['word2vec'] + models_dict['GPT2_1sent']

    models_dict['full'] = models_dict['annotated'] + models_dict['motion'] + models_dict['alexnet'] + models_dict['word2vec'] + models_dict['sbert']
    models_dict['joint_transformers'] = models_dict['annotated'] + models_dict['motion'] + models_dict['SimCLR_attention'] + models_dict['word2vec'] + models_dict['GPT2_1sent']
    models_dict['vislang_transformers'] = models_dict['motion'] + models_dict['SimCLR'] +  models_dict['hubert'] + models_dict['word2vec'] + models_dict['GPT2_1sent']

    models_dict['GPT2_context_sentences'] = models_dict['SimCLR_attention'] + models_dict['motion'] + models_dict['hubert'] + models_dict['word2vec'] + models_dict['GPT2_1word'] + models_dict['GPT2_1sent'] + models_dict['GPT2_3sent']
    models_dict['GPT2_SimCLR_SLIP_word2vec'] = models_dict['GPT2_1sent']+ models_dict['SimCLR']+models_dict['SLIP'] + models_dict['word2vec']
    for layer in [1,2,3,4,5,6,7,8,9,10,11,12]:
        models_dict['SimCLR_SLIP_layer'+str(layer)] = [model+'_'+layer_type+'_layer'+str(layer) for layer_type in ['attention','embedding'] for model in ['SimCLR','SLIP']]
    
    return models_dict
def get_unique_variance_models(feature_name):
    feature_dict = {
        'alexnet':['vislang','vislang-alexnet'],
        'sbert':['vislang','vislang-sbert'],
        'alexnet_layer1':['vislang','vislang_alexnet_2-7'],
        'alexnet_layer2':['vislang','vislang_alexnet_nolayer2'],
        'alexnet_layer3':['vislang','vislang_alexnet_nolayer3'],
        'alexnet_layer4':['vislang','vislang_alexnet_nolayer4'],
        'alexnet_layer5':['vislang','vislang_alexnet_nolayer5'],
        'alexnet_layer6':['vislang','vislang_alexnet_nolayer6'],
        'alexnet_layer7':['vislang','vislang_alexnet_nolayer7'],
        'sbert_layer1-2':['vislang','vislang_sbert_3-12'],
        'sbert_layer3-4':['vislang','vislang_sbert_nolayer3-4'],
        'sbert_layer5-6':['vislang','vislang_sbert_nolayer5-6'],
        'sbert_layer7-8':['vislang','vislang_sbert_nolayer7-8'],
        'sbert_layer9-10':['vislang','vislang_sbert_nolayer9-10'],
        'sbert_layer11-12':['vislang','vislang_sbert_nolayer11-12']
    }
    return feature_dict[feature_name]
def get_added_variance_models(feature_name):
    #returns a dictionary specifying the two models that will define the unique variance explained by a feature
    #the first model in the list is the model including the feature and the second is the model excluding the feature
    #unique variance should be calculated by subtracting the second model performance (R^2) from the first
    feature_dict = {

        'alexnet_layer1_forward':['vislang_alexnet_1','vislang-alexnet'],
        'alexnet_layer2_forward':['vislang_alexnet_1-2','vislang_alexnet_1'],
        'alexnet_layer3_forward':['vislang_alexnet_1-3','vislang_alexnet_1-2'],
        'alexnet_layer4_forward':['vislang_alexnet_1-4','vislang_alexnet_1-3'],
        'alexnet_layer5_forward':['vislang_alexnet_1-5','vislang_alexnet_1-4'],
        'alexnet_layer6_forward':['vislang_alexnet_1-6','vislang_alexnet_1-5'],
        'alexnet_layer7_forward':['vislang','vislang_alexnet_1-6'],
        'alexnet_layer7_backward':['vislang_alexnet_7',['vislang-alexnet']],
        'alexnet_layer6_backward':['vislang_alexnet_6-7',['vislang_alexnet_7','vislang-alexnet']],
        'alexnet_layer5_backward':['vislang_alexnet_5-7',['vislang_alexnet_6-7','vislang_alexnet_7','vislang-alexnet']],
        'alexnet_layer4_backward':['vislang_alexnet_4-7',['vislang_alexnet_5-7','vislang_alexnet_6-7','vislang_alexnet_7','vislang-alexnet']],
        'alexnet_layer3_backward':['vislang_alexnet_3-7',['vislang_alexnet_4-7','vislang_alexnet_5-7','vislang_alexnet_6-7','vislang_alexnet_7','vislang-alexnet']],
        'alexnet_layer2_backward':['vislang_alexnet_2-7',['vislang_alexnet_3-7','vislang_alexnet_4-7','vislang_alexnet_5-7','vislang_alexnet_6-7','vislang_alexnet_7','vislang-alexnet']],
        'alexnet_layer1_backward':['vislang',['vislang_alexnet_2-7','vislang_alexnet_4-7','vislang_alexnet_5-7','vislang_alexnet_6-7','vislang_alexnet_7','vislang-alexnet']],
        # 'alexnet_layer7_backward':['vislang_alexnet_7','vislang-alexnet'],
        # 'alexnet_layer6_backward':['vislang_alexnet_6-7','vislang_alexnet_7'],
        # 'alexnet_layer5_backward':['vislang_alexnet_5-7','vislang_alexnet_6-7'],
        # 'alexnet_layer4_backward':['vislang_alexnet_4-7','vislang_alexnet_5-7'],
        # 'alexnet_layer3_backward':['vislang_alexnet_3-7','vislang_alexnet_4-7'],
        # 'alexnet_layer2_backward':['vislang_alexnet_2-7','vislang_alexnet_3-7'],
        # 'alexnet_layer1_backward':['vislang','vislang_alexnet_2-7'],
        'GPT2_context_1s_forward':['GPT2_context_1s','vislang-sbert'],
        'GPT2_context_5s_forward':['GPT2_context_1-5s','GPT2_context_1s'],
        'GPT2_context_10s_forward':['GPT2_context_1-10s','GPT2_context_1-5s'],
        'GPT2_context_20s_forward':['GPT2_context_1-20s','GPT2_context_1-10s'],
        'GPT2_context_20s_backward':['GPT2_context_20s','vislang-sbert'],
        'GPT2_context_10s_backward':['GPT2_context_10-20s','GPT2_context_20s'],
        'GPT2_context_5s_backward':['GPT2_context_5-20s','GPT2_context_10-20s'],
        'GPT2_context_1s_backward':['GPT2_context_1-20s','GPT2_context_5-20s'],
        'sbert_layer11-12_backward':['vislang_sbert_11-12',['vislang-sbert']],
        'sbert_layer9-10_backward':['vislang_sbert_9-12',['vislang-sbert','vislang_sbert_11-12']],
        'sbert_layer7-8_backward':['vislang_sbert_7-12',['vislang-sbert','vislang_sbert_11-12','vislang_sbert_9-12']],
        'sbert_layer5-6_backward':['vislang_sbert_5-12',['vislang-sbert','vislang_sbert_11-12','vislang_sbert_9-12','vislang_sbert_7-12']],
        'sbert_layer3-4_backward':['vislang_sbert_3-12',['vislang_sbert_5-12','vislang-sbert','vislang_sbert_11-12','vislang_sbert_9-12','vislang_sbert_7-12','vislang_sbert_5-12']],
        'sbert_layer1-2_backward':['vislang',['vislang-sbert','vislang_sbert_11-12','vislang_sbert_9-12','vislang_sbert_7-12','vislang_sbert_5-12','vislang_sbert_3-12']]
    }
    return(feature_dict[feature_name])

def get_combined_features():
    return ['vision','vision_transformers','language','language_transformers','hubert','GPT2_1word','GPT2_1sent','GPT2_3sent','sbert_layers','alexnet_layers','alexnet','sbert+word2vec','sbert','alexnet+motion','SimCLR','GPT2+word2vec','sbert_early','sbert_early-mid','sbert_mid-late','sbert_late','sbert_mid']

def get_top_percent(data, percent):
    import numpy as np
    
    # data = np.abs(data) ???
    percentile = 100-percent

    threshold = np.nanpercentile(data, percentile)

    num_top_voxels = len(data[data>threshold])

    return (threshold, num_top_voxels)

def get_top_n(data, n):
    import numpy as np
    
    # data = np.abs(data)
    if(data.shape[0]>100):
        x = n/data.shape[0]*100
        percentile = 100-x
        if(percentile<0):
            percentile=0
    else:
        percentile = 0 #take all of the voxels if there are less than 100
    
    # print('percentile')
    # print(percentile)
    # print('data')
    # print(data)
    threshold = np.nanpercentile(data, percentile)

    num_top_voxels = len(data[data>threshold])

    return (threshold, num_top_voxels)
def get_bottom_n(data, n):
    import numpy as np
    
    # data = np.abs(data)
    if(data.shape[0]>100):
        x = n/data.shape[0]*100
        percentile = x
    else:
        percentile = 0 #take all of the voxels if there are less than 100
    
    print('percentile')
    print(percentile)
    threshold = np.nanpercentile(data, percentile)

    num_bottom_voxels = len(data[data<threshold])

    return (threshold, num_bottom_voxels)

def get_bottom_percent(data, percent):
    import numpy as np
    
    data = np.abs(data)
    percentile = percent

    threshold = np.nanpercentile(data, percentile)

    num_bottom_voxels = len(data[data<threshold])

    return (threshold, num_bottom_voxels)

def load_mask(self,mask_name):
    """Returns a specified mask as a Nifti object. 
    If given just a parcel, ie 'STS', will return a binary mask including both left and right hemisphere versions of mask.
    If you want a specific hemisphere, you must follow the convention 'left-STS'.
    This applies for anterior and posterior portions as well. ie 'pSTS' will get both left and right pSTS.
    and 'left-pSTS' will get the left pSTS. """
    import os
    import nibabel
    import nilearn
    import numpy as np
    
    dataset_directory = self.dir
    
    def get_hemi(hemi,mask,radiological):
        fullway = mask.shape[0]
        halfway = int(fullway/2)

        if(radiological):
            #if radiological, swap the left and right
            if(hemi=='right'):
                mask[halfway+1:fullway] = 0 
            elif(hemi=='left'):
                mask[0:halfway]=0 
            elif(hemi=='both'):
                pass
        else:
            if(hemi=='left'):
                mask[halfway+1:fullway] = 0 #remove right hemi
            elif(hemi=='right'):
                mask[0:halfway]=0 #remove left hemi
            elif(hemi=='both'):
                pass
        return mask
    def get_sagittal(sagittal,mask): #if also specifying hemi, hemi should be first for this to be accurate since their might be diff in left and right!!
        Y_index_mask = np.zeros(mask.shape)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                for k in range(mask.shape[2]):
                    Y_index_mask[i, j, k] = j
        Y_mask_indices = Y_index_mask[mask==1]
        posteriorY = np.min(Y_mask_indices)
        anteriorY = np.max(Y_mask_indices)
        midY = int((anteriorY+posteriorY)/2 )
        if(sagittal=='a'): #anterior
            mask[Y_index_mask<midY] = 0 #take out posterior
        elif(sagittal=='p'):#posterior
            mask[Y_index_mask>midY] = 0 #take out anterior
        elif(sagittal=='both'):
            pass
        
        return mask
    #mask_name should always be 'hemi'-'sagittal''mask'
    #hemi = 'left','right'
    #sagittal = 'a','p' (anterior,posterior)
    #example: right-aSTS
    hemi = 'both' #default is both hemispheres
    sagittal = ''
    og_mask_name = mask_name
    split = mask_name.split('-')
    if(len(split)>1):
        hemi = split[0]
        mask_name = split[1]
    else:
        mask_name=split[0]

    #check for a sagittal specifier based on mask
    possible_masks = ['','STS','MT','lateral','ventral','ISC','language','frontal','temporal_language','pTemp','aTemp','combined_parcels','STS+language']
    if(mask_name not in possible_masks):

        sagittal = mask_name[0] #take the sagittal specifier from the front
        mask_name = mask_name[1:] #keep only the mask

        if(mask_name not in possible_masks):
            print('ERROR! mask not specified correctly! mask = ' + mask_name)

    #load mask
    radiological=False
    if(mask_name==''): #no mask is assumed to be whole brain
        mask = nibabel.load(os.path.join(dataset_directory,'analysis','parcels','tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz'))
        mask_affine = mask.affine
        mask = nilearn.image.resample_img(mask, target_affine=mask_affine, target_shape=self.brain_shape,interpolation='nearest')
        mask = mask.get_fdata()*1
    if(mask_name=='STS'):
        left_mask = nibabel.load(os.path.join(dataset_directory,'analysis','parcels','lSTS.nii.gz')) #Ben Deen's parcels (STS+TPJ)
        right_mask = nibabel.load(os.path.join(dataset_directory,'analysis','parcels','rSTS.nii.gz'))
        mask_affine = left_mask.affine
        mask = ((left_mask.get_fdata() + right_mask.get_fdata())>0)*1
        radiological=True
        #74 is left STS, 149 is right STS
    elif(mask_name=='MT'):
        mask = nibabel.load(os.path.join(dataset_directory,'analysis','parcels','MT.nii.gz'))
        mask_affine = mask.affine
        mask = mask.get_fdata()
    elif(mask_name=='lateral'):
        mask = nibabel.load(os.path.join(dataset_directory,'analysis','parcels','lateral_STS_mask.nii.gz'))
        mask_affine = mask.affine 
        mask = mask.get_fdata()
    elif(mask_name=='ventral'):
        mask = nibabel.load(os.path.join(dataset_directory,'analysis','parcels','MNI152_T1_1mm.nii.gz'))
        mask_affine = mask.affine
        mask = mask.get_fdata()
        print(mask[mask>0])
        mask = ((mask==1)|(mask==3)|(mask==5)|(mask==7)|(mask==8)|(mask==9)|(mask==10)|(mask==1))*1
        print(len(mask[mask>0]))
    elif(mask_name =='ISC'):
        # mask = nibabel.load(os.path.join(dataset_directory,'analysis','IntersubjectCorrelation','intersubject_correlation','sub-NT_smoothingfwhm-6.0_type-leave_one_out_mask-None_measure-intersubject_correlation.nii.gz'))
        # mask_affine = mask.affine
        # mask = nilearn.image.resample_img(mask, target_affine=mask_affine, target_shape=self.brain_shape,interpolation='nearest')

        # mask = (mask.get_fdata()>0.15)*1
        mask = nibabel.load(os.path.join(dataset_directory,'analysis','SecondLevelGroup','performance','sub-NT_brain2brain_correlation_smoothingfwhm-6.0_chunklen-20_measure-perf_p_fdr.nii.gz'))
        mask_affine = mask.affine
        mask = mask.get_fdata()*1
    elif(mask_name =='language'):
        mask = nibabel.load(os.path.join(dataset_directory,'analysis','parcels','langloc_n806_top10%_atlas.nii'))
        mask_affine = mask.affine
        mask = (mask.get_fdata()>0.3)*1 #require 30% overlap across 806 participants for this map
        radiological=True
    elif(mask_name =='frontal'):
        mask = nibabel.load(os.path.join(dataset_directory,'analysis','parcels','allParcels_language_SN220.nii'))
        mask_affine = mask.affine
        mask = mask.get_fdata() #frontal regions
        mask = ( (mask==1) | (mask==2) | (mask==3) | (mask==7) | (mask==8) | (mask==9) )*1
        radiological=True
    elif(mask_name =='temporal_language'):
        mask = nibabel.load(os.path.join(dataset_directory,'analysis','parcels','allParcels_language_SN220.nii'))
        mask_affine = mask.affine
        mask = mask.get_fdata() #temporal regions
        mask = ( (mask==4) | (mask==5) | (mask==6) | (mask==10) | (mask==11) | (mask==12) )*1
        radiological=True
    elif(mask_name =='pTemp'):
        mask = nibabel.load(os.path.join(dataset_directory,'analysis','parcels','allParcels_language_SN220.nii'))
        mask_affine = mask.affine
        mask = mask.get_fdata() #temporal regions
        mask = ( (mask==5) | (mask==6) | (mask==11) | (mask==12) )*1
        radiological=True
    elif(mask_name =='aTemp'):
        mask = nibabel.load(os.path.join(dataset_directory,'analysis','parcels','allParcels_language_SN220.nii'))
        mask_affine = mask.affine
        mask = mask.get_fdata() #temporal regions
        mask = ( (mask==4) | (mask==10)  )*1
        radiological=True
    elif(mask_name=='combined_parcels'):
        left_mask = nibabel.load(os.path.join(self.dir,'analysis','parcels','lSTS.nii.gz')) #Ben Deen's parcels (STS+TPJ)
        self.mask_affine = left_mask.affine
        left_mask = nilearn.image.resample_img(left_mask, target_affine=self.mask_affine, target_shape=self.brain_shape,interpolation='nearest')
        right_mask = nibabel.load(os.path.join(self.dir,'analysis','parcels','rSTS.nii.gz'))
        right_mask = nilearn.image.resample_img(right_mask, target_affine=self.mask_affine, target_shape=self.brain_shape,interpolation='nearest')
        STS_mask = ((left_mask.get_fdata() + right_mask.get_fdata())>0)*1.0
        
        mask = nibabel.load(os.path.join(self.dir,'analysis','parcels','MT.nii.gz'))
        mask = nilearn.image.resample_img(mask, target_affine=self.mask_affine, target_shape=self.brain_shape,interpolation='nearest')
        MT_mask = (mask.get_fdata()>0)*1.0
        
        #just temporal language regions right now
        mask = nibabel.load(os.path.join(self.dir,'analysis','parcels','allParcels_language_SN220.nii'))
        mask = nilearn.image.resample_img(mask, target_affine=self.mask_affine, target_shape=self.brain_shape,interpolation='nearest')
        mask_data = mask.get_fdata() #temporal and frontal regions
        lang_mask = ( (mask_data==1) | (mask_data==2) | (mask_data==3) | (mask_data==7) | (mask_data==8) | (mask_data==9) | (mask_data==4) | (mask_data==5) | (mask_data==6) | (mask_data==10) | (mask_data==11) | (mask_data==12) )*1.0
        
        mask = ((STS_mask+MT_mask+lang_mask)>0)*1.0
        mask_affine = self.mask_affine

    #need to do both hemis to get accurate sagittal splits (because left and right hemis might be different)
    mask_dict = {}
    for hemi_label in ['left','right']:
        temp_mask = get_hemi(hemi_label,mask.copy(),radiological)
        
        if(len(sagittal)>0):
            temp_mask = get_sagittal(sagittal,temp_mask)
        mask_dict[hemi_label]=temp_mask

    if(hemi=='both'):
        mask = mask_dict['left'] + mask_dict['right']
        mask = (mask>0)*1
    else:
        mask = mask_dict[hemi]

    mask = nibabel.Nifti1Image(mask.astype('int32'), mask_affine)

    if(mask_name!=''):
        whole_mask = load_mask(self,'')
        mask_affine = whole_mask.affine
        whole_mask = whole_mask.get_fdata()
        
        # print('ISC mask, ', len(whole_mask[(whole_mask==1)]))
        #now self.mask is set to the overarching mask and self.mask_affine is set to mask
        #resample mask so we can overlay the whole_mask and the mask
        mask = nilearn.image.resample_img(mask, target_affine=mask_affine, target_shape=whole_mask.shape,interpolation='nearest')
    
        #only take voxels that are also in the overarching mask
        mask = (mask.get_fdata()==1)&(whole_mask==1)
        mask = nibabel.Nifti1Image(mask.astype('int32'), mask_affine)
        
    return mask
