"""
Intrinsic Dimensionality Analysis Module

This module provides a flexible interface for analyzing the intrinsic dimensionality
of neural network feature spaces using PCA. It integrates with the existing model
organization from helpers.get_models_dict() and provides options for concatenating
layers and feature spaces before dimensionality estimation.

Key Features:
- Accepts model names from helpers.get_models_dict()
- Flexible concatenation options (layers, all features)
- Unified PCA analysis across multiple models
- Single-figure visualization for comparison
- PC time series plotting for temporal analysis
- Summary statistics export

Author: Claude Code
Date: 2026-03-04
Updated: 2026-03-05 (added plot_pc_timeseries method)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src import helpers
from src import plotting_helpers


class IntrinsicDimensionality:
    """
    Analyze the intrinsic dimensionality of neural network feature spaces.

    This class loads features for specified models, organizes them according to
    concatenation options, performs PCA to estimate dimensionality, and provides
    visualization and summary tools.

    Attributes:
        model_list (list): List of model names to analyze
        dir (str): Base directory of the project
        concat_layers (bool): Whether to concatenate layers within models
        concat_all_features (bool): Whether to concatenate all features within composite models
        variance_threshold (float): Variance threshold for dimensionality estimation
        models_dict (dict): Dictionary from helpers.get_models_dict()
        features_dict (dict): Mapping of feature names to CSV filenames
        loaded_features (dict): Dictionary storing loaded feature arrays
        feature_groups (dict): Dictionary of organized feature groups for PCA
        pca_results (dict): Dictionary storing PCA results per feature group

    Methods:
        load_features(): Load feature data from CSV files
        organize_features(): Organize features according to concatenation options
        compute_pca(): Perform PCA analysis on feature groups
        plot_results(): Plot cumulative variance explained curves
        plot_pc_trajectories(): Plot PC1 vs PC2 scatter plots colored by time
        plot_pc_timeseries(): Plot PC values as time series (line plots)
        plot_subsample_analysis(): Analyze dimensionality vs sample size
        save_summary(): Export summary statistics to CSV
        save_pcs(): Save PC time series to CSV files
        run_analysis(): Run complete analysis pipeline
    """

    def __init__(self,
                 model_list,
                 dir=helpers.PROJECT_DIR,
                 concat_layers=True,
                 concat_all_features=False,
                 variance_threshold=0.90):
        """
        Initialize the IntrinsicDimensionality analyzer.

        Parameters:
            model_list (list): List of model names from helpers.get_models_dict()
                              e.g., ['alexnet', 'hubert', 'GPT2_chunk60']
            dir (str): Base directory containing the features folder
            concat_layers (bool): If True, concatenate all layers within each model
            concat_all_features (bool): If True, concatenate all features within composite models
            variance_threshold (float): Variance threshold for dimensionality (default: 0.90)
        """
        self.model_list = model_list
        self.dir = dir
        self.features_dir = os.path.join(dir, 'features')
        self.concat_layers = concat_layers
        self.concat_all_features = concat_all_features
        self.variance_threshold = variance_threshold

        # Get models dictionary from helpers
        self.models_dict = helpers.get_models_dict()

        # Build features dictionary (feature name -> CSV filename mapping)
        self.features_dict = self._build_features_dict()

        # Storage for loaded data and results
        self.loaded_features = {}
        self.feature_groups = {}
        self.pca_results = {}

        print(f"Initialized IntrinsicDimensionality")
        print(f"Models to analyze: {model_list}")
        print(f"Concatenation options: concat_layers={concat_layers}, concat_all_features={concat_all_features}")
        print(f"Variance threshold: {variance_threshold*100}%")

    def _build_features_dict(self):
        """
        Build mapping from feature names to CSV filenames.

        Reuses the exact logic from encoding.py to ensure consistency.

        Returns:
            dict: Mapping of feature names to CSV filenames (without .csv extension)
        """
        features_dict = {
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
            'alexnet_pca' : 'pcs/AlexNet_all_layers_pcs_363dims',
            'motion_pca': 'pcs/motion_pcs_363dims',
            'hubert_pca': 'pcs/HuBERT_all_layers_pcs_363dims',
            'word2vec_pca': 'pcs/word2vec_pcs_363dims',
            'GPT2_pca': 'pcs/GPT2_chunk30_all_layers_pcs_363dims',
            'SimCLR_pca': 'pcs/SimCLR_attention_all_layers_pcs_363dims',
            'word2vec_speech_masked_pc90':'word2vec_pcs_90pct_speech_masked',
            'word2vec_speech_masked':'word2vec_speech_masked',
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
                        'alexnet_eps0.3_layer1':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-3_srp_eps0.3',
                        'alexnet_eps0.3_layer2':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-6_srp_eps0.3',
                        'alexnet_eps0.3_layer3':'torchvision_alexnet_imagenet1k_v1_ReLU-2-8_srp_eps0.3',
                        'alexnet_eps0.3_layer4':'torchvision_alexnet_imagenet1k_v1_ReLU-2-10_srp_eps0.3',
                        'alexnet_eps0.3_layer5':'torchvision_alexnet_imagenet1k_v1_MaxPool2d-2-13_srp_eps0.3',
                        'alexnet_eps0.3_layer6':'torchvision_alexnet_imagenet1k_v1_ReLU-2-16_srp_eps0.3',
                        'alexnet_eps0.3_layer7':'torchvision_alexnet_imagenet1k_v1_ReLU-2-19_srp_eps0.3',
                       
        }
        
        alexnet_layers = ['MaxPool2d-2-3', 'MaxPool2d-2-6', 'ReLU-2-8', 'ReLU-2-10',
                          'MaxPool2d-2-13', 'ReLU-2-16', 'ReLU-2-19']
        for layer_num, layer_name in enumerate(alexnet_layers, start=1):
            features_dict[f'alexnet_layer{layer_num}_raw'] = f'torchvision_alexnet_imagenet1k_v1_{layer_name}_raw'


        # Add layer-specific mappings
        for layer in self.models_dict['sbert']:
            features_dict[layer] = 'downsampled_all-mpnet-base-v2_' + layer.split('_')[1]

        for layer in self.models_dict['GPT2_1sent']:
            features_dict[layer] = 'downsampled_GPT2_' + layer.split('_')[2] + '_context-sent-1_embeddings'

        for layer in self.models_dict['GPT2_3sent']:
            features_dict[layer] = 'downsampled_GPT2_' + layer.split('_')[2] + '_context-sent-3_embeddings'

        for layer in self.models_dict['GPT2_1word']:
            features_dict[layer] = 'GPT2_' + layer.split('_')[2] + '_word'

        for layer in self.models_dict['hubert']:
            features_dict[layer] = 'hubert-base-ls960-ft_' + layer.split('_')[1] + '_TR'

        # Add chunk-based models
        for chunk_len in [20, 30, 40, 60]:
            model_key = f'GPT2_chunk{chunk_len}'
            if model_key in self.models_dict:
                for layer in self.models_dict[model_key]:
                    layer_num = layer.split('_layer')[1]
                    features_dict[layer] = f'gpt2_layer{layer_num}_chunklen{chunk_len}_maxcontext30'

        for chunk_len in [80,120]:
            model_key = f'GPT2_chunk{chunk_len}'
            if model_key in self.models_dict:
                for layer in self.models_dict[model_key]:
                    layer_num = layer.split('_layer')[1]
                    features_dict[layer] = f'gpt2_layer{layer_num}_chunklen{chunk_len}_maxcontext60'

        # Add chunk-based HuBERT models
        for chunk_len in [20, 40, 60, 80, 120]:
            model_key = f'hubert_chunk{chunk_len}'
            if model_key in self.models_dict:
                for layer in self.models_dict[model_key]:
                    layer_num = layer.split('_layer')[1]
                    features_dict[layer] = f'hubert_layer{layer_num}_chunklen{chunk_len}'


        tracker=2
        for layer in self.models_dict['SimCLR_attention_eps0.1']:
            features_dict[layer]='slip_vit_b_simclr_attention-3-'+str(tracker) +'_srp_eps0.1' #get the SRP reduced version
            tracker=tracker+8
        
        tracker=2
        for layer in self.models_dict['SimCLR_attention_eps0.2']:
            features_dict[layer]='slip_vit_b_simclr_attention-3-'+str(tracker) +'_srp_eps0.2' #get the SRP reduced version
            tracker=tracker+8
        
        tracker=2
        for layer in self.models_dict['SimCLR_attention_eps0.3']:
            features_dict[layer]='slip_vit_b_simclr_attention-3-'+str(tracker) +'_srp_eps0.3' #get the SRP reduced version
            tracker=tracker+8

        tracker=6
        for layer in self.models_dict['SimCLR_embedding_eps0.3']:
            features_dict[layer]='slip_vit_b_simclr_mlp-3-'+str(tracker) +'_srp_eps0.3' #get the SRP reduced version
            tracker=tracker+8

        tracker=2
        for layer in self.models_dict['SimCLR_attention_eps0.4']:
            features_dict[layer]='slip_vit_b_simclr_attention-3-'+str(tracker) +'_srp_eps0.4' #get the SRP reduced version
            tracker=tracker+8
        
        tracker=2
        for layer in self.models_dict['SimCLR_attention_eps0.6']:
            features_dict[layer]='slip_vit_b_simclr_attention-3-'+str(tracker) +'_srp_eps0.6' #get the SRP reduced version
            tracker=tracker+8
        # Add SimCLR and SLIP models
        tracker = 2
        for layer in self.models_dict['SimCLR_attention']:
            features_dict[layer] = 'slip_vit_b_simclr_yfcc15m_attention-3-' + str(tracker) + '_srp'
            tracker += 8

        tracker = 6
        for layer in self.models_dict['SimCLR_embedding']:
            features_dict[layer] = 'slip_vit_b_simclr_yfcc15m_mlp-3-' + str(tracker) + '_srp'
            tracker += 8

        tracker = 2
        for layer in self.models_dict['SLIP_attention']:
            features_dict[layer] = 'slip_vit_b_slip_cc12m_attention-3-' + str(tracker) + '_srp'
            tracker += 8

        tracker = 6
        for layer in self.models_dict['SLIP_embedding']:
            features_dict[layer] = 'slip_vit_b_slip_cc12m_mlp-3-' + str(tracker) + '_srp'
            tracker += 8

        # Add word2vec chunk60
        features_dict['word2vec_chunk60'] = 'word2vec_chunklen60'
        features_dict['word2vec_chunk60_speech_masked'] = 'word2vec_chunklen60_speech_masked'
        features_dict['word2vec_chunk120'] = 'word2vec_chunklen120'

        # Add eps-reduced chunk60 features for GPT2
        for eps in ['0.1', '0.2', '0.4', '0.6']:
            model_key = f'GPT2_chunk60_eps{eps}'
            if model_key in self.models_dict:
                for layer in self.models_dict[model_key]:
                    layer_num = layer.split('_')[3]
                    if eps in ['0.1', '0.2']:
                        features_dict[layer] = f'GPT2_{layer_num}_chunklen60_maxcontext30'
                    elif eps == '0.4':
                        features_dict[layer] = f'GPT2_{layer_num}_chunklen60_maxcontext30_srp_eps0.4'
                    elif eps == '0.6':
                        features_dict[layer] = f'GPT2_{layer_num}_chunklen60_maxcontext30_srp_eps0.6'

        # Add eps-reduced chunk60 features for HuBERT
        for eps in ['0.1', '0.2', '0.4', '0.6']:
            model_key = f'hubert_chunk60_eps{eps}'
            if model_key in self.models_dict:
                for layer in self.models_dict[model_key]:
                    layer_num = layer.split('_')[3]
                    if eps in ['0.1', '0.2']:
                        features_dict[layer] = f'hubert_{layer_num}_chunklen60'
                    elif eps == '0.4':
                        features_dict[layer] = f'hubert_{layer_num}_chunklen60_srp_eps0.4'
                    elif eps == '0.6':
                        features_dict[layer] = f'hubert_{layer_num}_chunklen60_srp_eps0.6'

        # Add word2vec eps variants (already included above, but add eps-specific ones)
        for eps in ['0.1', '0.2', '0.4', '0.6']:
            model_key = f'word2vec_chunk60_eps{eps}'
            if model_key in self.models_dict:
                for feature in self.models_dict[model_key]:
                    # word2vec_chunk60_eps variants use same file mapping as base word2vec_chunk60_eps
                    if feature not in features_dict:
                        features_dict[feature] = features_dict.get(f'word2vec_chunk60_eps{eps}', 'word2vec_chunklen60')

        return features_dict

    def _group_features_by_base_model(self, feature_spaces):
        """
        Group feature names by their base model.

        This helper method groups features like ['alexnet_layer1', 'alexnet_layer2', 'motion']
        into {'alexnet': ['alexnet_layer1', 'alexnet_layer2'], 'motion': ['motion']}.

        Handles complex feature names like:
        - 'alexnet_layer1' -> base: 'alexnet'
        - 'hubert_chunk60_eps0.1_layer1' -> base: 'hubert_chunk60_eps0.1'
        - 'word2vec_chunk60_eps0.1' -> base: 'word2vec_chunk60_eps0.1'
        - 'motion' -> base: 'motion'

        Parameters:
            feature_spaces (list): List of feature names

        Returns:
            dict: Dictionary mapping base model names to lists of feature names
        """
        grouped = {}
        for feature_name in feature_spaces:
            # Extract base model name
            if '_layer' in feature_name:
                # Has layers - extract everything before '_layer'
                # e.g., 'alexnet_layer1' -> 'alexnet'
                # e.g., 'hubert_chunk60_eps0.1_layer1' -> 'hubert_chunk60_eps0.1'
                base_model = feature_name.split('_layer')[0]
            else:
                # No layers - the whole name is the base model
                # e.g., 'motion', 'word2vec', 'word2vec_chunk60_eps0.1'
                base_model = feature_name

            if base_model not in grouped:
                grouped[base_model] = []
            grouped[base_model].append(feature_name)

        return grouped

    def load_features(self):
        """
        Load features for all models in model_list.

        For each model, gets the list of feature spaces from models_dict,
        then loads the corresponding CSV files.

        Returns:
            dict: self.loaded_features containing loaded arrays
        """
        print("\n" + "="*80)
        print("LOADING FEATURES")
        print("="*80)

        self.loaded_features = {}

        for model_name in self.model_list:
            if model_name not in self.models_dict:
                print(f"WARNING: Model '{model_name}' not found in models_dict")
                continue

            # Get feature space names for this model
            feature_spaces = self.models_dict[model_name]
            print(f"\nModel: {model_name}")
            print(f"  Feature spaces: {len(feature_spaces)}")

            for feature_name in feature_spaces:
                # Skip if already loaded (in case of overlapping models)
                if feature_name in self.loaded_features:
                    continue

                # Get CSV filename
                if feature_name not in self.features_dict:
                    print(f"  WARNING: Feature '{feature_name}' not found in features_dict, skipping")
                    continue

                csv_filename = self.features_dict[feature_name].lower()
                print(csv_filename)
                filepath = os.path.join(self.features_dir, f'{csv_filename}.csv')

                # Load data
                try:
                    data = pd.read_csv(filepath, header=None).values.astype('float32')
                    self.loaded_features[feature_name] = data
                    print(f"  Loaded {feature_name}: shape {data.shape}")
                except FileNotFoundError:
                    print(f"  WARNING: File not found: {filepath}")
                except Exception as e:
                    print(f"  ERROR loading {feature_name}: {e}")

        print(f"\nSuccessfully loaded {len(self.loaded_features)} feature spaces")
        print("="*80)

        return self.loaded_features

    def organize_features(self):
        """
        Organize features into groups based on concatenation options.

        Behavior depends on concat_layers and concat_all_features:
        - concat_all_features=True: Concatenate ALL features within each model
        - concat_layers=True: Concatenate layers within each sub-model
        - Both False: Keep each feature/layer separate

        Returns:
            dict: self.feature_groups ready for PCA
        """
        print("\n" + "="*80)
        print("ORGANIZING FEATURES FOR PCA")
        print("="*80)
        print(f"concat_layers: {self.concat_layers}")
        print(f"concat_all_features: {self.concat_all_features}\n")

        self.feature_groups = {}

        for model_name in self.model_list:
            if model_name not in self.models_dict:
                continue

            # Get all feature spaces for this model
            feature_spaces = self.models_dict[model_name]

            # Filter to only loaded features
            feature_spaces = [f for f in feature_spaces if f in self.loaded_features]

            if not feature_spaces:
                print(f"WARNING: No loaded features for model '{model_name}'")
                continue

            print(f"Model: {model_name} ({len(feature_spaces)} features)")

            if self.concat_all_features:
                # Concatenate ALL features within this model
                if self.concat_layers:
                    # First concat layers within each sub-model, then concat sub-models
                    grouped_features = self._group_features_by_base_model(feature_spaces)
                    model_parts = []

                    for base_model, layers in sorted(grouped_features.items()):
                        layer_features = [self.loaded_features[f] for f in layers]
                        if len(layer_features) > 1:
                            concatenated = np.concatenate(layer_features, axis=1)
                            print(f"  {base_model}: {len(layers)} layers → {concatenated.shape}")
                        else:
                            concatenated = layer_features[0]
                            print(f"  {base_model}: {concatenated.shape}")
                        model_parts.append(concatenated)

                    self.feature_groups[model_name] = np.concatenate(model_parts, axis=1)
                    print(f"  → Combined: {self.feature_groups[model_name].shape}")
                else:
                    # Concat all individual layers/features directly
                    all_features = [self.loaded_features[f] for f in feature_spaces]
                    self.feature_groups[model_name] = np.concatenate(all_features, axis=1)
                    print(f"  → Combined: {self.feature_groups[model_name].shape}")

            elif self.concat_layers:
                # One group per sub-model, layers concatenated
                grouped_features = self._group_features_by_base_model(feature_spaces)

                for base_model, layers in sorted(grouped_features.items()):
                    layer_features = [self.loaded_features[f] for f in layers]

                    if len(layer_features) > 1:
                        # Multiple layers to concatenate
                        concatenated = np.concatenate(layer_features, axis=1)
                        group_key = f"{model_name}_{base_model}" if len(grouped_features) > 1 else base_model
                        self.feature_groups[group_key] = concatenated
                        print(f"  {group_key}: {len(layers)} layers → {concatenated.shape}")
                    else:
                        # Single feature
                        group_key = f"{model_name}_{base_model}" if len(grouped_features) > 1 else base_model
                        self.feature_groups[group_key] = layer_features[0]
                        print(f"  {group_key}: {layer_features[0].shape}")

            else:
                # One group per individual feature/layer
                for feature_name in feature_spaces:
                    self.feature_groups[feature_name] = self.loaded_features[feature_name]
                    print(f"  {feature_name}: {self.loaded_features[feature_name].shape}")

        print(f"\nCreated {len(self.feature_groups)} feature groups for PCA analysis")
        print("="*80)

        return self.feature_groups

    def compute_pca(self):
        """
        Compute PCA for each feature group.

        For each group, standardizes features and fits PCA to estimate
        intrinsic dimensionality.

        Returns:
            dict: self.pca_results containing PCA objects and statistics
        """
        print("\n" + "="*80)
        print("COMPUTING PCA DIMENSIONALITY")
        print("="*80)
        print(f"Variance threshold: {self.variance_threshold*100}%\n")

        self.pca_results = {}

        for group_name, features in self.feature_groups.items():
            print(f"Analyzing {group_name}...")

            n_samples, n_features = features.shape
            print(f"  Input shape: {features.shape}")

            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Fit PCA
            n_components = min(n_samples, n_features)
            pca = PCA(n_components=n_components)
            pca.fit(features_scaled)

            # Calculate cumulative variance
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)

            # Find n_components for variance threshold
            n_components_threshold = np.argmax(cumulative_variance >= self.variance_threshold) + 1

            # Store results
            self.pca_results[group_name] = {
                'pca': pca,
                'scaler': scaler,
                'explained_variance_ratio': explained_variance_ratio,
                'cumulative_variance': cumulative_variance,
                'n_components_90': n_components_threshold,
                'original_dims': n_features,
                'n_samples': n_samples,
                'compression_ratio': n_components_threshold / n_features
            }

            print(f"  Original dims: {n_features}")
            print(f"  Dims for {self.variance_threshold*100}% variance: {n_components_threshold}")
            print(f"  Compression ratio: {n_components_threshold/n_features:.3f}")
            print(f"  Actual variance: {cumulative_variance[n_components_threshold-1]:.4f}\n")

        print(f"PCA analysis complete for {len(self.pca_results)} feature groups")
        print("="*80)

        return self.pca_results

    def plot_results(self, save_path=None, figsize=(12, 8), max_components=None):
        """
        Plot cumulative variance explained for all feature groups.

        Creates a single figure with all PCA results for easy comparison.

        Parameters:
            save_path (str, optional): Path to save figure. If None, displays plot
            figsize (tuple): Figure size (width, height)
            max_components (int, optional): Maximum number of components to plot.
                                           If None, automatically determined from data.

        Returns:
            matplotlib.figure.Figure: The created figure

        When plotting_helpers.figure_width_mm is set, the figure is saved as an editable
        pdf of that width (save_path's extension is replaced by .pdf).
        """
        if not self.pca_results:
            print("No PCA results to plot. Run compute_pca() first.")
            return None
        publication = plotting_helpers.figure_width_mm is not None
        _fs = plotting_helpers._fs
        plotting_helpers._update_font_rc(12)

        print("\n" + "="*80)
        print("PLOTTING RESULTS")
        print("="*80)

        fig, ax = plt.subplots(figsize=figsize)

        label_map = {
            'alexnet_eps0.3':    'AlexNet',
            'SimCLR_eps0.3':    'SimCLR',
            'motion':            'Motion',
            'hubert_chunk120':   'HuBERT',
            'word2vec_chunk120': 'word2vec',
            'GPT2_chunk120':     'GPT-2',
            'vision':            'AlexNet+Motion',
            'vision_simclr':      'SimCLR+Motion',
            'language':          'HuBERT+word2vec+GPT-2',
        }
        # only the groups that were analyzed are plotted (AlexNet or SimCLR as the image model)
        plot_order = ['alexnet_eps0.3', 'SimCLR_eps0.3', 'motion', 'hubert_chunk120',
                      'word2vec_chunk120', 'GPT2_chunk120',
                      'vision', 'vision_simclr', 'language']
        colors_dict = plotting_helpers.get_colors_dict()

        # Determine max components to show if not specified
        if max_components is None:
            # Find the maximum n_90 across all groups, add 20% buffer
            max_n90 = max(results['n_components_90'] for results in self.pca_results.values())
            max_components = int(max_n90 * 1.2)

        # Plot each feature group in the specified order
        source_rows = []  # the plotted curves, saved next to the figure
        for group_name in plot_order:
            if group_name not in self.pca_results or group_name not in label_map:
                continue
            results = self.pca_results[group_name]
            cumulative_var = results['cumulative_variance']
            n_90 = results['n_components_90']

            # Limit plotting to max_components for clarity
            plot_range = min(len(cumulative_var), max_components)

            is_combined = group_name in ('vision', 'vision_simclr', 'language')
            ax.plot(range(1, plot_range + 1),
                   cumulative_var[:plot_range],
                   label=f'{label_map[group_name]} (n={n_90})',
                   color=colors_dict[group_name],
                   linewidth=1 if publication else 2,
                   linestyle='-' if is_combined else '--')
            source_rows += [{'feature_space': label_map[group_name], 'n_components_90': n_90,
                             'n_components': i + 1, 'cumulative_variance_explained': cumulative_var[i]}
                            for i in range(plot_range)]

        # Add variance threshold line
        ax.axhline(y=self.variance_threshold,
                  color='black',
                  linestyle='--',
                  linewidth=0.75 if publication else 1.5,
                  label=f'{self.variance_threshold*100}% variance threshold')

        ax.set_xlabel('Number of Principal Components', fontsize=_fs(12))
        ax.set_ylabel('Cumulative Variance Explained', fontsize=_fs(12))
        ax.legend(loc='lower right', fontsize=_fs(9), frameon=not publication)
        ax.grid(True, alpha=0.3, linewidth=0.3 if publication else None)
        ax.set_xlim(1, max_components)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()

        if save_path and publication:
            save_path = os.path.splitext(save_path)[0] + '.pdf'
            plotting_helpers.save_figure(fig, save_path)
            print(f"Figure saved to: {save_path}")
        elif save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        else:
            plt.show()
        if save_path:
            source_path = os.path.splitext(save_path)[0] + '_source_data.csv'
            pd.DataFrame(source_rows).to_csv(source_path, index=False)
            print(f"Source data saved to: {source_path}")

        print("="*80)

        return fig

    def save_summary(self, save_path=None):
        """
        Create and save summary DataFrame of dimensionality results.

        Parameters:
            save_path (str, optional): Path to save CSV. If None, uses default location

        Returns:
            pd.DataFrame: Summary table
        """
        if not self.pca_results:
            print("No PCA results available. Run compute_pca() first.")
            return None

        print("\n" + "="*80)
        print("CREATING SUMMARY")
        print("="*80)

        summary_data = []

        for group_name, results in sorted(self.pca_results.items()):
            summary_data.append({
                'feature_group': group_name,
                'original_dims': results['original_dims'],
                'n_samples': results['n_samples'],
                'n_components_90': results['n_components_90'],
                'compression_ratio': results['compression_ratio'],
                'variance_at_90': results['cumulative_variance'][results['n_components_90']-1]
            })

        summary_df = pd.DataFrame(summary_data)

        # Display summary
        print("\nDimensionality Summary:")
        print(summary_df.to_string(index=False))

        # Save to CSV
        if save_path is None:
            save_path = os.path.join(self.dir, 'analysis', 'intrinsic_dimensionality_summary.csv')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        summary_df.to_csv(save_path, index=False)
        print(f"\nSummary saved to: {save_path}")
        print("="*80)

        return summary_df

    def save_pcs(self, save_dir=None, use_self_dims=True):
        """
        Save principal components as time series for each feature group.

        Transforms the original features to PC space and saves as CSV files,
        preserving the temporal structure.

        Parameters:
            save_dir (str, optional): Directory to save PC time series.
                                     If None, uses './features/pcs_intrinsic/'
            use_self_dims (bool): If True, each feature group uses its own n_components_90.
                                 If False, all groups use the same dimensionality (max n_components_90).

        Returns:
            dict: Information about saved files including paths and dimensions
        """
        if not self.pca_results:
            print("No PCA results available. Run compute_pca() first.")
            return None

        if not self.feature_groups:
            print("No feature groups available. Run organize_features() first.")
            return None

        print("\n" + "="*80)
        print("SAVING PRINCIPAL COMPONENTS AS TIME SERIES")
        print("="*80)
        print(f"Mode: {'Self-reference (each group uses own dims)' if use_self_dims else 'Unified dimensionality'}\n")

        # Set up save directory
        if save_dir is None:
            save_dir = os.path.join(self.dir, 'features', 'pcs_intrinsic')
        os.makedirs(save_dir, exist_ok=True)

        # Determine target dimensionality
        if use_self_dims:
            print("Each feature group will use its own optimal dimensionality.\n")
        else:
            # Find max n_components_90 across all groups
            max_n90 = max(results['n_components_90'] for results in self.pca_results.values())
            print(f"All feature groups will use unified dimensionality: {max_n90} components\n")

        saved_files = {}

        # For each feature group
        for group_name in sorted(self.feature_groups.keys()):
            if group_name not in self.pca_results:
                print(f"WARNING: {group_name} not in PCA results, skipping")
                continue

            print(f"Processing {group_name}...")

            # Get the feature data
            feature_data = self.feature_groups[group_name]

            # Get PCA object and scaler
            pca = self.pca_results[group_name]['pca']
            scaler = self.pca_results[group_name]['scaler']

            # Standardize the data using the SAME scaler from PCA fitting
            feature_standardized = scaler.transform(feature_data)

            # Transform to PC space
            feature_pcs = pca.transform(feature_standardized)

            # Determine number of components to save
            if use_self_dims:
                n_components = self.pca_results[group_name]['n_components_90']
            else:
                n_components = max_n90

            # Truncate or pad to target dimensionality
            n_components_available = feature_pcs.shape[1]
            if n_components_available < n_components:
                print(f"  WARNING: Model has only {n_components_available} components, " +
                      f"padding with zeros to {n_components}")
                # Pad with zeros if model has fewer components
                feature_pcs_final = np.zeros((feature_pcs.shape[0], n_components))
                feature_pcs_final[:, :n_components_available] = feature_pcs
            else:
                # Truncate to target dimensionality
                feature_pcs_final = feature_pcs[:, :n_components]

            # Save to CSV
            save_filename = f"{group_name}_pcs_{n_components}dims.csv"
            save_path = os.path.join(save_dir, save_filename)

            # Save without header or index
            pd.DataFrame(feature_pcs_final).to_csv(save_path, header=False, index=False)

            # Calculate variance explained
            variance_explained = self.pca_results[group_name]['cumulative_variance'][
                min(n_components, n_components_available) - 1
            ]

            # Store information
            saved_files[group_name] = {
                'path': save_path,
                'n_components': n_components,
                'original_dims': self.pca_results[group_name]['original_dims'],
                'variance_explained': variance_explained,
                'n_samples': feature_data.shape[0]
            }

            print(f"  Original dims: {saved_files[group_name]['original_dims']}")
            print(f"  Saved {n_components} PCs")
            print(f"  Variance explained: {variance_explained:.4f}")
            print(f"  Saved to: {save_filename}\n")

        print("="*80)
        print(f"Successfully saved {len(saved_files)} PC time series files")
        print(f"Location: {save_dir}")
        print("="*80)

        # Create summary CSV
        summary_rows = []
        for group_name, info in saved_files.items():
            summary_rows.append({
                'group': group_name,
                'filename': os.path.basename(info['path']),
                'n_components': info['n_components'],
                'original_dims': info['original_dims'],
                'variance_explained': info['variance_explained'],
                'compression_ratio': info['n_components'] / info['original_dims'],
                'n_samples': info['n_samples']
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(save_dir, 'pcs_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: pcs_summary.csv\n")

        return saved_files

    def subsample_analysis(self, subsample_fractions=[0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
                          n_iterations=10, variance_threshold=None, random_seed=42):
        """
        Subsample features to test if dimensionality depends on sample size.

        This analysis tests whether intrinsic dimensionality is robust to sample size reduction,
        or whether it depends on having many samples.

        Parameters:
            subsample_fractions (list): Fractions of samples to retain (e.g., [0.15, 0.30, 0.60])
            n_iterations (int): Number of bootstrap iterations per fraction (default: 10)
            variance_threshold (float, optional): Variance threshold for dimensionality
                                                 (default: use self.variance_threshold)
            random_seed (int): Random seed for reproducibility (default: 42)

        Returns:
            dict: Results containing subsample curves for each feature group
        """
        if not hasattr(self, 'feature_groups') or not self.feature_groups:
            raise ValueError("Must run organize_features() first")

        if variance_threshold is None:
            variance_threshold = self.variance_threshold

        print("\n" + "="*80)
        print("SUBSAMPLING ANALYSIS")
        print("="*80)
        print(f"Testing how dimensionality changes with sample size")
        print(f"Subsample fractions: {subsample_fractions}")
        print(f"Bootstrap iterations: {n_iterations}")
        print(f"Variance threshold: {variance_threshold*100}%\n")

        np.random.seed(random_seed)

        # Initialize results storage
        self.subsample_results = {}

        # For each feature group
        for group_name, feature_data in self.feature_groups.items():
            print(f"\nSubsampling {group_name}...")

            n_samples, n_features = feature_data.shape
            print(f"  Total samples: {n_samples}")
            print(f"  Features: {n_features}")

            # Storage for this group's results
            fraction_results = {}

            # For each subsample fraction
            for frac in subsample_fractions:
                n_subsample = int(n_samples * frac)

                if n_subsample < 2:
                    print(f"  Fraction {frac:.0%}: Skipping (too few samples: {n_subsample})")
                    continue

                print(f"  Fraction {frac:.0%}: {n_subsample} samples", end='')

                # Storage for bootstrap iterations
                n_components_list = []
                cumulative_var_list = []

                # Bootstrap iterations
                for iter_idx in range(n_iterations):
                    # Randomly select subsample of indices
                    subsample_indices = np.random.choice(n_samples, size=n_subsample, replace=False)
                    feature_subsample = feature_data[subsample_indices]

                    # Standardize
                    scaler = StandardScaler()
                    data_standardized = scaler.fit_transform(feature_subsample)

                    # Fit PCA
                    n_components = min(n_subsample, n_features)
                    pca = PCA(n_components=n_components)
                    pca.fit(data_standardized)

                    # Get cumulative variance
                    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

                    # Find n_components for threshold
                    n_components_threshold = np.argmax(cumulative_variance >= variance_threshold) + 1

                    n_components_list.append(n_components_threshold)
                    cumulative_var_list.append(cumulative_variance)

                # Compute mean and std across iterations
                mean_n_components = np.mean(n_components_list)
                std_n_components = np.std(n_components_list)

                # Average cumulative variance curves
                # Pad shorter curves with 1.0 to make them same length
                max_len = max(len(cv) for cv in cumulative_var_list)
                padded_curves = []
                for cv in cumulative_var_list:
                    if len(cv) < max_len:
                        padded = np.concatenate([cv, np.ones(max_len - len(cv))])
                    else:
                        padded = cv
                    padded_curves.append(padded)

                mean_cumulative_var = np.mean(padded_curves, axis=0)
                std_cumulative_var = np.std(padded_curves, axis=0)

                fraction_results[frac] = {
                    'n_subsample': n_subsample,
                    'mean_n_components_90': mean_n_components,
                    'std_n_components_90': std_n_components,
                    'mean_cumulative_var': mean_cumulative_var,
                    'std_cumulative_var': std_cumulative_var,
                    'compression_ratio': mean_n_components / n_features
                }

                print(f" → {mean_n_components:.1f} ± {std_n_components:.1f} components")

            self.subsample_results[group_name] = fraction_results

        print("\n" + "="*80)
        print("Subsampling analysis complete!")
        print("="*80)

        return self.subsample_results

    def plot_subsample_analysis(self, save_path=None, figsize=(14, 8)):
        """
        Plot the results of subsampling analysis.

        Creates a plot showing how dimensionality (n_components for variance threshold)
        changes as a function of sample size for all feature groups.

        Parameters:
            save_path (str, optional): Path to save the figure
            figsize (tuple): Figure size (width, height)

        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if not hasattr(self, 'subsample_results'):
            raise ValueError("Must run subsample_analysis() first")

        import seaborn as sns

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        sns.set_style("whitegrid")

        # Get all feature groups
        all_groups = sorted(self.subsample_results.keys())
        colors = sns.color_palette("husl", n_colors=len(all_groups))

        for group_idx, group_name in enumerate(all_groups):
            results = self.subsample_results[group_name]

            if not results:  # Skip if no results for this group
                continue

            # Extract data
            fractions = sorted(results.keys())
            n_samples = [results[frac]['n_subsample'] for frac in fractions]
            mean_dims = [results[frac]['mean_n_components_90'] for frac in fractions]
            std_dims = [results[frac]['std_n_components_90'] for frac in fractions]

            # Plot with error bars
            ax.errorbar(n_samples, mean_dims, yerr=std_dims,
                       label=group_name,
                       color=colors[group_idx],
                       linewidth=2, marker='o', markersize=6, capsize=5, alpha=0.8)

        ax.set_xlabel('Number of Samples', fontsize=14, fontweight='normal')
        ax.set_ylabel(f'Dims for {self.variance_threshold*100:.0f}% Variance', fontsize=14, fontweight='normal')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best', framealpha=0.9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Subsample analysis figure saved to: {save_path}")

        plt.show()

        return fig

    def plot_pc_trajectories(self, save_path=None, figsize=None, cmap='viridis', alpha=0.6, s=20):
        """
        Plot PC1 vs PC2 for each feature group, colored by timepoint.

        Creates scatter plots showing the trajectory through PC space over time,
        with color indicating temporal progression.

        Parameters:
            save_path (str, optional): Path to save the figure
            figsize (tuple, optional): Figure size (width, height).
                                      If None, auto-calculated based on number of groups
            cmap (str): Colormap for timepoints (default: 'viridis')
            alpha (float): Transparency of points (0-1, default: 0.6)
            s (float): Size of scatter points (default: 20)

        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if not self.pca_results:
            print("No PCA results available. Run compute_pca() first.")
            return None

        if not self.feature_groups:
            print("No feature groups available. Run organize_features() first.")
            return None

        print("\n" + "="*80)
        print("PLOTTING PC TRAJECTORIES (PC1 vs PC2, colored by time)")
        print("="*80)

        # Determine subplot layout
        n_groups = len(self.pca_results)
        n_cols = min(3, n_groups)  # Max 3 columns
        n_rows = int(np.ceil(n_groups / n_cols))

        # Auto-calculate figure size if not provided
        if figsize is None:
            figsize = (6 * n_cols, 5 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        # For each feature group
        for idx, (group_name, results) in enumerate(sorted(self.pca_results.items())):
            ax = axes[idx]

            # Get the feature data and PCA transform
            feature_data = self.feature_groups[group_name]
            pca = results['pca']
            scaler = results['scaler']

            # Transform to PC space
            feature_standardized = scaler.transform(feature_data)
            feature_pcs = pca.transform(feature_standardized)

            # Get PC1 and PC2
            pc1 = feature_pcs[:, 0]
            pc2 = feature_pcs[:, 1]

            # Create timepoint array for coloring
            n_timepoints = len(pc1)
            timepoints = np.arange(n_timepoints)

            # Create scatter plot
            scatter = ax.scatter(pc1, pc2, c=timepoints, cmap=cmap, alpha=alpha, s=s)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Timepoint', fontsize=10)

            # Get variance explained by PC1 and PC2
            var_pc1 = results['explained_variance_ratio'][0] * 100
            var_pc2 = results['explained_variance_ratio'][1] * 100

            # Labels and title
            ax.set_xlabel(f'PC1 ({var_pc1:.1f}% var)', fontsize=11, fontweight='normal')
            ax.set_ylabel(f'PC2 ({var_pc2:.1f}% var)', fontsize=11, fontweight='normal')
            ax.grid(True, alpha=0.3)

            print(f"{group_name}: PC1={var_pc1:.2f}%, PC2={var_pc2:.2f}%, n_points={n_timepoints}")

        # Hide unused subplots
        for idx in range(n_groups, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPC trajectories figure saved to: {save_path}")

        plt.show()

        print("="*80)
        return fig

    # TEMPORARY METHOD - May delete later
    def correlate_pcs_with_annotated(self, n_pcs=10, method='pearson', save_path=None):
        """
        TEMPORARY: Correlate PCs with annotated features (human-labeled stimulus attributes).

        For each feature group, computes correlations between the first n PCs and
        all annotated features (pixel, hue, face, social, valence, etc.).

        Parameters:
            n_pcs (int): Number of PCs to correlate (default: 10)
            method (str): Correlation method - 'pearson', 'spearman', or 'kendall' (default: 'pearson')
            save_path (str, optional): Path to save correlation matrices as CSV

        Returns:
            dict: Correlation results for each feature group
                  Keys: feature group names
                  Values: dict with 'correlations' (DataFrame) and 'p_values' (DataFrame)
        """
        if not self.pca_results:
            print("No PCA results available. Run compute_pca() first.")
            return None

        if not self.feature_groups:
            print("No feature groups available. Run organize_features() first.")
            return None

        print("\n" + "="*80)
        print("CORRELATING PCs WITH ANNOTATED FEATURES [TEMPORARY]")
        print("="*80)
        print(f"Method: {method}")
        print(f"Number of PCs: {n_pcs}\n")

        # Load annotated features
        print("Loading annotated features...")
        annotated_features = self.models_dict['annotated']
        annotated_data = {}

        for feature_name in annotated_features:
            if feature_name not in self.features_dict:
                print(f"  WARNING: Annotated feature '{feature_name}' not in features_dict, skipping")
                continue

            csv_filename = self.features_dict[feature_name]
            filepath = os.path.join(self.features_dir, f'{csv_filename}.csv')

            try:
                data = pd.read_csv(filepath, header=None).values.astype('float32')
                # If multi-dimensional, take mean across features (e.g., for pixel/hue)
                if data.ndim > 1 and data.shape[1] > 1:
                    data = data.mean(axis=1, keepdims=True)
                annotated_data[feature_name] = data.flatten()
                print(f"  Loaded {feature_name}: {len(annotated_data[feature_name])} timepoints")
            except FileNotFoundError:
                print(f"  WARNING: File not found for '{feature_name}', skipping")

        print(f"\nLoaded {len(annotated_data)} annotated features")

        # Add timepoint as a feature (temporal progression)
        # Note: Timepoint will be added per feature group based on its length
        print("Note: Timepoint will be added as a feature for correlation")
        print("="*80)

        # Storage for results
        correlation_results = {}

        # For each feature group
        for group_name, results in sorted(self.pca_results.items()):
            print(f"\nAnalyzing {group_name}...")

            # Get the feature data and transform to PC space
            feature_data = self.feature_groups[group_name]
            pca = results['pca']
            scaler = results['scaler']

            # Transform to PC space
            feature_standardized = scaler.transform(feature_data)
            feature_pcs = pca.transform(feature_standardized)

            # Limit to n_pcs
            n_pcs_available = min(n_pcs, feature_pcs.shape[1])
            feature_pcs_subset = feature_pcs[:, :n_pcs_available]

            print(f"  Using {n_pcs_available} PCs")
            print(f"  Timepoints: {feature_pcs_subset.shape[0]}")

            # Check length compatibility
            n_timepoints = feature_pcs_subset.shape[0]
            annotated_compatible = {
                name: data for name, data in annotated_data.items()
                if len(data) == n_timepoints
            }

            # Add timepoint as a special feature (temporal progression)
            annotated_compatible['timepoint'] = np.arange(n_timepoints).astype('float32')

            if len(annotated_compatible) == 0:
                print(f"  WARNING: No annotated features match timepoint length {n_timepoints}, skipping")
                continue

            print(f"  Compatible annotated features: {len(annotated_compatible)} (including timepoint)")

            # Compute correlations
            from scipy.stats import pearsonr, spearmanr, kendalltau

            corr_func = {
                'pearson': pearsonr,
                'spearman': spearmanr,
                'kendall': kendalltau
            }[method]

            # Initialize correlation matrix and p-value matrix
            corr_matrix = np.zeros((n_pcs_available, len(annotated_compatible)))
            pval_matrix = np.zeros((n_pcs_available, len(annotated_compatible)))

            for pc_idx in range(n_pcs_available):
                pc_data = feature_pcs_subset[:, pc_idx]

                for annot_idx, (annot_name, annot_data) in enumerate(sorted(annotated_compatible.items())):
                    corr, pval = corr_func(pc_data, annot_data)
                    corr_matrix[pc_idx, annot_idx] = corr
                    pval_matrix[pc_idx, annot_idx] = pval

            # Convert to DataFrames
            pc_labels = [f'PC{i+1}' for i in range(n_pcs_available)]
            annot_labels = sorted(annotated_compatible.keys())

            corr_df = pd.DataFrame(corr_matrix, index=pc_labels, columns=annot_labels)
            pval_df = pd.DataFrame(pval_matrix, index=pc_labels, columns=annot_labels)

            # Store results
            correlation_results[group_name] = {
                'correlations': corr_df,
                'p_values': pval_df,
                'n_pcs': n_pcs_available,
                'n_annotated': len(annotated_compatible)
            }

            # Find strongest correlations
            abs_corr = np.abs(corr_matrix)
            max_corr_idx = np.unravel_index(np.argmax(abs_corr), abs_corr.shape)
            max_corr_pc = pc_labels[max_corr_idx[0]]
            max_corr_feature = annot_labels[max_corr_idx[1]]
            max_corr_value = corr_matrix[max_corr_idx]
            max_corr_pval = pval_matrix[max_corr_idx]

            print(f"  Strongest correlation: {max_corr_pc} <-> {max_corr_feature}")
            print(f"    r = {max_corr_value:.3f}, p = {max_corr_pval:.4e}")

        print("\n" + "="*80)
        print(f"Correlation analysis complete for {len(correlation_results)} feature groups")
        print("="*80)

        # Save results if path provided
        if save_path and correlation_results:
            save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
            os.makedirs(save_dir, exist_ok=True)

            for group_name, results_dict in correlation_results.items():
                # Save correlation matrix
                corr_filename = save_path.replace('.csv', f'_{group_name}_correlations.csv')
                results_dict['correlations'].to_csv(corr_filename)
                print(f"Saved correlations for {group_name}: {corr_filename}")

                # Save p-values
                pval_filename = save_path.replace('.csv', f'_{group_name}_pvalues.csv')
                results_dict['p_values'].to_csv(pval_filename)
                print(f"Saved p-values for {group_name}: {pval_filename}")

        return correlation_results

    def plot_pc_annotated_correlations(self, correlation_results=None, n_pcs=10,
                                       save_path=None, figsize=None, cmap='RdBu_r',
                                       vmin=-1, vmax=1, annotate_significant=True,
                                       p_threshold=0.05):
        """
        TEMPORARY: Plot heatmaps of PC correlations with annotated features.

        Parameters:
            correlation_results (dict, optional): Results from correlate_pcs_with_annotated().
                                                 If None, runs the correlation first.
            n_pcs (int): Number of PCs to include (default: 10)
            save_path (str, optional): Path to save the figure
            figsize (tuple, optional): Figure size. If None, auto-calculated.
            cmap (str): Colormap for heatmap (default: 'RdBu_r')
            vmin, vmax (float): Color scale limits (default: -1, 1)
            annotate_significant (bool): If True, mark significant correlations with asterisks
            p_threshold (float): P-value threshold for significance (default: 0.05)

        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        # Run correlation if not provided
        if correlation_results is None:
            print("Running correlation analysis first...")
            correlation_results = self.correlate_pcs_with_annotated(n_pcs=n_pcs)

        if not correlation_results:
            print("No correlation results available.")
            return None

        print("\n" + "="*80)
        print("PLOTTING PC-ANNOTATED CORRELATIONS [TEMPORARY]")
        print("="*80)

        import seaborn as sns

        # Determine subplot layout
        n_groups = len(correlation_results)
        n_cols = min(2, n_groups)  # Max 2 columns for heatmaps
        n_rows = int(np.ceil(n_groups / n_cols))

        # Auto-calculate figure size if not provided
        if figsize is None:
            figsize = (8 * n_cols, 6 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        # For each feature group
        for idx, (group_name, results_dict) in enumerate(sorted(correlation_results.items())):
            ax = axes[idx]

            corr_df = results_dict['correlations']
            pval_df = results_dict['p_values']

            # Create heatmap
            sns.heatmap(corr_df, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                       center=0, annot=False, fmt='.2f',
                       cbar_kws={'label': 'Correlation'},
                       linewidths=0.5, linecolor='gray')

            # Annotate significant correlations
            if annotate_significant:
                for i in range(corr_df.shape[0]):
                    for j in range(corr_df.shape[1]):
                        if pval_df.iloc[i, j] < p_threshold:
                            # Add asterisk for significant correlations
                            ax.text(j + 0.5, i + 0.5, '*',
                                   ha='center', va='center',
                                   fontsize=16, fontweight='normal',
                                   color='black')

            ax.set_xlabel('Annotated Features', fontsize=11, fontweight='normal')
            ax.set_ylabel('Principal Components', fontsize=11, fontweight='normal')

            # Rotate x-axis labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # Hide unused subplots
        for idx in range(n_groups, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nCorrelation heatmap saved to: {save_path}")

        plt.show()

        print("="*80)
        return fig

    def _group_feature_groups_by_model(self, feature_group_names):
        """
        Group feature group names by their base model.

        Takes a list of feature group names and groups them by their base model identifier.
        For example: ['alexnet_layer1', 'alexnet_layer2', 'hubert_layer1'] becomes
        {'alexnet': ['alexnet_layer1', 'alexnet_layer2'], 'hubert': ['hubert_layer1']}

        Parameters:
            feature_group_names (list): List of feature group names to group

        Returns:
            dict: Dictionary mapping base model names to lists of feature group names
        """
        model_groups = {}

        for group_name in feature_group_names:
            # Parse the base model name
            # Common patterns: 'alexnet_layer1', 'hubert_chunk60_layer5', 'GPT2_chunk60_layer3', etc.
            base_model = None

            if 'alexnet' in group_name.lower():
                base_model = 'alexnet'
            elif 'hubert' in group_name.lower():
                # Extract chunk info if present
                if 'chunk' in group_name.lower():
                    # e.g., hubert_chunk60_layer5 -> hubert_chunk60
                    parts = group_name.split('_')
                    chunk_parts = [p for p in parts if 'chunk' in p.lower()]
                    if chunk_parts:
                        base_model = f"hubert_{chunk_parts[0]}"
                    else:
                        base_model = 'hubert'
                else:
                    base_model = 'hubert'
            elif 'gpt2' in group_name.lower():
                # Extract chunk info if present
                if 'chunk' in group_name.lower():
                    parts = group_name.split('_')
                    chunk_parts = [p for p in parts if 'chunk' in p.lower()]
                    if chunk_parts:
                        base_model = f"GPT2_{chunk_parts[0]}"
                    else:
                        base_model = 'GPT2'
                else:
                    base_model = 'GPT2'
            elif 'simclr' in group_name.lower():
                if 'attention' in group_name.lower():
                    base_model = 'SimCLR_attention'
                elif 'embedding' in group_name.lower():
                    base_model = 'SimCLR_embedding'
                else:
                    base_model = 'SimCLR'
            elif 'slip' in group_name.lower():
                if 'attention' in group_name.lower():
                    base_model = 'SLIP_attention'
                elif 'embedding' in group_name.lower():
                    base_model = 'SLIP_embedding'
                else:
                    base_model = 'SLIP'
            elif 'word2vec' in group_name.lower():
                base_model = 'word2vec'
            elif 'motion' in group_name.lower():
                base_model = 'motion'
            else:
                # Fallback: use the group name itself
                base_model = group_name

            # Add to appropriate group
            if base_model not in model_groups:
                model_groups[base_model] = []
            model_groups[base_model].append(group_name)

        return model_groups

    def plot_pc_timeseries(self, n_pcs=2, save_path=None, figsize=None,
                           group_by_model=True, show_variance=True):
        """
        Plot time series of principal components for each feature group.

        Creates line plots showing how PC values change over time. When group_by_model=True,
        all layers from the same model are displayed as subplots in a single figure.

        Parameters:
            n_pcs (int): Number of principal components to plot (default: 2)
            save_path (str, optional): Path to save figure. If None, displays plot.
                                      When group_by_model=True, saves one figure per model
                                      with '_modelname' appended to the filename.
            figsize (tuple, optional): Figure size (width, height). If None, auto-calculated.
            group_by_model (bool): If True, group layers from same model into one figure.
                                  If False, create separate figure for each layer. (default: True)
            show_variance (bool): If True, include variance explained % in legend labels (default: True)

        Returns:
            dict: Dictionary mapping model names to matplotlib figures (if group_by_model=True)
                  or feature group names to figures (if group_by_model=False)
        """
        if not self.pca_results:
            print("ERROR: No PCA results available. Run compute_pca() first.")
            return None

        if not self.feature_groups:
            print("ERROR: No feature groups available. Run organize_features() first.")
            return None

        print("\n" + "="*80)
        print(f"PLOTTING PC TIME SERIES ({n_pcs} components)")
        print("="*80)
        print(f"Group by model: {group_by_model}")
        print(f"Show variance: {show_variance}\n")

        figures = {}

        if group_by_model:
            # Group feature groups by base model
            model_groups = self._group_feature_groups_by_model(list(self.pca_results.keys()))

            # Create one figure per model
            for model_name, group_names in sorted(model_groups.items()):
                print(f"\nPlotting {model_name} ({len(group_names)} layers)...")

                # Determine subplot layout
                n_layers = len(group_names)
                n_cols = min(3, n_layers)  # Max 3 columns
                n_rows = int(np.ceil(n_layers / n_cols))

                # Auto-calculate figure size if not provided
                if figsize is None:
                    fig_width = 6 * n_cols
                    fig_height = 4 * n_rows
                    current_figsize = (fig_width, fig_height)
                else:
                    current_figsize = figsize

                fig, axes = plt.subplots(n_rows, n_cols, figsize=current_figsize, squeeze=False)
                axes = axes.flatten()

                # Plot each layer
                for idx, group_name in enumerate(sorted(group_names)):
                    ax = axes[idx]

                    # Get data
                    if group_name not in self.feature_groups:
                        print(f"  WARNING: {group_name} not in feature_groups, skipping")
                        continue

                    feature_data = self.feature_groups[group_name]
                    results = self.pca_results[group_name]
                    pca = results['pca']
                    scaler = results['scaler']

                    # Check if enough PCs are available
                    n_available_pcs = pca.n_components_
                    if n_pcs > n_available_pcs:
                        print(f"  WARNING: {group_name} only has {n_available_pcs} PCs, plotting all available")
                        n_to_plot = n_available_pcs
                    else:
                        n_to_plot = n_pcs

                    # Transform to PC space
                    feature_standardized = scaler.transform(feature_data)
                    feature_pcs = pca.transform(feature_standardized)

                    # Create timepoint array
                    n_timepoints = feature_data.shape[0]
                    timepoints = np.arange(n_timepoints)

                    # Plot each PC
                    for pc_idx in range(n_to_plot):
                        pc_values = feature_pcs[:, pc_idx]
                        var_explained = results['explained_variance_ratio'][pc_idx] * 100

                        if show_variance:
                            label = f'PC{pc_idx+1} ({var_explained:.1f}%)'
                        else:
                            label = f'PC{pc_idx+1}'

                        ax.plot(timepoints, pc_values, label=label, linewidth=1.5, alpha=0.8)

                    # Formatting
                    ax.set_xlabel('Timepoint', fontsize=10)
                    ax.set_ylabel('PC Value (standardized)', fontsize=10)
                    ax.legend(loc='best', fontsize=9)
                    ax.grid(True, alpha=0.3)

                    print(f"  {group_name}: {n_timepoints} timepoints, {n_to_plot} PCs plotted")

                # Hide unused subplots
                for idx in range(n_layers, len(axes)):
                    axes[idx].axis('off')

                plt.tight_layout()

                # Save or show
                if save_path:
                    # Create model-specific filename
                    import os
                    base_path, ext = os.path.splitext(save_path)
                    model_save_path = f"{base_path}_{model_name}{ext}"
                    plt.savefig(model_save_path, dpi=300, bbox_inches='tight')
                    print(f"\n  Saved: {model_save_path}")

                figures[model_name] = fig

        else:
            # Create separate figure for each feature group
            for group_name in sorted(self.pca_results.keys()):
                print(f"\nPlotting {group_name}...")

                # Get data
                feature_data = self.feature_groups[group_name]
                results = self.pca_results[group_name]
                pca = results['pca']
                scaler = results['scaler']

                # Check if enough PCs are available
                n_available_pcs = pca.n_components_
                if n_pcs > n_available_pcs:
                    print(f"  WARNING: Only {n_available_pcs} PCs available, plotting all")
                    n_to_plot = n_available_pcs
                else:
                    n_to_plot = n_pcs

                # Transform to PC space
                feature_standardized = scaler.transform(feature_data)
                feature_pcs = pca.transform(feature_standardized)

                # Create timepoint array
                n_timepoints = feature_data.shape[0]
                timepoints = np.arange(n_timepoints)

                # Create figure
                if figsize is None:
                    current_figsize = (12, 4)
                else:
                    current_figsize = figsize

                fig, ax = plt.subplots(1, 1, figsize=current_figsize)

                # Plot each PC
                for pc_idx in range(n_to_plot):
                    pc_values = feature_pcs[:, pc_idx]
                    var_explained = results['explained_variance_ratio'][pc_idx] * 100

                    if show_variance:
                        label = f'PC{pc_idx+1} ({var_explained:.1f}%)'
                    else:
                        label = f'PC{pc_idx+1}'

                    ax.plot(timepoints, pc_values, label=label, linewidth=1.5, alpha=0.8)

                # Formatting
                ax.set_xlabel('Timepoint', fontsize=11)
                ax.set_ylabel('PC Value (standardized)', fontsize=11)
                ax.legend(loc='best', fontsize=10)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()

                # Save or show
                if save_path:
                    import os
                    base_path, ext = os.path.splitext(save_path)
                    group_save_path = f"{base_path}_{group_name}{ext}"
                    plt.savefig(group_save_path, dpi=300, bbox_inches='tight')
                    print(f"  Saved: {group_save_path}")

                figures[group_name] = fig

                print(f"  {n_timepoints} timepoints, {n_to_plot} PCs plotted")

        print("\n" + "="*80)
        print(f"PC time series plotting complete ({len(figures)} figures created)")
        print("="*80)

        if not save_path:
            plt.show()

        return figures

    def run_analysis(self, plot_save_path=None, summary_save_path=None, save_pcs=False, pcs_save_dir=None, use_self_dims=True):
        """
        Run the complete analysis pipeline.

        Convenience method that executes:
        1. Load features
        2. Organize features
        3. Compute PCA
        4. Plot results
        5. Save summary
        6. Optionally save PC time series

        Parameters:
            plot_save_path (str, optional): Path to save plot
            summary_save_path (str, optional): Path to save summary CSV
            save_pcs (bool): If True, save PC time series to CSV files
            pcs_save_dir (str, optional): Directory to save PC files
            use_self_dims (bool): If True, each group uses its own n_components_90

        Returns:
            tuple: (pca_results, summary_df, figure, saved_pcs_info)
                  saved_pcs_info will be None if save_pcs=False
        """
        print("\n" + "="*80)
        print("RUNNING FULL INTRINSIC DIMENSIONALITY ANALYSIS")
        print("="*80)

        # Run pipeline
        self.load_features()
        self.organize_features()
        self.compute_pca()
        fig = self.plot_results(save_path=plot_save_path)
        summary_df = self.save_summary(save_path=summary_save_path)

        # Optionally save PC time series
        saved_pcs_info = None
        if save_pcs:
            saved_pcs_info = self.save_pcs(save_dir=pcs_save_dir, use_self_dims=use_self_dims)

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)

        return self.pca_results, summary_df, fig, saved_pcs_info
