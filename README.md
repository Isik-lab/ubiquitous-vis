# ubiquitous-vis
 Code for paper 'Ubiquitous cortical sensitivity to visual information during naturalistic, audiovisual movie viewing' 

 ### conda environment
 ubiquitous-vis: ``` environment.yml ```
 
 ### data and analysis outputs
Preprocessed fMRI data and output of first level analyses (GLM and encoding models) are available on OpenNeuro. The first level outputs include ``` Brain2Brain ``` (intersubject correlation and cross-subject encoding), ``` EncodingModel ```, and ``` GLM ```. To run second level analyses from scratch, these need to be downloaded and put into ``` /analysis ```.

Precomputed feature space similarity results are in ``` /analysis/FeatureSpaceCorrelation ```.

Precomputed outputs of second level analyses are in this repository under ``` /analysis/SecondLevelGroup ``` and ``` /analysis/SecondLevelIndividual ```. 

### running code
See the sbatch scripts in ``` /scripts ``` to run the first level analyses.

The second level analyses and plotting code are in ``` /scripts/SecondLevel Analyses and Plot figures.ipynb ```

All of the second level outputs and subfigures are precomputed and available in this repository. These are used to plot all figures. To do this, just run Setup and then Main Figures and Supplemental Figures. All figures are created in ``` /figures ```. 

If desired, the second level analyses can be run from scratch in the notebook by setting ``` generate_subfigures = True ``` and ``` load = False ``` in the Setup portion. However, in order for these to run, you must download the first level outputs (see data above) and put them in  ``` /analysis ```. 
