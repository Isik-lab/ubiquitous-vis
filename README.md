# ubiquitous-vis
 Code for paper 'Ubiquitous cortical sensitivity to visual information during naturalistic, audiovisual movie viewing' 

 ### conda environment
 create the environment: ``` conda env create -f environment.yml ```
 activate environment ``` conda activate ubiquitous-vis ```
 install custom src code (while in project directory) ``` pip install . ```

 ### data and analysis outputs
Raw and preprocessed (fMRIprep) fMRI files are available on OpenNeuro.

The outputs of first and second level analyses are available on OSF. The first level outputs include ``` Brain2Brain ``` (intersubject correlation and cross-subject encoding), ``` EncodingModel ```, and ``` GLM ```. To run second level analyses from scratch, these need to be downloaded and put into ``` /analysis ```. 

Precomputed feature space similarity results are in ``` /analysis/FeatureSpaceCorrelation ```.

The minimum required outputs of second level analyses for plotting are in this repository under ``` /analysis/SecondLevelGroup ``` and ``` /analysis/SecondLevelIndividual ```. Additional interim output files can be downloaded from OSF. 

### running code
See the sbatch scripts in ``` /scripts ``` to run the first level analyses.

The second level analyses can be run in ``` /scripts/SecondLevel Analyses.ipynb ```. 

Plotting code is in ``` /scripts/Plot Figures.ipynb ```. 

All precomputed second level outputs and subfigures are available on OSF. The outputs in this repository are sufficient to plot all figures. To do this, just run Setup and then Main Figures and Supplemental Figures. All figures are created in ``` /figures ```. 

If desired, the second level analyses can be run from scratch in the notebook by setting ``` generate_subfigures = True ``` and ``` load = False ``` in the Setup portion. However, in order for these to run, you must download the first and second level outputs (see data above). The easiest way to do this is to replace the ``` analysis ``` directory with the downloaded version from OSF. 
