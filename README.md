# ubiquitous-vis
 Code for paper 'Ubiquitous cortical sensitivity to visual information during naturalistic, audiovisual movie viewing' 

 ### conda environment
 create the environment: ``` conda env create -f environment.yml ```

 activate environment: ``` conda activate ubiquitous-vis ```

 install custom src code (while in project directory): ``` pip install . ```

 ### data and analysis outputs
Raw and preprocessed (fMRIprep) fMRI files are available on OpenNeuro.

The outputs of first and second level analyses are available on OSF. The first level outputs include ``` Brain2Brain ``` (intersubject correlation and cross-subject encoding), ``` EncodingModel ```, and ``` GLM ```. To run second level analyses from scratch, these need to be downloaded and put into ``` /analysis ```. 

Precomputed feature space similarity results are in ``` /analysis/FeatureSpaceCorrelation ```.

The minimum required outputs of second level analyses for plotting are in this repository under ``` /analysis/SecondLevelGroup ``` and ``` /analysis/SecondLevelIndividual ```. Additional interim output files can be downloaded from OSF. 

### running code
See the sbatch scripts in ``` /scripts ``` to run the first level analyses. The joint encoding model took about 3 hours on a a100 gpu per subject when saving fitted weights.

The second level analyses can be run in ``` /scripts/SecondLevel Analyses.ipynb ```. To run these you will need to download some data from OSF. The notebook contains details on which data to download for your specific needs!

Plotting code for all main and supplemental figures is in ``` /scripts/Plot Figures.ipynb ```. This can be run if you clone this repository. All figures are created in ``` /figures ```. 

