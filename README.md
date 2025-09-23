# ubiquitous-vis
 Code for paper 'Ubiquitous cortical sensitivity to visual information during naturalistic, audiovisual movie viewing'. [Read paper here!](https://osf.io/preprints/psyarxiv/b5p4n_v1)

 ### conda environment
 create the environment: ``` conda env create -f environment.yml ```

 activate environment: ``` conda activate ubiquitous-vis ```

 install custom src code (while in project directory): ``` pip install . ```

 ### data and analysis outputs
Raw fMRI files will be [available on OpenNeuro](https://openneuro.org/datasets/ds006677) at a later date.

The annotated and extracted features used in the encoding model and the outputs of first (encoding model and glms) and second level analyses are [available on OSF](https://osf.io/5zjae/). To run the second level analyses from scratch (see code), both ```analysis``` and ```features``` need to be downloaded.

Precomputed feature space similarity results are in ``` /analysis/FeatureSpaceCorrelation ```.

The minimum required files for plotting are in this repository under ``` /analysis/SecondLevelGroup ``` and ``` /analysis/SecondLevelIndividual ```. Additional interim output files are available on [OSF](https://osf.io/5zjae/).

### running code
See the sbatch scripts in ``` /scripts ``` for preprocessing (fMRIprep) and running the first level analyses. The encoding models took about 3 hours on an a100 gpu per subject when saving the fitted weights.

The second level analyses can be run in ``` /scripts/SecondLevel Analyses.ipynb ```. To run these you will need to download data from [OSF](https://osf.io/5zjae/). The notebook contains details on which data to download for your specific needs!

Plotting code for all main and supplemental figures is in ``` /scripts/Plot Figures.ipynb ```. This can be run if you clone this repository. All figures are created in ``` /figures ```. 
