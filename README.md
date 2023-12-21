# Protein Fitness Prediction - ML4Science - EPFL 2023

Welcome to our ML4science project in collaboration with the [Laboratory of the Physics and Biological Systems at EPFL](https://www.epfl.ch/labs/lpbs/). The aim of this project is to predict the amino acid sequence of a protein based on a desired fitness value. For our case, we want to maximize the activity of the EL222 optoprotein when exposed to light while minimizing it when in dark conditions. This protein can be used for optogenetic techniques.

This project is built upon the paper ['Learning protein fitness models from evolutionary and assay-labeled'](https://www.nature.com/articles/s41587-021-01146-5) and its corresponding GitHub repository ['combining-evolutionary-and-assay-labelled-data'](https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data).


## Installation

To get started, follow the steps below to set up the project on your local machine. Please be aware that our development was based on the EPFL's Izar cluster. You may need to make adjustments according to your own setup.

### 1. Clone the repository
```bash
git clone https://github.com/CS-433/ml-project-2-oknneig.git
```
### 2. Navigate to the project directory
```bash
cd ml-project-2-oknneighbors
```
### 3. Create the environment
```bash
conda env create -f environment_updated.yml
```
- To activate the environment, run: 
```bash
conda activate protein_fitness_prediction
```
### 4. Install the plmc package
```bash
cd $HOME  # (or use another directory for plmc <directory_to_install_plmc> and modify `scripts/plmc.sh` accordingly with the custom directory)
git clone https://github.com/debbiemarkslab/plmc.git
cd plmc
make all-openmp
```
## Data

## Assay-labeled dataset

- `data.csv`: Contains protein sequences experimentally measured or labeled with fitness-related information.
- `wt.fasta`: Corresponds to the sequence of the wild-type protein.

## Evolutionary dataset

- `-.a2m`:This dataset consists of homologous sequences from different species or related sequences from the same species but at different evolutionary distances. An A2M file (alignment to model) is used to represent multiple sequence alignments (MSA) in bioinformatics. Processed multiple sequence alignments are stored in .a2m files.

# How to Run the Models

## Information about models used in the project

We utilized the ev Potts model, both augmented and non-augmented simple ridge regresion.

The output of the scripts are saved in:

- `inference`: for intermediate files.
- `results`: for csv files.

To run a model, use the provided scripts. For example, to run the Light dataset and the Q2NB98.a2m file:

Supervised (Ridge)

```bash
python src/evaluate.py Light onehot --n_seeds=20 --n_threads=1 --n_train=-1
```

EV unsupervised:

Train your model

```bash
bash scripts/plmc.sh Q2NB98 Light
```
```bash
python src/evaluate.py Light ev --n_seeds=20 --n_threads=1 --n_train=0
```
EV augmented (supervised + unsupervised)

```bash
python src/evaluate.py Light ev+onehot --n_seeds=20 --n_threads=1 --n_train=-1
```
Note: Modify dataset names (Q2NB98, Light) according to your requirements. To run the demo data from the 'combining-evolutionary-and-assay-labelled-data' GitHub repository, they'll become respectively BLAT_ECOLX and BLAT_ECOLX_Ranganathan2015-2500. 

 

## In silico evolution

The next step is to predict the mutations that will improve the sensitivity of EL222, aiming for high fitness under light conditions but low fitness in darkness.

Before running the models, you will have to generate mutated sequences with `sequence_generator.ipynb`. Note that you can find the `single_mutants.csv` in the folder `generated` , the `double_mutants.csv` was too big to be added in the Github.

### In sillico evolution supervised (Ridge)
First modify `train_and_predict.py` as needed, specify 

- the desired output CSV path
- the dataset (for example for Darkness: "data/Darkness/data.csv")
- the dataset_name (for example: "Darkness")
- to_predict CSV path (= path to the file you want to predict, in our case single_mutants.csv and double_mutants.csv)
- joint_training = "store_true"
- predictor_params = {}
- predictor_name = "onehot"
  
Note that to find the best optogenitic protein sequence, you will have to run for both the Light and Darkness dataset.

```bash
python src/train_and_predict.py 
```
### In silico evolution EV augmented (supervised + unsupervised):
First train your model for Light and Darkness (note that here we used the dataset of the lab)
```bash
bash scripts/plmc.sh Q2NB98 Light
```
```bash
bash scripts/plmc.sh Q2NB98 Darkness
```

Then modify the `train_and_predict.py` as needed, specify 

- the desired output CSV path
- the dataset (for example for Darkness: "data/Darkness/data.csv")
- the dataset_name (for example: "Darkness")
- to_predict CSV path (= path to the file you want to predict, in our case single_mutants.csv and double_mutants.csv)
- joint_training = "store_true"
- predictor_params = {}
- predictor_name = "ev+onehot"
Note that to find the best optogenitic protein sequence, you will have to run for both the Light and Darkness dataset.

```bash
python src/train_and_predict.py 
```
Once the model are trained and that you have predicted the fitness of your mutated sequences, use `sequence_analysis.ipynb` to determine what is the best sequence. 

(Note that you can find the predicted fitness of our single mutants for both Light and Darkness, onehot and ev+onehot, in the folder `generated` as `output_dark_sinlge_mutants_onehot.csv`, `output_light_sinlge_mutants_onehot.csv`,`output_dark_sinlge_mutants_evonehot.csv`, `output_light_sinlge_mutants_evonehot.csv`. The double mutants files where too big to be added in the github)

# Notebooks

- `demo_model_analysis.ipynb`: Evaluate the models' performance using our data and demo data from the 'combining-evolutionary-and-assay-labelled-data' GitHub repository.
- `model_analysis.ipynb`: Evaluate the models' performance based on our data and the demo data from the 'combining-evolutionary-and-assay-labelled-data' GitHub repository.
- `label_data.ipynb`: Process the label dataset to suit our models.
- `msa_exploration.ipynb`: Analyze the evolutionary dataset and convert it into an A2M format.
- `sequence_analysis.ipynb`: Identify the optimal protein sequence for a specific function, in our case, high fitness under light conditions while low fitness in dark conditions.
- `sequence_generator.ipynb`: Generate mutant sequences.


# References
Chloe Hsu, Hunter Nisonoff, Clara Fannjiang & Jennifer Listgarten [Learning protein fitness models from evolutionary and assay-labelled data, Nature Biotechnology 2022](https://www.nature.com/articles/s41587-021-01146-5)



