# Protein Fitness Prediction - ML4Science - EPFL 2023

Welcome to our ML4science project in collaboration with the [Laboratory of the Physics and Biological Systems at EPFL](https://www.epfl.ch/labs/lpbs/). The aim of this project is to predict the amino acid sequence of a protein based on a desired fitness value. For our case, we want to maximize the activity of the EL222 optoprotein when exposed to light while minimizing it when in dark conditions. 

This project is built upon the paper ['Learning protein fitness models from evolutionary and assay-labeled'](https://www.nature.com/articles/s41587-021-01146-5) and its corresponding GitHub repository ['combining-evolutionary-and-assay-labelled-data'](https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data).


## Installation

To get started, follow the steps below to set up the project on your local machine. Please be aware that our development was based on the EPFL's Izar cluster. You may need to make adjustments according to your own setup.

### 1. Install Miniconda3
```bash
bash Miniconda3-latest-Linux-x86_64.sh -b
source ~/.bashrc
rm -rf Miniconda3-latest-Linux-x86_64.sh
conda init
conda update conda
```

### 2. Clone the repository
```bash
git clone https://github.com/CS-433/ml-project-2-oknneig.git
```
### 3. Navigate to the project directory
```bash
cd ml-project-2-oknneighbors
```
### 4. Create the environment
```bash
conda env create -f environment_updated.yml
```
- To activate the environment, run: 
```bash
conda activate protein_fitness_prediction
```
### 5. Install additional package
```bash
conda install pytorch==1.4.0 cudatoolkit=10.1 -c pytorch
```
```bash
pip install torchvision==0.5.0 (conda fails here)
```
```bash
pip install torchaudio==0.5.0
```

### 6. Install the plmc package
```bash
cd $HOME  # (or use another directory for plmc <directory_to_install_plmc> and modify `scripts/plmc.sh` accordingly with the custom directory)
git clone https://github.com/debbiemarkslab/plmc.git
cd plmc
make all-openmp
```
## Data

### Assay-labeled Dataset

The assay-labeled datasets can be found in the `data/<targetprotein>/` folder.

- **data.csv:** This file includes protein sequences `seq` of variants along with their corresponding `log_fitness` values.
- **wt.fasta:** This file represents the sequence of the wild-type protein, needed for the alignment search process.

### Evolutionary Dataset

The evolutionary datasets are stored in the `alignments` folder.

- **_.a2m:** The datasets comprises homologous sequences from various species or related sequences for a target protein. The multiple sequence alignment must follow the .a2m convention and can be obtained using the [hh-suite](https://github.com/soedinglab/hh-suite) package. 



### Information about Models Used in the Project

Our project focuses on the following models:

- Ridge Regression
- Supervised EV Potts
- Augmented Potts

The outputs of the scripts are saved in:

- `inference/`: for the model parameters.
- `results/`: for model evaluation CSV files.

### Training and Evaluating a Model

To run a model, use the provided scripts. For instance, when working with the Light assay-labeled dataset and the Q2NB98.a2m evolutionary file:

#### Supervised Ridge Regression Model Evaluation:

- Evaluation:
```bash
python src/evaluate.py Light onehot --n_seeds=20 --n_threads=1 --n_train=-1
```
- `n_train = -1` corresponds to an 80/20 split for training and test sets.
- `n_seeds` corresponds to different random splits.

#### Supervised EV pott model evaluation:

1. Train your unsupervised model and save the parameters

```bash
bash scripts/plmc.sh Q2NB98 Light
```
2. Evaluation

```bash
python src/evaluate.py Light ev --n_seeds=20 --n_threads=1 --n_train=0
```
#### EV augmented model evaluation:

1. Train your unsupervised model and save the parameters

```bash
bash scripts/plmc.sh Q2NB98 Light
```
2. Evaluation
```bash
python src/evaluate.py Light ev+onehot --n_seeds=20 --n_threads=1 --n_train=-1
```


## In Silico Evolution

The in silico evolution step aims to identify the EL222 sequence that results in high fitness under light conditions but low fitness in darkness.

First, generate double and single-mutant sequences using `sequence_generator.ipynb`. Please note that the output file `single_mutants.csv` can be found in the `generated/` folder. `double_mutants.csv` exceeds the size limit for GitHub.

### 1. In Silico Evolution Supervised with the Ridge Regression Model

First, modify `src/train_and_predict.py` as needed. Specify the following parameters:

- The desired output CSV path
- The dataset (for example, for Darkness: "data/Darkness/data.csv")
- The dataset_name (for example: "Darkness")
- The to_predict CSV path (the path to the file you want to predict, in our case `single_mutants.csv` and `double_mutants.csv`)
- `joint_training = "store_true"`
- `predictor_params = {}`
- `predictor_name = "onehot"`

Note that to find the best optogenetic protein sequence, you will have to run for both the Light and Darkness datasets.

```bash
python src/train_and_predict.py 
```

### 2. In silico evolution with the EV augmented model:
First train your model for Light and Darkness 
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

Note that to find the best optogenetic protein sequence, you will have to run for both the Light and Darkness datasets.

```bash
python src/train_and_predict.py 
```

### Find the Right Sequence

After training the models and predicting the fitness of your mutated sequences, use `sequence_analysis.ipynb` to determine the best sequence.

## Notebooks

- **`label_data.ipynb`**: Process the original LPBS assay-labeled dataset.
- **`msa_exploration.ipynb`**: Analyze the evolutionary dataset and convert it into an A2M format.
- **`demo_model_analysis.ipynb`** and **`additional_exp_analysis.ipynb`**: Analyze the models' performance using the LPBS datasets and the demo datasets from the 'combining-evolutionary-and-assay-labelled-data' GitHub repository.
- **`sequence_generator.ipynb`**: Generate mutant sequences.
- **`sequence_analysis.ipynb`**: Identify the optimal protein sequence for a specific function—in our case, achieving high fitness under light conditions while maintaining low fitness in dark conditions.

## Key file scripts

Note: **`src/evaluate_multiprocessing.py`**,**`src/predictors/onehot_predictors.py`**, **`src/predictors/ev_predictors.py`** , **`src/utils/metrics.py`** are key file scripts of our project. 



# References
Chloe Hsu, Hunter Nisonoff, Clara Fannjiang & Jennifer Listgarten [Learning protein fitness models from evolutionary and assay-labelled data, Nature Biotechnology 2022](https://www.nature.com/articles/s41587-021-01146-5)



