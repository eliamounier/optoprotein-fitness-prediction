# Protein Fitness Prediction - ML4Science - EPFL 2023

Welcome to our ML4science project in collaboration with the Laboratory of the Physics and Biological Systems. The aim of this project is to predict the best amino acid sequence of a protein for a high fitness value. For our case, we want to maximize the activity of the EL22 protein when exposed to light while minimizing it when in dark conditions. This protein can be used for optogenetic techniques.

This project is built upon the paper 'Learning protein fitness models from evolutionary and assay-labeled' and its corresponding GitHub repository 'combining-evolutionary-and-assay-labelled-data' from Chloe Hsu.

# Installation

To get started, follow the steps below to set up the project on your local machine.


## Clone the repository
```bash
git clone https://github.com/CS-433/ml-project-2-oknneig.git
```
## Navigate to the project directory
```bash
cd ml-project-2-oknneig
```
## Install the environment
```bash
conda env create -f environment_updated.yml
```
```bash
conda activate protein_fitness_prediction
```
## Install the plmc package
```bash
cd $HOME  # (or use another directory for plmc <directory_to_install_plmc> and modify `scripts/plmc.sh` accordingly with the custom directory)
git clone https://github.com/debbiemarkslab/plmc.git
cd plmc
make all-openmp
```
# Data

## Assay-labeled dataset

- `data.csv`: Contains protein sequences experimentally measured or labeled with fitness-related information.
- `wt.fasta`: Corresponds to the sequence of the wild-type protein.

## Evolutionary dataset

- `-.a2m`:This dataset consists of homologous sequences from different species or related sequences from the same species but at different evolutionary distances. An A2M file (alignment to model) is used to represent multiple sequence alignments (MSA) in bioinformatics. Processed multiple sequence alignments are stored in .a2m files.

# How to Run the Models

## Information about models used in the project

We utilized the ev Potts model augmented and simple ridge regresion.

Before running the models, you will have to generate mutated sequences with `sequence_generator.ipynb`. Note that you can find the single_mutants.csv in the folder ... , the double_mutants.csv was too big to be added in the Github. 

To run a model, use the provided scripts. For example, to run the Light dataset and the Q2NB98.a2m file:

### Supervised (Ridge)

First modify the file as needed, specify the desired output CSV path, specify your dataset_name, to_predict CSV path (= path to the file you want to predict, in our case single_mutants.csv and double_mutants.csv), joint_training, and predictor_para
Note that to find the best optogenitic protein sequence, you will have to run for both the Light and Darkness dataset.

```bash
python src/train_and_predict.py 
```
### EV augmented (supervised + unsupervised):
First train your model for Light and Darkness 
```bash
bash scripts/plmc.sh Q2NB98 Light
```
```bash
bash scripts/plmc.sh Q2NB98 Darkness
```

Then modify the file as needed, specify the desired output CSV path, specify your dataset_name (Light or Darkness), to_predict CSV path (= path to the file you want to predict, in our case single_mutants.csv and double_mutants.csv), joint_training, and predictor_param, predictor_name = ev+onehot
Note that to find the best optogenitic protein sequence, you will have to run for both the Light and Darkness dataset.

```bash
python src/train_and_predict.py 
```
Once the model are trained and that you have predict the fitness of your mutated sequences, use `sequence_analysis.ipynb` to determine what is the best sequence. 

(Note that you can find the predicted fitness of our single mutants for both Light and Darkness, onehot and ev+onehot, in the folder ... . The double mutants files where too big to be added in the github)

# Notebooks

- `demo_model_analysis.ipynb`: Evaluate the models' performance using our data and demo data from the 'combining-evolutionary-and-assay-labelled-data' GitHub repository.
- `model_analysis.ipynb`: Evaluate the models' performance based on our data and demo data from the 'combining-evolutionary-and-assay-labelled-data' GitHub repository.
- `label_data.ipynb`: Process the label dataset to suit our models.
- `msa_exploration.ipynb`: Analyze the evolutionary dataset and convert it into an A2M format.
- `sequence_analysis.ipynb`: Identify the optimal protein sequence for a specific function, in our case, high fitness under light conditions while low fitness in dark conditions.
- `sequence_generator.ipynb`: Generate mutant sequences.

# Additional Models

# References
[Learning protein fitness models from evolutionary and assay-labelled data, Nature Biotechnology 2022](https://www.nature.com/articles/s41587-021-01146-5)

# License

This project is licensed under the MIT License - see the LICENSE file for details as the same format as# Data


