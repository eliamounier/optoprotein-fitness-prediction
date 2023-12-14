# Protein Fitness Prediction - ML4Science - EPFL 2023

Welcome to our ML4science project in collaboration with the Laboratory of the Physics and Biological Systems. The aim of this project is to predict the amino acid sequence of a protein based on a desired fitness value. For our case we want to maximize the activity of the EL22 protein when exposed to light while minimize it when in dark condition. This protein can be used for optogenetic techniques.

This project is built upon the paper 'Learning protein fitness models from evolutionary and assay-labeled' and its corresponding GitHub repository 'combining-evolutionary-and-assay-labelled-data' from Chloe Hsu. 

## Installation

To get started, follow the steps below to set up the project on your local machine.


# Clone the repository
//bash
git clone https://github.com/CS-433/ml-project-2-oknneig.git

Navigate to the project directory and install the environement with the following command:

//bash
cd ml-project-2-oknneig

//bash
conda env create -f environment_updated.yml

//bash
conda activate protein_fitness_prediction

Install the plmc package:
//bash
    cd $HOME (or use another directory for plmc <directory_to_install_plmc> and
modify `scripts/plmc.sh` accordingly with the custom directory)
    git clone https://github.com/debbiemarkslab/plmc.git
    cd plmc
    make all-openmp

# Data
## Assay-labeled dataset
- `data.csv`: contains protein sequences that have been experimentally measured or labeled with information related to their fitness. 

- `wt.fasta`: correspond to the sequence of the wild-type protein

## Evolutionary dataset
This dataset consist of homologous sequences from different species or related sequences from the same species but at different evolutionary distances. 
An A2M file, or "alignment to model" file, is a file format commonly used to represent multiple sequence alignments (MSA) in bioinformatics. It contains a sequence alignment where each position in the alignment corresponds to a column, and each row represents a different sequence. 
To generate this alignment file we !!!!!!!!!

- `.a2m`:  Processed multiple sequence alignments

# How to run the models

[Information about models used in the project]
We used the ev Potts model, augmented and non augmented. 

To run a model, you have to use the scripts provided.
Example to run the Light dataset and the Q2NB98.a2m file
Supervised (Rige)
//bash
python src/evaluate.py Light onehot --n_seeds=20 --n_threads=1 --n_train=-1


EV unsupervised:
//bash
bash scripts/plmc.sh Q2NB98 Light

This command allow you to train your model
//bash
python src/evaluate.py Light ev --n_seeds=20 --n_threads=1 --n_train=-1

EV augmented (supervised + unsupervised)
//bash
python src/evaluate.py Light ev+onehot --n_seeds=20 --n_threads=1 --n_train=-1

Note that to run an other dataset you can modify Light and Q2NB98 according to what you want to run

# Notebooks

- `demo_model_analysis.ipynb`: In this notebook we evaluate the models's perfomranve based on the result we obtain from our data and for the demo data of the 'combining-evolutionary-and-assay-labelled-data' github

- `model_analysis.ipynb`: In this notebook we evaluate the models's perfomranve based on the result we obtain from our data and for the demo data of the 'combining-evolutionary-and-assay-labelled-data' github

- `label_data.ipynb`: In this notebook we process the label dataset so it is suitable for our models

- `msa_exploration.ipynb`: In this notebook we analyze the evolutionnary dataset and convert it into an a2m format

- `sequence_analysis.ipynb`: In this notebook we identify the optimal protein sequence for a specific function, in our case high fitness under light condition while low fitness in dark condition

- `sequence_generator.ipynb`: In this notebook we generate mutant sequences



# Additional model


# References


# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
