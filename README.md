# Protein Fitness Prediction - ML4Science - EPFL 2023

Welcome to our ML4science project in collaboration with the Laboratory of the Physics and Biological Systems. The aim of this project is to predict the amino acid sequence of a protein based on a desired fitness value. For our case, we want to maximize the activity of the EL22 protein when exposed to light while minimizing it when in dark conditions. This protein can be used for optogenetic techniques.

This project is built upon the paper 'Learning protein fitness models from evolutionary and assay-labeled' and its corresponding GitHub repository 'combining-evolutionary-and-assay-labelled-data' from Chloe Hsu.

## Installation

To get started, follow the steps below to set up the project on your local machine.


# Clone the repository
git clone https://github.com/CS-433/ml-project-2-oknneig.git

# Navigate to the project directory
cd ml-project-2-oknneig

# Install the environment
conda env create -f environment_updated.yml
conda activate protein_fitness_prediction

# Install the plmc package
cd $HOME  # (or use another directory for plmc <directory_to_install_plmc> and modify `scripts/plmc.sh` accordingly with the custom directory)
git clone https://github.com/debbiemarkslab/plmc.git
cd plmc
make all-openmp

## Data

### Assay-labeled dataset

- `data.csv`: Contains protein sequences experimentally measured or labeled with fitness-related information.
- `wt.fasta`: Corresponds to the sequence of the wild-type protein.

### Evolutionary dataset

This dataset consists of homologous sequences from different species or related sequences from the same species but at different evolutionary distances. An A2M file (alignment to model) is used to represent multiple sequence alignments (MSA) in bioinformatics. Processed multiple sequence alignments are stored in .a2m files.

## How to Run the Models

### Information about models used in the project

We utilized the ev Potts model, both augmented and non-augmented.

To run a model, use the provided scripts. For example, to run the Light dataset and the Q2NB98.a2m file:

#### Supervised (Ridge)

python src/evaluate.py Light onehot --n_seeds=20 --n_threads=1 --n_train=-1

#### EV unsupervised:
Train your model
bash scripts/plmc.sh Q2NB98 Light

bash
python src/evaluate.py Light ev --n_seeds=20 --n_threads=1 --n_train=-1

#### EV augmented (supervised + unsupervised)
bash
python src/evaluate.py Light ev+onehot --n_seeds=20 --n_threads=1 --n_train=-1

Note: Modify dataset names (Light, Q2NB98) according to your requirements.

## Notebooks

- `demo_model_analysis.ipynb`: Evaluate the models' performance using our data and demo data from the 'combining-evolutionary-and-assay-labelled-data' GitHub repository.
- `model_analysis.ipynb`: Evaluate the models' performance based on our data and demo data from the 'combining-evolutionary-and-assay-labelled-data' GitHub repository.
- `label_data.ipynb`: Process the label dataset to suit our models.
- `msa_exploration.ipynb`: Analyze the evolutionary dataset and convert it into an A2M format.
- `sequence_analysis.ipynb`: Identify the optimal protein sequence for a specific function, in our case, high fitness under light conditions while low fitness in dark conditions.
- `sequence_generator.ipynb`: Generate mutant sequences.

## Additional Models

## References

## License

This project is licensed under the MIT License - see the LICENSE file for details as the same format as# Data


