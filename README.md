
# Protein Fitness Prediction - ML4science - EPFL 2023 - CS-433

Welcome to our ML4science project in collaboration with the [Laboratory of the Physics and Biological Systems](https://www.epfl.ch/labs/lpbs/) based at EPFL.

. The goal is to predict an individual's risk to cardiovascular diseases. To achieve this, we implemented various machine learning methods and tested them. We found Ridge Regression as the most effective method, achieving an accuracy of 86.8% and an F1 score of 42.0%.

## Getting Started

### Data Importation
1. Download the following dataset files at https://github.com/epfml/ML_course/tree/master/projects/project1/data:
   - `x_train.csv` 
   - `y_train.csv`
   - `x_test.csv`

2. Place these downloaded datasets into the `data` folder of this repository.

### Running the Final Model
- To obtain the best predictions, run the `run.ipynb` notebook. The resulting CSV file will be saved in the `predictions` folder.

## Additional Information

### Code Files
- `implementations.py`: Contains the mandatory implementations for this project.
- `additional_functions.py`: Includes functions for cross-validation and prediction.
- `data_cleaning.py`: Provides functions for data manipulation and cleaning.
- `helpers.py`: Contains functions for data loading and csv creation.

### Exploration and Analysis
- `first_insights.ipynb`: This notebook is where we initially investigated the dataset and models (using common parameters).

### Regularization
- `regularization.ipynb`: In this notebook, we fine-tuned our regularized models through cross-validation and adapt thresholds to improve predictive accuracy.

### Polynomial Ridge Regression
- `threshold_poly.ipynb`: This notebook explores the tuning of thresholds for the optimized Ridge Regression models with and without polynomial expansion.
