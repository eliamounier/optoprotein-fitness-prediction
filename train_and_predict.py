"""
Train on a dataset and applied the learned model for prediction on another dataset.
"""

import numpy as np
import pandas as pd

from predictors import get_predictor_cls, BoostingPredictor, JointPredictor


def train_and_predict(
    data, dataset_name, to_predict, predictor_name, joint_training, predictor_params
):
    print(f"----- predictor {predictor_name} -----")

    predictor_cls = get_predictor_cls(predictor_name)

    if len(predictor_cls) == 1:
        predictor = predictor_cls[0](dataset_name, **predictor_params)
    elif joint_training:
        predictor = JointPredictor(
            dataset_name, predictor_cls, predictor_name, **predictor_params
        )
    else:
        predictor = BoostingPredictor(dataset_name, predictor_cls, **predictor_params)

    predictor.train(data.seq.values, data.log_fitness.values)
    to_predict["pred"] = predictor.predict(to_predict.seq.values)

    # Specify the desired output CSV path

    output_csv_path = "output_darkness_single_mutant.csv"
    to_predict.to_csv(output_csv_path, index=False)

    print("Predictions - successful")


if __name__ == "__main__":
    
    # Specify your dataset, dataset_name, to_predict CSV path, joint_training, and predictor_params, predictor_name

    dataset = "data/Darkness/data.csv"
    to_predict_csv_path = "data/single_mutants.csv"
    joint_training = "store_true"
    predictor_params = {}
    predictor_name = "ev+onehot"
    dataset_name = "Darkness"

    to_predict = pd.read_csv(to_predict_csv_path)
    data = pd.read_csv(dataset)

    train_and_predict(
        data, dataset_name, to_predict, predictor_name, joint_training, predictor_params
    )
