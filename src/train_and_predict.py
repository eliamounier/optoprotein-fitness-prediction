'''
Train and Predict.
'''
import functools
import logging
import os

import numpy as np
import pandas as pd

from predictors import get_predictor_cls, BoostingPredictor, JointPredictor
from utils.metric_utils import spearman, topk_mean, r2, hit_rate, aucroc, ndcg
from utils.io_utils import load_data_split, get_wt_log_fitness, get_log_fitness_cutoff
from utils.data_utils import dict2str



def train_and_predict(data, to_predict, predictor_name, joint_training, predictor_params):
    print(f'----- predictor {predictor_name} -----')


    predictor_cls = get_predictor_cls(predictor_name)
    
    if len(predictor_cls) == 1:
        predictor = predictor_cls[0](dataset_name, **predictor_params)
    elif joint_training:
        predictor = JointPredictor(dataset_name, predictor_cls, predictor_name, **predictor_params)
    else:
	predictor = BoostingPredictor(dataset_name, predictor_cls, **predictor_params)

    predictor.train(data.seq.values, data.log_fitness.values)
    to_predict['pred'] = predictor.predict(to_predict.seq.values)

    # Specify the desired output CSV path
    output_csv_path = '/home/emounier/combining-evolutionary-and-assay-labelled-data/output_darkness_single_mutant.csv'
    to_predict.to_csv(output_csv_path, index=False)

    print('Predictions - successfully saved')

if __name__ == '__main__':
   
    dataset_name = '/home/emounier/combining-evolutionary-and-assay-labelled-data/data/Darkness/data.csv'
    to_predict_csv_path = '/home/emounier/combining-evolutionary-and-assay-labelled-data/data/single_mutants.csv'
    joint_training = True  
    predictor_params = {}  
    
    to_predict = pd.read_csv(to_predict_csv_path)
    data = pd.read_csv(dataset_name)

    predictor_name = 'onehot'

    train_and_predict(data, to_predict, predictor_name, joint_training, predictor_params)

    
