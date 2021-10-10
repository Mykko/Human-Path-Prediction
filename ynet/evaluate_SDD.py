import pandas as pd
import yaml
import argparse
import torch
from model import YNet



CONFIG_FILE_PATH = 'config/sdd_longterm.yaml'  # yaml config file containing all the hyperparameters
DATASET_NAME = 'sdd'

TEST_DATA_PATH = 'data/SDD/test_longterm.pkl'
TEST_IMAGE_PATH = 'data/SDD/test'
OBS_LEN = 5  # in timesteps
PRED_LEN = 30  # in timesteps
NUM_GOALS = 20  # K_e
NUM_TRAJ = 1  # K_a

ROUNDS = 3  # Y-net is stochastic. How often to evaluate the whole dataset
BATCH_SIZE = 8

#### Load config file and print hyperparameters

with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('config/')[1]
params


#### Load preprocessed Data

df_test = pd.read_pickle(TEST_DATA_PATH)

df_test.head()


#### Initiate model and load pretrained weights

model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)

model.load(f'pretrained_models/{experiment_name}_weights.pt')

#### Evaluate model

model.evaluate(df_test, params, image_path=TEST_IMAGE_PATH,
               batch_size=BATCH_SIZE, rounds=ROUNDS,
               num_goals=NUM_GOALS, num_traj=NUM_TRAJ, device=None, dataset_name=DATASET_NAME)
