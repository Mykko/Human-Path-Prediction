import pandas as pd
import yaml
import argparse
import torch
from model import YNet

#### Some hyperparameters and settings

CONFIG_FILE_PATH = 'config/sdd_longterm.yaml'  # yaml config file containing all the hyperparameters
EXPERIMENT_NAME = 'sdd_longterm'  # arbitrary name for this experiment
DATASET_NAME = 'sdd'

TRAIN_DATA_PATH = 'data/SDD/train_longterm.pkl'
TRAIN_IMAGE_PATH = 'data/SDD/train'
VAL_DATA_PATH = 'data/SDD/test_longterm.pkl'
VAL_IMAGE_PATH = 'data/SDD/test'
OBS_LEN = 5  # in timesteps
PRED_LEN = 30  # in timesteps
NUM_GOALS = 20  # K_e
NUM_TRAJ = 1  # K_a

BATCH_SIZE = 4

#### Load config file and print hyperparameters

with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('config/')[1]
params

#### Load preprocessed Data

#%%

df_train = pd.read_pickle(TRAIN_DATA_PATH)
df_val = pd.read_pickle(VAL_DATA_PATH)


df_train.head()

#### Initiate model and load pretrained weights


model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)

#### Start training
# Note, the Val ADE and FDE are without TTST and CWS to save time. Therefore, the numbers will be worse than the final values.


model.train(df_train, df_val, params, train_image_path=TRAIN_IMAGE_PATH, val_image_path=VAL_IMAGE_PATH,
            experiment_name=EXPERIMENT_NAME, batch_size=BATCH_SIZE, num_goals=NUM_GOALS, num_traj=NUM_TRAJ,
            device=None, dataset_name=DATASET_NAME)
