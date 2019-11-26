import os
import string
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import sys
import pickle
from datetime import datetime

# DATA_DIR = '/data/'
PROJECT_DIR = '/home/kaandonbekci/Projects/Projects_2/fashion-flow'
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
DEEPFASHION_DIR = os.path.join(DATA_DIR, 'DeepFashion')
DUMP_DIR = os.path.join(PROJECT_DIR, 'dumps')
LOG_DIR = os.path.join(PROJECT_DIR, 'logs')

def read_image(path, as_float=True):
    if as_float:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
    else: 
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

# def tf_fix(tf):
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     if gpus:
#         try:
#             for gpu in gpus:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#         except RuntimeError as e:
#             print(e)