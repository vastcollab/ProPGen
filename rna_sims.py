# rna 12 simulations 
# should randomly initialize population at a node 

import numpy as np
import math
import time
from tqdm import tqdm
from scipy.linalg import hadamard
import argparse
import pandas as pd
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import yaml
import argparse
import utils.calc_f_eq
import utils.utils
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--config', type=str, help='Path to the config file')
    parser.add_argument('--trial', type=int, help='Trial number')
    parser.add_argument('--out', type=str, default=None, help="Output filename for results (overrides default naming)")
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()
    config_file = args.config

    # parse config file 
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    seed = cfg['seed']

    outdir = args.out

    L = 12
    K = 4

    
     