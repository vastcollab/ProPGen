# simulation for persister cells
# need to supply two sets of environment fitnesses and a round number to switch 

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

    # random graph setup 
    if cfg['graph_setup'] == 'random':
        V = cfg['V'] # number of genotypes
        # generate random Erdős–Rényi graph
        # TODO: Read in p argument?
        G, A = generate_erdos_renyi_graph(V, 0.3, seed=seed) 

    # read in graph from file 
    elif cfg['graph_setup'] == 'file': 
        A_file = cfg['A']
        # A = pd.read_csv(A_file, delimiter=',', header=None).to_numpy()
        A = np.loadtxt(A_file, encoding='utf-8-sig')
        V = np.shape(A)[0] 

    N = cfg['N']
    T = cfg['timesteps']
    c = cfg['c']
    trial = args.trial  
    mu = cfg['mu'] 

    # override mapping probabilities
    if args.env_time is not None:
        env_time = args.env_time

    num_gens = cfg['num_gens'] # number of generations before dilution / downsampling

    if len(sys.argv) > 1:
        file_suffix = f"env_time_{env_time}_trial_{trial}"

    # random phenotype probability assignment 
    if cfg['phenotype_probs_setup'] == 'random':
        Q = cfg['Q'] # number of phenotypes

        # for each genotype, generate a probability vector mapping it to each phenotype
        pi = np.zeros((V, Q)) # probability matrix of size # genotypes (V) x # phenotypes (Q)
        for i in range(V):
            pi[i] = generate_pheno_prob_vector(Q)

    # read in phenotype probability assignment from file 
    elif cfg['phenotype_probs_setup'] == 'file':
        probs_file = cfg['pheno_probs']
        # pi = pd.read_csv(probs_file, delimiter=',', header=None).to_numpy() 
        pi = np.loadtxt(probs_file, dtype='float64', encoding='utf-8-sig') # g -> p probabilities
        Q = pi.shape[1]

    # random reproduction probability assignment 
    if cfg['repro_probs_setup'] == 'file':
        repro_file_1 = cfg['repro_probs_1'] 
        repro_file_2 = cfg['repro_probs_2'] 
        r_1 = np.loadtxt(repro_file_1, dtype='float64', encoding='utf-8-sig') # probability of each phenotype reproducing at single timestep, env 1
        r_2 = np.loadtxt(repro_file_2, dtype='float64', encoding='utf-8-sig') # probability of each phenotype reproducing at single timestep, env 2

    # start in environment 1
    r = r_1.copy()
    generation_counter = 0
    env_index = 0  # 0 = env1, 1 = env2

    ## PrFL dynamics simulation
    starttime = time.time()

    # constants
    burn_in = 0 # time before starting to record frequency vec

    #############################
    ### initialize population ###
    #############################

    # population randomly initialized across genotypes and phenotypes, unless 
    # specified (G, P) has 0 probability

    allowed_pairs = [(0, 0), (0, 1)]
    
    # for g in range(V):
    #     for p in range(Q):
    #         if pi[g, p] > 0:  # check whether this (g, p) pair has nonzero probability
    #             allowed_pairs.append((g, p))
    
    num_allowed_pairs = len(allowed_pairs)
    indices = np.random.choice(num_allowed_pairs, size=N, replace=True) # sample with replacement

    # Gamma refers to the set of individuals in the population
    Gamma_geno = np.array([allowed_pairs[i][0] for i in indices])
    Gamma_pheno = np.array([allowed_pairs[i][1] for i in indices])
    # Gamma_geno = np.random.randint(0, V, size=(N)) # assign every individual to a genotype
    # Gamma_pheno = np.random.randint(0, Q, size=(N)) # assign every individual to a phenotype

    # keep track of frequency time series
    freq_timeseries = np.zeros((V, Q, T-burn_in)) # (g, p, T)

    ######################
    ### run simulation ###
    ######################

    for t in tqdm(range(T)):
        if t >= burn_in:

            for gen in range(num_gens):

                ###################################
                # ENVIRONMENT SWITCHING OCCURS HERE
                ###################################
                if generation_counter > 0 and generation_counter % env_time == 0:
                    if env_index == 0:
                        r = r_2.copy()
                        env_index = 1
                    else:
                        r = r_1.copy()
                        env_index = 0

                generation_counter += 1
                ###################################

                pop_size = len(Gamma_pheno)
                # choose individuals to reproduce
                chosen_to_reproduce = np.where(np.random.binomial(1, r[Gamma_pheno], size=pop_size))[0]

                # genotypes of offspring
                offspring_genotypes = np.repeat(Gamma_geno[chosen_to_reproduce], c)

                # mutation step
                chosen_to_mutate = np.where(np.random.binomial(1, mu, size=len(offspring_genotypes)))[0]
                for i in chosen_to_mutate:
                    curr_genotype = offspring_genotypes[i]
                    neighbors = np.nonzero(A[curr_genotype])[0]
                    mutation = np.random.choice(neighbors)
                    offspring_genotypes[i] = mutation

                # map genotypes to phenotypes
                pheno_probs = pi[offspring_genotypes]
                offspring_phenotypes = np.array([
                    np.random.choice(len(pi[0]), p=pheno_probs[i]) for i in range(len(offspring_genotypes))
                ])

                # add offspring to the population
                Gamma_geno = np.concatenate([Gamma_geno, offspring_genotypes])
                Gamma_pheno = np.concatenate([Gamma_pheno, offspring_phenotypes])

            # Downsample back to original population size N
            select_ids = np.random.choice(len(Gamma_geno), size=N, replace=False)
            Gamma_geno = Gamma_geno[select_ids].astype(int)
            Gamma_pheno = Gamma_pheno[select_ids].astype(int)

            # update frequency time series
            combined = np.column_stack((Gamma_geno, Gamma_pheno))
            unique, counts = np.unique(combined, axis=0, return_counts=True)
            freq_temp = counts / N
            for k in range(len(unique)):
                freq_timeseries[unique[k][0], unique[k][1], t-burn_in] = freq_temp[k]

        else:
            print('ERROR')   
        
    # with open(outdir + '/freq_timeseries_' + file_suffix + '.pkl', 'wb') as file:
    #     pickle.dump(freq_timeseries, file)  

    # calculate theoretical equilibrium frequencies and mean fitness
    f_eq, Xbar_theory = utils.calc_f_eq.calc_f_eq(pheno_probs=pi,repro_probs=r,A=A,mu=mu,c=c)
        
    data = {
            'A': A,
            'N': N,
            'T': T,
            'mu': mu,
            'c': c,
            'pheno_probs': pi,
            'repro_probs_1': r1,
            'repro_probs_2': r2,
            'trial': trial,
            'freq_timeseries': freq_timeseries,
            'Gamma_geno': Gamma_geno,
            'Gamma_pheno': Gamma_pheno,
            'f_eq': f_eq,
            'Xbar_theory': Xbar_theory
            }

    with open(outdir + '/sim_data_' + file_suffix + '.pkl', 'wb') as file:
        pickle.dump(data, file)  




