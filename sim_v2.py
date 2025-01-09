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

# # function to parse arguments
# def parse_args():
#     parser = argparse.ArgumentParser(description='Adjacency, probabilities, mutation rate, fitness, and population size.')
#     # parser.add_argument('--A',type=str,help='Adjacency matrix file.')
#     parser.add_argument('--seed',type=int,help='Random graph seed.')
#     parser.add_argument('--V',type=int,help='Number of genotypes.')
#     parser.add_argument('--Q',type=int,help='Number of phenotypes.')
#     parser.add_argument('--N',type=int,help='Population size.')
#     parser.add_argument('--T',type=int,help='Number of time steps.')
#     parser.add_argument('--mu',type=float,help='Mutation rate.')
#     parser.add_argument('--c',type=int,help='Number of offspring produced.')
#     # parser.add_argument('--phenoprobs',type=str,help='Phenotype probability vector file.')
#     # parser.add_argument('--reproprobs',type=str,help='Reproduction probability vector file.')
#     parser.add_argument('--trial',type=int,help='Trial number.')
#     parser.add_argument('--dir',type=str,help='Output directory.')
#     args = parser.parse_args()
#     return args

def generate_erdos_renyi_graph(n, p, seed=None):
    # generate the ER graph with n nodes and edge probability p
    
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    while not nx.is_connected(G):
        seed += 1
        G = nx.erdos_renyi_graph(n, p, seed=seed)
    
    # get the adjacency matrix of the graph
    adj_matrix = nx.to_numpy_array(G)
    return G, adj_matrix

if __name__ == '__main__':

    # parse config file 
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    seed = cfg['seed']

    # random graph setup 
    if cfg['graph_setup'] == 'random':
        V = cfg['V'] # number of genotypes

        # generate random Erdős–Rényi graph
        # TODO: Read in p argument?
        G, A = generate_erdos_renyi_graph(V, 0.3, seed=seed)   

    # read in from file 
    elif cfg['graph_setup'] == 'file': 
        A_file = cfg['A']
        

#     N = args.N
#     T = args.T
#     mu = args.mu
#     c = args.c
#     Q = args.Q

    # randomly assign 

    # probs_file = args.phenoprobs
    # pi = pd.read_csv(probs_file,delimiter=',',header=None).to_numpy() # g -> p probabilities
    # repro_file = args.reproprobs
    # r = pd.read_csv(repro_file,delimiter=',',header=None).to_numpy() # probability of each phenotype reproducing at single timestep

    # trial = args.trial
    # outdir = args.dir

    # ## PrFL dynamics simulation
    # starttime = time.time()

    # # constants
    # burn_in = 0 # time before starting to record frequency vec

    # # initialize population
    # # population randomly initialized across genotypes and phenotypes
    # # phenotype = 0 is high fitness, phenotype = 1 is low fitness

    # # Gamma refers to the set of individuals in the population
    # Gamma_ind = np.random.randint(0, 2, size=(N))
    # Gamma_pheno = np.random.randint(0, 2, size=(N))

    # # keep track of frequency time series
    # freq_timeseries = np.zeros((2*V,T-burn_in)) # (g*p, T)

    # # run simulation
    # for t in tqdm(range(T)): 
    #     if t >= burn_in:

    #         # update frequency time series 
    #         Gamma_reindexed = np.zeros((N), dtype=int)
    #         for i in range(N):
    #             temp = Gamma_ind[i]
    #             if Gamma_pheno[i] == 1:
    #                 temp += V
    #             Gamma_reindexed[i] = temp

    #         unique, counts = np.unique(Gamma_reindexed, return_counts=True)

    #         freq_temp = counts / N
    #         for k in range(len(unique)):
    #             freq_timeseries[unique[k],t-burn_in] = freq_temp[k]

    #         # choose inidividuals from population to reproduce at this timestep
    #         chosen_to_reproduce = np.where(np.random.binomial(1, np.squeeze(r[Gamma_pheno]), size = N))[0]
    #         # print('Parent genotypes: ', Gamma_ind)
    #         # print('Chosen to reproduce: ', chosen_to_reproduce)

    #         # genotypes of offspring for chosen individuals
    #         offspring_genotypes = np.repeat(Gamma_ind[chosen_to_reproduce], c)
    #         # print('Offspring genotypes', offspring_genotypes)

    #         # offspring mutate according to neighbors
    #         chosen_to_mutate = np.where(np.random.binomial(1, mu, size = len(offspring_genotypes)))[0]
    #         # print('Chosen to mutate', chosen_to_mutate)

    #         for i in chosen_to_mutate:
    #             curr_genotype = offspring_genotypes[i]

    #             neighbors = np.nonzero(A[curr_genotype])[0]
    #             mutation = np.random.choice(neighbors)

    #             offspring_genotypes[i] = mutation

    #         # print('New offspring genotypes', offspring_genotypes)

    #         # offspring genotypes are mapped to phenotypes according to pi 
    #         pheno_probs = np.squeeze(pi[offspring_genotypes])
    #         # print('phenotype probs', pheno_probs)
    #         offspring_phenotypes = 1 - np.random.binomial(np.ones((len(offspring_genotypes)), dtype=int), p=pheno_probs)
    #         # print('offspring_phenotypes', offspring_phenotypes)

    #         # add offspring to entire population
    #         pop_genotypes = np.concatenate([Gamma_ind, offspring_genotypes])
    #         pop_phenotypes = np.concatenate([Gamma_pheno, offspring_phenotypes])

    #         # selection back to original population size
    #         select_ids = np.random.choice(len(pop_genotypes), size=N, replace=False)

    #         # update Gamma_ind and Gamma_pheno
    #         Gamma_ind = np.array([pop_genotypes[j] for j in select_ids]).astype(int)
    #         Gamma_pheno = np.array([pop_phenotypes[j] for j in select_ids]).astype(int)

    #     else:

    #         print('ERROR')  

    # file_suffix =  '_trial' + str(trial)
        
    # np.savetxt(outdir + '/sim2_freq_timeseries' + file_suffix + '.csv', freq_timeseries,delimiter=',')
        
    # data = {
    #         'A': A,
    #         'N': N,
    #         'T': T,
    #         'mu': mu,
    #         'c': c,
    #         'pheno_probs': pi,
    #         'repro_probs': r,
    #         'trial': trial,
    #         'freq_timeseries': freq_timeseries,
    #         'Gamma_ind': Gamma_ind,
    #         'Gamma_pheno': Gamma_pheno
    #         }

    # with open(outdir + '/sim2_data' + file_suffix + '.pkl', 'wb') as file:
    #     pickle.dump(data, file)  