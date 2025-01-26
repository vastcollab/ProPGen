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

def generate_erdos_renyi_graph(n, p, seed=None):
    # generate the ER graph with n nodes and edge probability p
    
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    while not nx.is_connected(G):
        seed += 1
        G = nx.erdos_renyi_graph(n, p, seed=seed)
    
    # get the adjacency matrix of the graph
    adj_matrix = nx.to_numpy_array(G)
    return G, adj_matrix

def generate_pheno_prob_vector(q):
    random_vec = np.random.rand(q)
    return random_vec / np.sum(random_vec)

if __name__ == '__main__':

    # parse config file 
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    seed = cfg['seed']

    np.random.seed(seed) # setting seed for all random numbers

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
    mu = cfg['mu']
    c = cfg['c']
    trial = cfg['Trial number'] 
    outdir = cfg['outdir']

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

    print('Number of phenotypes: ' + str(Q))

    # random reproduction probability assignment 
    if cfg['repro_probs_setup'] == 'random':
        r = np.random.rand(Q) # random probability for each phenotype

    elif cfg['repro_probs_setup'] == 'file':
        repro_file = cfg['repro_probs'] 
        # r = pd.read_csv(repro_file, delimiter=',', header=None).to_numpy() 
        r = np.loadtxt(repro_file, dtype='float64', encoding='utf-8-sig') # probability of each phenotype reproducing at single timestep

    ## PrFL dynamics simulation
    starttime = time.time()

    # constants
    burn_in = 0 # time before starting to record frequency vec

    # initialize population
    # population randomly initialized across genotypes and phenotypes
    # phenotype = 0 is high fitness, phenotype = 1 is low fitness

    # Gamma refers to the set of individuals in the population
    Gamma_geno = np.random.randint(0, V, size=(N)) # assign every individual to a genotype
    Gamma_pheno = np.random.randint(0, Q, size=(N)) # assign every individual to a phenotype

    # keep track of frequency time series
    freq_timeseries = np.zeros((V, Q, T-burn_in)) # (g, p, T)

    # run simulation
    for t in tqdm(range(T)): 
        if t >= burn_in:

            # update frequency time series 
            combined = np.column_stack((Gamma_geno, Gamma_pheno))
            unique, counts = np.unique(combined, axis=0, return_counts=True) # get unique (g, p) combinations and their counts

            freq_temp = counts / N
            for k in range(len(unique)): # for each unique pair
                freq_timeseries[unique[k][0], unique[k][1], t-burn_in] = freq_temp[k]

            # choose inidividuals from population to reproduce at this timestep
            chosen_to_reproduce = np.where(np.random.binomial(1, np.squeeze(r[Gamma_pheno]), size = N))[0]

            # genotypes of offspring for chosen individuals
            offspring_genotypes = np.repeat(Gamma_geno[chosen_to_reproduce], c)

            # offspring mutate according to neighbors
            chosen_to_mutate = np.where(np.random.binomial(1, mu, size = len(offspring_genotypes)))[0]

            for i in chosen_to_mutate:
                curr_genotype = offspring_genotypes[i]

                neighbors = np.nonzero(A[curr_genotype])[0]
                mutation = np.random.choice(neighbors)

                offspring_genotypes[i] = mutation

            # offspring genotypes are mapped to phenotypes according to pi 
            pheno_probs = pi[offspring_genotypes]
            offspring_phenotypes = np.array([np.random.choice(len(pi[0]), p=pheno_probs[i]) for i in range(len(offspring_genotypes))])

            # add offspring to entire population
            pop_genotypes = np.concatenate([Gamma_geno, offspring_genotypes])
            pop_phenotypes = np.concatenate([Gamma_pheno, offspring_phenotypes])

            # selection back to original population size
            select_ids = np.random.choice(len(pop_genotypes), size=N, replace=False)

            # update Gamma_ind and Gamma_pheno
            Gamma_geno = np.array([pop_genotypes[j] for j in select_ids]).astype(int)
            Gamma_pheno = np.array([pop_phenotypes[j] for j in select_ids]).astype(int)

        else:
            print('ERROR')  

    file_suffix =  '_trial' + str(trial)
        
    # np.savetxt(outdir + '/sim2_freq_timeseries' + file_suffix + '.csv', freq_timeseries,delimiter=',')
    with open(outdir + '/sim2_freq_timeseries' + file_suffix + '.pkl', 'wb') as file:
        pickle.dump(freq_timeseries, file)  
        
    data = {
            'A': A,
            'N': N,
            'T': T,
            'mu': mu,
            'c': c,
            'pheno_probs': pi,
            'repro_probs': r,
            'trial': trial,
            'freq_timeseries': freq_timeseries,
            'Gamma_geno': Gamma_geno,
            'Gamma_pheno': Gamma_pheno
            }

    with open(outdir + '/sim2_data' + file_suffix + '.pkl', 'wb') as file:
        pickle.dump(data, file)  