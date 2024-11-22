import numpy as np
import math
import time
from tqdm import tqdm
from scipy.linalg import hadamard
import argparse
import pandas as pd
import pickle

# converts an index to a Hamming graph position
def ind2sub(ind,K,L):
    if ind == 0:
        return np.zeros(L, dtype=int)
    else:
        mysub = int(np.base_repr(ind,K))
        digits = int(math.log10(mysub))+1
        mysub_str = '0'*(L-digits) + str(mysub)
        unjoined = list(mysub_str)
        return np.array([int(i) for i in unjoined])

# converts a Hamming graph position to an index
def sub2ind(sub,K,L):
    return int(sum(np.multiply(sub,[K**(L-i-1) for i in range(L)])))


# function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Adjacency, probabilities, mutation rate, fitness, and population size.')
    parser.add_argument('--A',type=str,help='Adjacency matrix file.')
    parser.add_argument('--N',type=int,help='Population size.')
    parser.add_argument('--T',type=int,help='Number of time steps.')
    parser.add_argument('--mu',type=float,help='Mutation rate.')
    parser.add_argument('--fitup',type=float,help='Fitness up.')
    parser.add_argument('--fitdown',type=float,help='Fitness down.')
    parser.add_argument('--probs',type=str,help='Probability vector file.')
    parser.add_argument('--trial',type=int,help='Trial number.')
    parser.add_argument('--dir',type=str,help='Output directory.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    A_file = args.A
    A = pd.read_csv(A_file,delimiter=',',header=None).to_numpy()
    V = np.shape(A)[0] # number of vertices = number of genotypes
    N = args.N
    T = args.T
    mu = args.mu
    X = np.array([args.fitup, args.fitdown])
    probs_file = args.probs
    probs = pd.read_csv(probs_file,delimiter=',',header=None).to_numpy()
    trial = args.trial
    outdir = args.dir

    ## Wright-Fisher dynamics simulation
    starttime = time.time()

    # constants
    burn_in = 0 # time before starting to record frequency vec

    # declare populations
    # populations are initialized to random sequence and high fitness phenotype
    # phenotype = 0 is high fitness, phenotype = 1 is low fitness
    random_start_ind = 0
    random_start_pheno = 0 

    # Gamma refers to the set of individuals in the population
    # Gamma_ind = random_start_ind*np.ones((N),dtype=int)
    Gamma_ind = np.random.randint(0,2,size=(N))
    # Gamma_pheno = np.zeros((N),dtype=int)
    Gamma_pheno = np.random.randint(0,2,size=(N))

    # vector of length N keeping track of each individual's fitness
    X_pop = np.squeeze(X[Gamma_pheno])

    # keep track of frequency time series
    freq_timeseries = np.zeros((2*V,T-burn_in))

    # run simulation
    for t in tqdm(range(T)):
        if t >= burn_in:
            # selection step
            Gamma_ind_temp = np.random.choice(Gamma_ind, size=N, replace=True, p=X_pop/np.sum(X_pop))
            
            # mutation step
            chosen_to_mutate = np.where(np.random.binomial(1,mu,size=N))[0]
            for i in chosen_to_mutate:
                Gamma_ind_temp[i] = np.random.choice(V,size=(1),p=A[Gamma_ind[i],:]/np.sum(A[Gamma_ind[i],:]))
            
            Gamma_ind = Gamma_ind_temp

            # phenotype specification step
            pi_pop = np.squeeze(probs[Gamma_ind])
            Gamma_pheno = 1 - np.random.binomial(np.ones((N),dtype=int), p=pi_pop)

            # fitness update
            X_pop = np.squeeze(X[Gamma_pheno])
            
            # update frequency time series
            Gamma_reindexed = np.zeros((N),dtype=int)
            for i in range(N):
                temp = Gamma_ind[i]
                if Gamma_pheno[i] == 1:
                    temp += V
                Gamma_reindexed[i] = temp

            unique, counts = np.unique(Gamma_reindexed, return_counts=True)
            
            freq_temp = counts / N
            for k in range(len(unique)):
                freq_timeseries[unique[k],t-burn_in] = freq_temp[k]
        else:
            # selection step
            Gamma_ind_temp = np.random.choice(Gamma_ind, size=N, replace=True, p=X_pop/np.sum(X_pop))
            
            # mutation step
            chosen_to_mutate = np.where(np.random.binomial(1,mu,size=N))[0]
            for i in chosen_to_mutate:
                Gamma_ind_temp[i] = np.random.choice(np.where(A[i,:]),size=1,p=A[i,:]/np.sum(A[i,:]))
            
            Gamma_ind = Gamma_ind_temp

            # phenotype specification step
            pi_pop = probs[Gamma_ind]
            Gamma_pheno = 1 - np.random.binomial(np.ones((N)), pi_pop)

            # we should not be here if burn-in is 0 (TEMPORARY)
            print('ERROR.')


    ## save to file
    # file_suffix = '_A' + str(A_file) + '_N' + str(N) + '_T' + str(T) + '_mu' + str(mu) + '_fitup' + str(X[0]) + '_fitdown' + str(X[1]) + '_probs' + str(probs_file) + '_trial' + str(trial)
    file_suffix =  '_trial' + str(trial)
    
    np.savetxt(outdir + '/sim1_freq_timeseries' + file_suffix + '.csv', freq_timeseries,delimiter=',')
    
    data = {
        'A': A,
        'N': N,
        'T': T,
        'mu': mu,
        'X': X,
        'probs': probs,
        'trial': trial,
        'freq_timeseries': freq_timeseries,
        'Gamma_ind': Gamma_ind,
        'Gamma_pheno': Gamma_pheno
        }
    with open(outdir + '/sim1_data' + file_suffix + '.pkl', 'wb') as file:
        pickle.dump(data, file)