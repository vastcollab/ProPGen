import numpy as np
import pickle
import scipy.linalg

def calc_f_eq(pheno_probs,repro_probs,A,mu,c):

    # load number of genotypes, phenotypes, and fitnesses
    Ng,Np = pheno_probs.shape
    X = repro_probs * c

    # compute selection matrix
    Xk = np.outer(np.ones(Ng*Np),np.ndarray.flatten(np.outer(np.ones((Ng)),X)))
    iterable = tuple([np.ones((Np,Np))]*Ng)
    block_dirac = scipy.linalg.block_diag(*iterable)
    phi_mat = np.outer(np.reshape(pheno_probs,(Ng*Np,1)),np.ones((Ng*Np)))
    selection_mat = phi_mat * block_dirac * Xk

    # compute mutation term
    n_neighbors = np.sum(A,axis=0)
    inverse_neighbor_mat = A / n_neighbors
    mu_rate_mat = mu * inverse_neighbor_mat - np.diag(np.sum(mu * inverse_neighbor_mat,axis=0))
    expanded_mu_mat = np.kron(mu_rate_mat,np.ones((Np,Np)))
    complete_mutation_mat = phi_mat * expanded_mu_mat * Xk

    # compute full matrix with both selection and mutation terms
    rhs_mat = selection_mat + complete_mutation_mat

    # perform eigenvalue decomposition
    eig_decomp = np.linalg.eig(rhs_mat)
    eigvecs = eig_decomp.eigenvectors

    # equilibrium frequency vec is the eigenvector with all elements having the same sign
<<<<<<< HEAD
    ind_list = np.where(np.sum(eigvecs > 0,axis=0) == Ng * Np)[0]
    if len(np.where(np.sum(eigvecs > 0,axis=0) == Ng * Np)[0]) == 0:
        ind = np.where(np.sum(eigvecs < 0,axis=0) == 0)[0][0]
    else:
        ind = ind_list[0]
    # print(len(np.where(np.sum(eigvecs < 0,axis=0) == Ng * Np)[0]),'eyy')
=======
    ind_list = np.where(np.sum(eigvecs >= 0,axis=0) == Ng * Np)[0]
    if len(ind_list) == 0:
        ind_list = np.where(np.sum(eigvecs < 0,axis=0) == Ng * Np)[0]
    
>>>>>>> 4712d2362cbdad1ad3b4c172e318fe0120286547
    f_eq_unnormalized = eigvecs[:,ind_list[0]]
    Xbar_theory = eig_decomp.eigenvalues[ind_list[0]]

    # verify that f_eq and mean fitness are real
    if Ng*Np - np.sum(np.isreal(f_eq_unnormalized)) > 1e-6:
        print('Equilibrium vector may have nonzero imaginary parts.')
        print(f_eq_unnormalized)
    f_eq = np.real(f_eq_unnormalized) / np.sum(np.real(f_eq_unnormalized))

    if not np.isreal(Xbar_theory):
        print('Mean fitness may have nonzero imaginary part.')
        print(Xbar_theory)
    Xbar_theory = np.real(Xbar_theory)
        
    return f_eq, Xbar_theory
