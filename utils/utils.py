import numpy as np
import math

# converts an index to a Hamming graph position
def ind2sub(ind,K,L):
    if ind == 0:
        return np.zeros(L, dtype=int)
    else:
        mysub = int(np.base_repr(ind,K))
        digits = int(round(np.log(mysub)/np.log(10),3))+1
        mysub_str = '0'*(L-digits) + str(mysub)
        unjoined = list(mysub_str)
        return np.array([int(i) for i in unjoined])

# converts a Hamming graph position to an index
def sub2ind(sub,K,L):
    return int(sum(np.multiply(sub,[K**(L-i-1) for i in range(L)])))