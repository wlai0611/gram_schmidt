import numpy as np
import time

def gram_schmidt(M):
    A   = M.astype(float)
    m,n = A.shape
    Q   = np.zeros(shape=(m,n),dtype=float)
    R   = np.zeros(shape=(n,n),dtype=float)
    for j in range(n):
        R[:j,j] = np.dot(Q.T[:j,:], A[:,j])         #coordinates
        Q[:,j]  = A[:,j] - np.dot(Q[:,:j], R[:j,j]) #current column - projection onto previous columns
        R[j,j]  = np.sqrt(np.dot(Q[:,j],Q[:,j]))    #the diagonals of R = lengths of coordinate axes
        Q[:,j]  = Q[:,j]/R[j,j]                     #every coordinate axis scaled to length 1

    return Q,R

np.random.seed(123)
size = 1000
M = np.random.random((size,size))
start = time.time()
Q, R = gram_schmidt(M)
print(time.time()-start)
I_expected   = np.identity(size)
I_observed   = np.dot(Q.T, Q)  #Are the columns of Q all orthogonal to each other?  Are they all length of 1?
I_difference = np.sqrt(np.sum((I_expected-I_observed)**2))
A_difference = np.sqrt(np.sum((M-(Q@R))**2)) #Does the produce of Q and R equal the original input matrix?

print()