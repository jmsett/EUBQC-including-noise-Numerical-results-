import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import math
from itertools import product
from itertools import permutations

####################
### Define Gates ###

# Universal SET
pi8 = np.array([[1,0],[0,np.exp(-1j/4*np.pi)]])
h = 1/np.sqrt(2)*np.array([[1,1],[1,-1]])
Cx = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
CNOT = np.reshape(Cx,[2,2,2,2])

# Initial 1-Qubit State
v0 = np.array([1,0])
v1 = np.array([0,1])

############################
### Define Gate Function ###

# Takes the memory "sgma", Gate "G" and qubit label "qbit" as INPUT
# The gate "G" is applied to the qubit label "qbit"
# qubit = [i] for single gate qubits, qubit = [control,target] for CNOT

def f(sgma,G,qbit,N):
    # N is defined before f() is called
    # indices 0,1,...,2*N-1
    ind_out = list(range(2*N))
    ind_sgm = list(range(2*N))

    #Define Summation indices for Gates
    indG = [q for q in qbit]
    indGH= [q+N for q in qbit]

    # Define summation indices: "x" & "y"
    # 52 is maximum value for EinSum indices
    # 51 is effective Max due to internal functions

    x = 51
    y = x-2
    for i in qbit:
        ind_sgm[i] = x
        ind_sgm[N+i] = y
        indG += [x]
        indGH+= [y]
        x-=1
        y-=1

    return np.einsum(sgma,ind_sgm,G,indG,np.conjugate(G),indGH,ind_out,optimize=True)

####################################################
### Initialize Server's Memory Function ("sgma") ###

def init(N):
    sgma = v0

    for i in range(1,N):
        sgma = np.kron(sgma,v0)

    sgma = np.outer(sgma,sgma)
    return np.reshape(sgma,[2 for i in range(2*N)])

#########################################################
### Noisy Memory after J steps Function (Mixed State) ###

# g(input memory "sgma", Instruction "inst" ~ U_i, "J" Steps, Noise Probability "p")

def g(sgma,inst,J,p,N):

    # z will be ploted, z is the entropy of the noisy (mixed) state
    z = np.zeros(J+1)
    z[0] = 0

    # Lambda prefactors fixed by instructions "inst"
    # Number of gates "g"
    g = N*(N+1)
    lbda = (p/g)*np.ones(g)
    lbda[inst] += (1-p)

    for step in range(J):

        Gcount = 0
        res=np.zeros(sgma.shape)+0j

        for i in range(N):
            res+=lbda[Gcount]*f(sgma,h,[i],N)
            Gcount+=1

        for i in range(N):
            res+=lbda[Gcount]*f(sgma,pi8,[i],N)
            Gcount+=1

        for i in range(N):
            for j in range(N):
                if i==j:
                    continue
                res+=lbda[Gcount]*f(sgma,CNOT,[i,j],N)
                Gcount+=1

        sgma = res
        EV = la.eigvals(np.reshape(res,[2**N,2**N]))
        z[step+1] = sum(-s*np.log2(s) if s != 0 else 0 for s in EV.real)

    return z,sgma


def g_lure(sgma,inst,J,p,N):#Lure Noise-Map

    # z will be ploted, z is the entropy of the noisy (mixed) state
    z = np.zeros(J+1)
    z[0] = 0

    # Lambda prefactors fixed by instructions "inst"
    # Number of gates "g"
    g = N*(N+1)
    lbda = (p/g)*np.ones(g)
    
    lure_len = len(inst)
    
    for k in inst:
        lbda[k] += (1-p)/lure_len

    for step in range(J):

        Gcount = 0
        res=np.zeros(sgma.shape)+0j

        for i in range(N):
            res+=lbda[Gcount]*f(sgma,h,[i],N)
            Gcount+=1

        for i in range(N):
            res+=lbda[Gcount]*f(sgma,pi8,[i],N)
            Gcount+=1

        for i in range(N):
            for j in range(N):
                if i==j:
                    continue
                res+=lbda[Gcount]*f(sgma,CNOT,[i,j],N)
                Gcount+=1

        sgma = res
        EV = la.eigvals(np.reshape(res,[2**N,2**N]))
        z[step+1] = sum(-s*np.log2(s) if s != 0 else 0 for s in EV.real)

    return z,sgma



###########################################
### Instructions without Noise Function ###

def f_vec(sgma,G,qbit,N):
    sgma = sgma.reshape([2 for i in range(N)])
    # N is defined before f() is called
    # indices 0,1,...,N-1
    ind_out = list(range(N))
    ind_sgm = list(range(N))

    #Define Summation indices for Gates
    indG = [q for q in qbit]

    # Define summation indices: "x" & "y"
    # 52 is maximum value for EinSum indices
    # 51 is effective Max due to internal functions

    x = 51
    for i in qbit:
        ind_sgm[i] = x
        indG += [x]
        x-=1

    return np.einsum(sgma,ind_sgm,G,indG,ind_out,optimize=True)


def noiseless(sgma,arr,N):
    gate = [h,pi8,CNOT]
    for i in range (len(arr)):
	# arr carries gate instruction on [0] argument
	# and qubit number on [1] argument
        sgma = f_vec(sgma,gate[arr[i][0]],arr[i][1],N)
    return sgma


def fidelity(vec,sgma,N):
    vec = vec.reshape(2**N)
    sgma = sgma.reshape(2**N,2**N)
    return np.einsum('i,ij,j',vec,sgma,vec)

######################################################
### Fidelity for Bell-Circuit (on initaial memory) ###

def bellFid(p):
    N=2
    vec = np.kron(v0,v0)
    sgma = init(N)

    # Circuit Instructions

    Inst = [[0,[0]],	# H on qbit [0]
            [2,[0,1]]]	# CNOT on qubits [0,1]

    vec = noiseless(vec,Inst,N)

    # g(sgma,inst,J,p):
    sgma = g(sgma,0,1,p,N)[1] # H on qbit [0] (gate #0)
    sgma = g(sgma,4,1,p,N)[1] # CNOT on qbits [0,1] (gate #4)

    return fidelity(vec,sgma,N)

###################################################
### Fidelity for GHZ-State (on initaial memory) ###

def ghzFid(p):
    N = 3
    vec = np.kron(np.kron(v0,v0),v0)
    sgma = init(N)

    # Circuit Instructions

    Inst = [[0,[0]],
            [2,[0,1]],
            [2,[1,2]]]

    vec = noiseless(vec,Inst,N)

    # g(sgma,inst,J,p):
    sgma = g(sgma,0,1,p,N)[1] # H on q0
    sgma = g(sgma,6,1,p,N)[1] # CNOT on [0,1]
    sgma = g(sgma,9,1,p,N)[1] # CNOT on [1,2]

    return fidelity(vec,sgma,N)

##############################
### Partial Trace Function ###

def partial_trace(rho, keep, dims, optimize=False):
    """Calculate the partial trace

    ρ_a = Tr_b(ρ)

    Parameters
    ----------
    ρ : 2D array
        Matrix to trace
    keep : array
        An array of indices of the spaces to keep after
        being traced. For instance, if the space is
        A x B x C x D and we want to trace out B and D,
        keep = [0,2]
    dims : array
        An array of the dimensions of each space.
        For instance, if the space is A x B x C x D,
        dims = [dim_A, dim_B, dim_C, dim_D]

    Returns
    -------
    ρ_a : 2D array
        Traced matrix
    """
    keep = np.asarray(keep)
    dims = np.asarray(dims)
    Ndim = dims.size
    Nkeep = np.prod(dims[keep])

    idx1 = [i for i in range(Ndim)]
    idx2 = [Ndim+i if i in keep else i for i in range(Ndim)]
    rho_a = rho.reshape(np.tile(dims,2))
    rho_a = np.einsum(rho_a, idx1+idx2, optimize=optimize)
    return rho_a.reshape(Nkeep, Nkeep)