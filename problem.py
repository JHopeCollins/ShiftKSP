from math import sqrt, sin, cos, pi
import numpy as np
from scipy.sparse import eye, spdiags, csr_matrix, kron, csr_matrix
from petsc4py import PETSc
import butchertableau as bt

def Amat(m, re, angle=pi/3, symmetric=True, petsc=False):
    n = m * m # number of unknowns
    I = eye(m)
    L = spdiags([[1]*m, [-1]*m], [0, 1], m, m)
    D = spdiags([[-1]*m, [2]*m, [-1]*m], [-1,0,1], m, m)
    A =  (1/re)*(m+1)**2 * (kron(D,I) + kron(I,D))

    if not symmetric:
        A +=  (m+1) * (cos(angle)*kron(L,I) + sin(angle)*kron(I,L))
    A += eye(n)

    if petsc:
        sizes = (n, n)
        amat = PETSc.Mat().createAIJ(
            size=(sizes, sizes),
            csr=(A.indptr, A.indices)
        )
        amat.setUp()
        amat.setValuesCSR(A.indptr, A.indices, A.data)
        amat.assemble()
        return A, amat
    return A

def Smat(order):
    tableau = bt.butcher(order, 15)
    S, _, _ = tableau.radau() 
    Sinv = np.array(tableau.inv(S), dtype=float)
    S = np.array(S, dtype=float)
    return S, Sinv
