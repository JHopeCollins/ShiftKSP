from math import sqrt, sin, cos, pi
import numpy as np
from scipy.sparse import eye, spdiags, kron, csr_matrix
from scipy.sparse.linalg import splu, LinearOperator, gmres
from scipy.linalg import solve_sylvester, norm
import butchertableau as bt

from sys import argv
import petsctools
from petsctools import set_from_options, inserted_options
petsctools.init(argv)
from petsc4py import PETSc

np.random.seed(6)

def mdot(x, y):
    n = len(x)
    r = np.ndarray(n)
    for i in range(n):
        r[i] = y.dot(x[i])
    return r

def dmult(x, y):
    wrk = x[0].duplicate()
    wrk.zeroEntries()
    for i, alpha in enumerate(y):
        wrk.axpy(alpha, x[i])
    return wrk

class ExtendedKrylovSpace:
    def __init__(self, v0, m):
        self._size = 1
        self._vectors = tuple(v0.duplicate() for _ in range(2*m+2))
        v0.copy(self.vectors[0])

    @property
    def vectors(self):
        return self._vectors[:self.size]

    @property
    def size(self):
        return self._size

    def __getitem__(self, i):
        return self.vectors[i]

    def add_basis_vector(self, w):
        self._size += 1
        V = self.vectors
        h = mdot(V[:-1], w)
        w = w - dmult(V[:-1], h)
        h1 = mdot(V[:-1], w)
        w = w - dmult(V[:-1], h1)
        h += h1
        normw = np.linalg.norm(w)
        w /= normw
        w.copy(V[-1])
        return w, normw, h

# Initialize
m = 10 # grid size
n = m * m # number of unknowns
symmetric = True
re = 10
angle = pi/3
I = eye(m)
L = spdiags([[1]*m, [-1]*m], [0, 1], m, m)
D = spdiags([[-1]*m, [2]*m, [-1]*m], [-1,0,1], m, m)
A =  (1/re)*(m+1)**2 * (kron(D,I) + kron(I,D))
if not symmetric:
    A +=  (m+1) * (cos(angle)*kron(L,I) + sin(angle)*kron(I,L))
A += eye(n)

sizes = (n, n)
amat = PETSc.Mat().createAIJ(
    size=(sizes, sizes),
    csr=(A.indptr, A.indices)
)
amat.setUp()
amat.setValuesCSR(A.indptr, A.indices, A.data)
amat.assemble()

ksp = PETSc.KSP().create()
ksp.setOperators(amat)

set_from_options(
    ksp, options_prefix="",
    parameters={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_reuse_preconditioner": None,
    }
)

w = amat.createVecRight()

# Generate Butcher tableau matrix
order = 8
tableau = bt.butcher(order, 15)
S, _, _ = tableau.radau() 
Sinv = np.array(tableau.inv(S),np.float64)
p = Sinv.shape[0]


# Set Extended Krylov space quantities
m_krylov = 160 # maximum iteration count
X = np.zeros((n,p)) # solution matrix
Varray = np.zeros((n, 2 * m_krylov + 2)) # matrix holding basis vectors
H = np.zeros((2 * m_krylov + 1, 2 * m_krylov)) # to hold projected problem
c = np.zeros((2 * m_krylov + 1, 1)) # projected rhs
y = np.zeros((2 * m_krylov, p)) # projected solution

# Create rhs matrix
b = np.random.randn(n, 1)

# First basis vector (orthogonalised rhs)
beta = norm(b)
c[0,0] = beta
w.array[:] = b.flatten()/beta

V = ExtendedKrylovSpace(w, m_krylov)

# New vector (multiply previous one by A then orthogonalise)
ksp.solve(V[-1], w)
w, normwp, hp = V.add_basis_vector(w)
hp = np.append(hp, normwp)

for i in range(m_krylov):

    # New basis vector (obtained by mult by A)
    amat.mult(V[-1], w)
    w, normw, h = V.add_basis_vector(w)

    # Put projection data in H
    ind = V.size - 2
    H[:ind+1, ind] = h
    H[ind+1, ind] = normw

    # Put projection data for previous set, see note below
    H[:ind, ind - 1] = -H[:ind, :ind - 1] @ hp[:ind - 1]
    H[ind - 1, ind - 1] += 1
    H[:ind + 2, ind - 1] -= np.append(h, normw) * normwp
    H[:ind + 2, ind - 1] /= hp[ind - 1]

    # Move on to add next set of basis vectors (obtained by mult by A^-1)
    ksp.solve(V[-1], w)
    w, normwp, hp = V.add_basis_vector(w)

    # Note: The following projection data (hp and normwp) are
    # not ready to be included in H at this stage. It's only
    # possible to add them into H once we multiply the 
    # corresponding set of vectors by A and use the new 
    # projection data to make the Krylov projection relation 
    # hold.
    # Hence, at this stage of the process, the relation that
    # hold is valid up to the so far constructed basis, that
    # is:
    #  A V[:,:2*i*p] = V[:,2*i*p+p] H[:2*i*p+p,2*i*p]
    
    # Solve projected problem
    temp = 2*i+2
    y = solve_sylvester(H[:temp,:temp],Sinv,np.outer(c[:temp],np.ones(p)))
    
    # Check residual norm
    r = H[temp:temp+p,:temp] @ y
    rnorms = np.zeros((p,1))
    for k in range(p):
        rnorms[k] = norm(r[:,k])/beta

    print(f"max r_{i}: {max(rnorms)[0]:.6e}")
    if(np.max(rnorms) < 1e-8):
        break

# Solution recovery    
for i, v in enumerate(V.vectors):
    Varray[:, i] = v.array_r
X = Varray[:,:temp]@y

# Check true residual norms
norms = np.zeros((p,1))
R = A@X+X@Sinv
for k in range(p):
    norms[k] = np.linalg.norm(b.T-R[:,k])/beta
print(f"Maximum of true residual norms: {max(np.abs(norms))[0]:.6e}")

