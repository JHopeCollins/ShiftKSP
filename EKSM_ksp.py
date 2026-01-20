from petsctools import init as petsc_init, set_from_options
petsc_init()
from petsc4py import PETSc

from math import sqrt, sin, cos, pi
import numpy as np
from scipy.sparse import eye, spdiags, kron, csr_matrix
from scipy.sparse.linalg import splu, LinearOperator, gmres
import butchertableau as bt

from eksm import eksm, KroneckerProductMat

np.random.seed(6)

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

# Generate Butcher tableau matrix
order = 8
tableau = bt.butcher(order, 15)
S, _, _ = tableau.radau() 
Sinv = np.array(tableau.inv(S),np.float64)
p = Sinv.shape[0]

smat = PETSc.Mat().createDense(
    size=((p, p), (p, p)),
    array=Sinv
)

sarray_r = smat.getDenseArray(readonly=True)
smat.convert(PETSc.Mat.Type.AIJ)

assert sarray_r.shape == Sinv.shape
assert np.allclose(sarray_r, Sinv)

parameters={
    "ksp_type": "cg" if symmetric else "gmres",
    "ksp_max_it": 100,
    "ksp_rtol": 1e-8,
    "ksp_atol": 1e-8,
    "pc_type": "hypre",
    "ksp_reuse_preconditioner": None,
    # "ksp_converged_reason": None,
}

N = n*p
mat_kron = PETSc.Mat().createPython(
    size=((N, N), (N, N)),
    context=KroneckerProductMat(amat, smat)
)

# Create rhs matrix
b0 = amat.createVecRight()
b0.array[:] = np.random.randn(n, 1).flatten()
beta = b0.norm()

b, x = mat_kron.createVecs()
b.zeroEntries()
b.array[:n] = b0.array

X = eksm(mat_kron, b, parameters, max_it=100)

ksp_kron = PETSc.KSP().create()
ksp_kron.setOperators(mat_kron)
set_from_options(
    ksp_kron, options_prefix="kron",
    parameters={
        "ksp_monitor": None,
        "ksp_converged_reason": None,
        "ksp_rtol": 1e-8,
        "ksp_type": "gmres",
        "pc_type": "none",

        # "ksp_type": "python",
        # "ksp_python_type": "eksm.ShiftedComplexKSP",
        # "ksp_max_it": 160,
        # "ksp_rtol": 1e-8,
        # "sub": {
        #     "ksp_reuse_preconditioner": None,
        #     "ksp_max_it": 100,
        #     "ksp_rtol": 1e-8,
        #     "ksp_atol": 1e-8,
        #     "ksp_type": "cg" if symmetric else "gmres",
        #     "pc_type": "hypre",
        # }
    }
)
# ksp_kron.solve(b, x)
Xkron = np.empty_like(X)
xr = x.array_r.reshape(p, n)
for k in range(p):
    Xkron[:, k] = xr[k, :]

# Check true residual norms
norms = np.zeros((p,1))
R = A@X+X@Sinv
for k in range(p):
    norms[k] = np.linalg.norm(b0.array_r.T-R[:,k])/beta
print(f"Maximum of true residual norms: {max(np.abs(norms))[0]:.6e}")

norms = np.zeros((p,1))
R = A@Xkron+Xkron@Sinv
for k in range(p):
    norms[k] = np.linalg.norm(b0.array_r.T-R[:,k])/beta
print(f"Maximum of true residual norms: {max(np.abs(norms))[0]:.6e}")

