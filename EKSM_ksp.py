import petsctools
PETSc = petsctools.init()

from math import sqrt, sin, cos, pi
import numpy as np
from scipy.sparse import eye, spdiags, kron, csr_matrix
from scipy.sparse.linalg import splu, LinearOperator, gmres
from scipy.linalg import solve_sylvester, norm
import butchertableau as bt

from eksm import eksm, KroneckerProductMat
from problem import Amat, Smat

np.random.seed(6)

options = PETSc.Options()

atol = options.getReal("atol", 1e-8)
m = options.getInt("nx", 10)
symmetric = options.getBool("symmetric", True)
re = options.getReal("re", 10)
angle = options.getReal("angle", pi/3)
order = options.getInt("order", 8)
max_it = options.getInt("max_it", 100) # maximum iteration count
m_krylov = max_it + 1

n = m * m # number of unknowns

# Block matrix
A, amat = Amat(m, re, angle=angle, symmetric=symmetric, petsc=True)

# Butcher tableau matrix
S, Sinv = Smat(order)
p = Sinv.shape[0]

sinv = PETSc.Mat().createDense(
    size=((p, p), (p, p)),
    array=Sinv
)
sinv.convert(PETSc.Mat.Type.AIJ)
sinv.setUp()
sinv.assemble()

Aksp = PETSc.KSP().create()
Aksp.setOperators(amat)

block_params = {
    "ksp_type": "cg" if symmetric else "gmres",
    "ksp_max_it": 100,
    "ksp_rtol": 1e-10,
    "pc_type": "hypre",
}

petsctools.set_from_options(
    Aksp, options_prefix="A",
    parameters=block_params,
)

# Create single rhs data
bdata = np.random.randn(n, 1)

# rhs vector
b = amat.createVecRight()
b.array[:] = bdata.flatten()

kronmat = PETSc.Mat().createPython(
    size=((n*p, n*p), (n*p, n*p)),
    context=KroneckerProductMat(amat, sinv),
)
kronmat.setUp()
kronmat.assemble()

X = eksm(kronmat, Aksp, b, m_krylov=m_krylov, rtol=atol)

# Check true residual norms
norms = np.zeros((p,1))
R = A@X+X@Sinv
beta = norm(b)
for k in range(p):
    norms[k] = np.linalg.norm(bdata.T-R[:,k])/beta
print(f"eksm: Maximum of true residual norms: {max(np.abs(norms))[0]:.6e}")
print(f"eksm: True residual norms:\n{np.abs(norms)}")

# KSP for the full kronecker matrix
kronksp = PETSc.KSP().create()
kronksp.setOperators(kronmat)
petsctools.set_from_options(
    kronksp, options_prefix="kron",
    parameters={
        "ksp_converged_reason": None,
        "ksp_monitor": None,
        "ksp_atol": atol,
        "ksp_max_it": max_it,
        "ksp_type": "python",
        "ksp_python_type": "eksm.SylvesterEKSP",
        "sylvester": block_params,
    }
)
bfull = kronmat.getPythonContext().vec_nest.duplicate()
xfull = kronmat.getPythonContext().vec_nest.duplicate()

# duplicate b into all blocks of the full rhs
bsubs = bfull.getNestSubVecs()
for bi in bsubs:
    b.copy(result=bi)
bfull.setNestSubVecs(bsubs)

with petsctools.inserted_options(kronksp):
    kronksp.solve(bfull, xfull)

# extract solution for each block
Xkron = np.empty_like(X)
xsubs = xfull.getNestSubVecs()
for i, xi in enumerate(xsubs):
    Xkron[:, i] = xi.array_r

R = A@Xkron+Xkron@Sinv
beta = norm(b)
for k in range(p):
    norms[k] = np.linalg.norm(bdata.T-R[:,k])/beta

print(f"ksp: Maximum of true residual norms: {max(np.abs(norms))[0]:.6e}")
print(f"ksp: True residual norms:\n{np.abs(norms)}")
