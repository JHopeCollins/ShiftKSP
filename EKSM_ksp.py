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

rtol = 1e-8

# Initialize
ref = PETSc.Options().getInt("ref", 3)
m = 10 * 2**ref # grid size
symmetric = False
re = PETSc.Options().getReal("re", 1e0)
angle = pi/3
order = PETSc.Options().getInt("order", 8)
m_krylov = 160 # maximum iteration count

n = m * m # number of unknowns
# Block matrix
A, amat = Amat(m, re, angle=angle, symmetric=symmetric, petsc=True)

# Butcher tableau matrix
S, Sinv = Smat(order)
p = Sinv.shape[0]

print(f"Problem size: {n} unknowns, {p} stages in S")

sinv = PETSc.Mat().createDense(
    size=((p, p), (p, p)),
    array=Sinv
)
sinv.convert(PETSc.Mat.Type.AIJ)
sinv.setUp()
sinv.assemble()

ksp = PETSc.KSP().create()
ksp.setOperators(amat)

petsctools.set_from_options(
    ksp, options_prefix="",
    parameters={
        "ksp_type": "cg" if symmetric else "gmres",
        "ksp_max_it": 100,
        "ksp_rtol": rtol*1e-2,
        "pc_type": "hypre",
        "ksp_reuse_preconditioner": None,
        # "ksp_converged_reason": None,
        # "ksp_monitor_true_residual": None,
    }
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

ddata = np.random.randn(p, 1)
d = sinv.createVecRight()
d.array[:] = ddata.flatten()

X = eksm(kronmat, ksp, b, d, m_krylov, rtol)

# Check true residual norms
norms = np.zeros((p,1))
R = A@X+X@Sinv
beta = norm(b)
for k in range(p):
    norms[k] = np.linalg.norm(d.array_r[k]*bdata.T-R[:,k])/((np.abs(d.array_r[k]) + 2 * np.finfo(d.array_r[k]).eps) * beta)
print(f"eksm: Maximum of true residual norms: {max(np.abs(norms))[0]:.6e}")
print(f"eksm: True residual norms:\n{np.abs(norms)}")

# KSP for the full kronecker matrix
kronksp = PETSc.KSP().create()
kronksp.setOperators(kronmat)
petsctools.set_from_options(
    kronksp, options_prefix="kron",
    parameters={
        "ksp_converged_reason": None,
        "ksp_rtol": rtol,
        "ksp_type": "gmres",
        "pc_type": "none",
    }
)
bfull = kronmat.getPythonContext().vec_nest.duplicate()
xfull = kronmat.getPythonContext().vec_nest.duplicate()

# duplicate b into all blocks of the full rhs
bsubs = bfull.getNestSubVecs()
for i, bi in enumerate(bsubs):
    b.copy(result=bi)
    bi.scale(d.array_r[i]) # scale by d[i]
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
    norms[k] = np.linalg.norm(d.array_r[k]*bdata.T-R[:,k])/((np.abs(d.array_r[k]) + 2 * np.finfo(d.array_r[k]).eps) * beta)

print(f"ksp: Maximum of true residual norms: {max(np.abs(norms))[0]:.6e}")
print(f"ksp: True residual norms:\n{np.abs(norms)}")

for k in range(p):
    norms[k] = np.linalg.norm(X[:,k]-Xkron[:,k])

print(f"Error norms:\n{np.abs(norms)}")