import petsctools
PETSc = petsctools.init()

from math import pi
import numpy as np
from scipy.linalg import solve_sylvester, norm
import butchertableau as bt

from eksm import OrthonormalBasis
from problem import Amat, Smat

np.random.seed(6)

rtol = 1e-8

# Initialize
m = 10 # grid size
symmetric = True
re = 10
angle = pi/3
order = 8
m_krylov = 160 # maximum iteration count

n = m * m # number of unknowns

# Block matrix
A, amat = Amat(m, re, angle=angle, symmetric=symmetric, petsc=True)

# Butcher tableau matrix
S, Sinv = Smat(order)
p = Sinv.shape[0]

smat = PETSc.Mat().createDense(
    size=((p, p), (p, p)),
    array=Sinv
)
smat.convert(PETSc.Mat.Type.AIJ)

ksp = PETSc.KSP().create()
ksp.setOperators(amat)

petsctools.set_from_options(
    ksp, options_prefix="",
    parameters={
        "ksp_type": "cg" if symmetric else "gmres",
        "ksp_max_it": 100,
        "ksp_rtol": 1e-8,
        "ksp_atol": 1e-8,
        "pc_type": "hypre",
        "ksp_reuse_preconditioner": None,
        # "ksp_converged_reason": None,
    }
)

w = amat.createVecRight()
ksprtol = ksp.getTolerances()[0]

# Set Extended Krylov space quantities
X = np.zeros((n,p)) # solution matrix
Varray = np.zeros((n, 2 * m_krylov + 2)) # matrix holding basis vectors
H = np.zeros((2 * m_krylov + 1, 2 * m_krylov)) # to hold projected problem
c = np.zeros((2 * m_krylov + 1, 1)) # projected rhs
y = np.zeros((2 * m_krylov, p)) # projected solution

# Create single rhs data
bdata = np.random.randn(n, 1)

# rhs vector
b = bdata

# First basis vector (orthogonalised rhs)
beta = norm(b)
c[0,0] = beta
w.array[:] = b.flatten()/beta

V = OrthonormalBasis(w)

# New vector (multiply previous one by A then orthogonalise)
with petsctools.inserted_options(ksp):
    ksp.solve(V[-1], w)
_, normwp, hp = V.append(w)
hp = np.append(hp, normwp)

for i in range(m_krylov):
    # New basis vector (obtained by mult by A)
    amat.mult(V[-1], w)
    _, normw, h = V.append(w)

    # Put projection data in H
    ind = len(V) - 2
    H[:ind+1, ind] = h
    H[ind+1, ind] = normw

    # Put projection data for previous set, see note below
    H[:ind, ind - 1] = -H[:ind, :ind - 1] @ hp[:ind - 1]
    H[ind - 1, ind - 1] += 1
    H[:ind + 2, ind - 1] -= np.append(h, normw) * normwp
    H[:ind + 2, ind - 1] /= hp[ind - 1]

    # Move on to add next set of basis vectors (obtained by mult by A^-1)
    with petsctools.inserted_options(ksp):
        ksp.solve(V[-1], w)
    _, normwp, hp = V.append(w)

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
    y = solve_sylvester(
        H[:temp,:temp], Sinv,
        np.outer(c[:temp], np.ones(p)))
    
    # Check residual norm
    r = H[temp:temp+p,:temp] @ y
    rnorms = np.zeros((p,1))
    for k in range(p):
        rnorms[k] = norm(r[:,k])/beta

    print(f"max r_{i}: {max(rnorms)[0]:.6e}")
    if(np.max(rnorms) < rtol):
        break
    if PETSc.Options().getBool("adaptive_rtol", False):
        maxNormR = np.max(rnorms)
        ksp.setTolerances(rtol=min(ksp.rtol/maxNormR, 0.1))
    
# Solution recovery    
for i, v in enumerate(V.vectors):
    Varray[:, i] = v.array_r
X = Varray[:,:temp]@y

# Check true residual norms
norms = np.zeros((p,1))
R = A@X+X@Sinv
for k in range(p):
    norms[k] = np.linalg.norm(bdata.T-R[:,k])/beta
print(f"Maximum of true residual norms: {max(np.abs(norms))[0]:.6e}")
print(f"True residual norms:\n{np.abs(norms)}")

