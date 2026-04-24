from math import pi
import numpy as np
from scipy.linalg import norm

import petsctools
PETSc = petsctools.init()

from eksm import KroneckerProductMat
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
    array=Sinv,
)
smat.convert(PETSc.Mat.Type.AIJ)
smat.setUp()
smat.assemble()

kronmat = PETSc.Mat().createPython(
    size=((n*p, n*p), (n*p, n*p)),
    context=KroneckerProductMat(amat, smat),
)

x = kronmat.getPythonContext().vec_nest.duplicate()
b = kronmat.getPythonContext().vec_nest.duplicate()

# random rhs on each block
b.array[:] = np.random.rand(n*p)
kronmat.mult(b, x)

B = np.empty((n, p))
bsubs = b.getNestSubVecs()
for i in range(p):
    B[:, i] = bsubs[i].array_r
X = A@B + B@Sinv

xsubs = x.getNestSubVecs()
for i in range(p):
    xm = xsubs[i].array
    xa = X[:, i]
    print(f"{i=}: {norm(xm - xa) = }")
