import numpy as np
from scipy.linalg import solve_sylvester, norm
import petsctools
from petsc4py import PETSc

__all__ = ["OrthonormalBasis", "eksm", "KroneckerProductMat"]


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


class OrthonormalBasis:
    def __init__(self, vectors=None, *, comm=None):
        from petsc4py import PETSc
        self._comm = comm or PETSc.COMM_WORLD
        self._vectors = []
        self._allocated = 0
        self._size = 0

        if isinstance(vectors, PETSc.Vec):
            v = vectors.copy()
            v.normalize()
            self._append(v)
        elif vectors is not None:
            v = vectors[0].copy()
            v.normalize()
            self._append(v)
            for v in vectors[1:]:
                self.append(v)

    def __getitem__(self, i):
        return self._vectors[i]

    def __len__(self):
        return self._size

    @property
    def allocated(self):
        return len(self._vectors)

    @property
    def vectors(self):
        return self._vectors

    @property
    def comm(self):
        return self._comm

    def _append(self, v):
        if len(self._vectors) and not self.is_compatible(v):
            raise ValueError(
                "Vec is does not have compatible"
                " size for OrthonormalBasis")
        if self._size < len(self._vectors):
            v = v.copy(self._vectors[self._size])
        else:
            self._vectors.append(v)
        self._size += 1
        return v

    def is_compatible(self, v):
        if len(self._vectors) == 0:
            return True
        if v.sizes != self._vectors[0].sizes:
            return False
        return True

    def create_vector(self):
        return self.vectors[0].duplicate()

    def append(self, v):
        if len(self) == 0:
            v = v.copy()
            normw = v.normalize()
            v = self._append(v)
            return v, normw, None

        V = self.vectors
        h = mdot(V, v)
        v = v - dmult(V, h)
        h1 = mdot(V, v)
        v = v - dmult(V, h1)
        h += h1
        normw = v.normalize()
        v = self._append(v)
        return v, normw, h


class KroneckerProductMat:
    """Apply: S*I_{a} + I_{s}*A  (* is kronecker product)
    """
    def __init__(self, A, S):
        self.A = A
        self.S = S

        self.S.convert(PETSc.Mat.Type.DENSE)
        self.Sa = S.getDenseArray(readonly=True).copy()
        self.S.convert(PETSc.Mat.Type.AIJ)

        self.vec_nest = PETSc.Vec().createNest(
            vecs=[A.createVecRight()
                  for _ in range(S.sizes[0][0])],
            comm=A.comm
        )

    def mult(self, mat, x, y):
        xn = self.vec_nest.duplicate()
        yn = self.vec_nest.duplicate()
        w = self.A.createVecRight()

        x.copy(result=xn)
        y.zeroEntries()

        xsubs = xn.getNestSubVecs()
        ysubs = yn.getNestSubVecs()
        Sa = self.Sa

        for i in range(Sa.shape[0]):
            xi = xsubs[i]
            yi = ysubs[i]
            self.A.mult(xi, w)
            yi += w
            for j in range(Sa.shape[1]):
                yi += float(Sa[j, i])*xsubs[j]

        yn.setNestSubVecs(ysubs)
        yn.copy(result=y)


def eksm(kronmat, Aksp, b, d, m_krylov, rtol):
    kronctx = kronmat.getPythonContext()
    amat, smat = kronctx.A, kronctx.S

    n = amat.size[0]
    p = smat.size[0]

    # extract dense numpy array for S
    smat.convert(PETSc.Mat.Type.DENSE)
    S = smat.getDenseArray(readonly=True).copy()
    smat.convert(PETSc.Mat.Type.AIJ)

    # Set Extended Krylov space quantities
    X = np.zeros((n,p)) # solution matrix
    Varray = np.zeros((n, 2 * m_krylov + 2)) # matrix holding basis vectors
    H = np.zeros((2 * m_krylov + 1, 2 * m_krylov)) # to hold projected problem
    c = np.zeros((2 * m_krylov + 1, 1)) # projected rhs
    y = np.zeros((2 * m_krylov, p)) # projected solution

    # First basis vector (orthogonalised rhs)
    beta = b.norm()
    c[0,0] = beta
    w = b.copy()
    w /= beta

    V = OrthonormalBasis(w)

    # New vector (multiply previous one by A then orthogonalise)
    with petsctools.inserted_options(Aksp):
        Aksp.solve(V[-1], w)
    _, normwp, hp = V.append(w)
    hp = np.append(hp, normwp)

    Aksprtol = Aksp.rtol
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
        with petsctools.inserted_options(Aksp):
            Aksp.solve(V[-1], w)
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
            H[:temp,:temp], S,
            np.outer(c[:temp], d.array_r[:p]))

        # Check residual norm
        r = H[temp:temp+1,:temp] @ y
        rnorms = np.zeros((p,1))
        for k in range(p):
            rnorms[k] = norm(r[:,k]) / ((np.abs(d.array_r[k])+2*np.finfo(d.array_r[k]).eps) * beta)

        print(f"max r_{i}: {max(rnorms)[0]:.6e}")
        if(np.max(rnorms) < rtol):
            break
        if PETSc.Options().getBool("adaptive_rtol", False):
            maxNormR = np.max(rnorms)
            Aksp.setTolerances(rtol=min(Aksprtol/maxNormR, 0.1))

    # Solution recovery
    for i, v in enumerate(V.vectors):
        Varray[:, i] = v.array_r
    X = Varray[:,:temp]@y
    return X


class ShiftedExtendedKSP:
    """Solve: S*I_{a} + I_{s}*A  (* is kronecker product)
    """

    prefix = "shift_"

    def setUp(self, ksp, b, x):
        Akron, Pkron = ksp.getOperators()

        A = Akron.getMat(0, 1)
        S = Akron.getMat(1, 0)

        self.n, _ = A.getSizes()
        self.p, _ = S.getSizes()

        self.s = S.getDiagonal().array_r

        Ap = Pkron.getMat(0, 1)

        prefix = ksp.getOptionsPrefix() or ""
        self.Aksp = PETSc.KSP().create()
        self.Aksp.setOperators(A, Ap)
        self.Aksp.setPrefix(prefix+self.prefix)
        self.Aksp.setFromOptions()
        self.Aksp.setUp()

        self.adaptive_rtol = PETSc.Options().getBool(
            f"{prefix}ksp_{self.prefix}adaptive_rtol")

    def solve(self, ksp, b, x):

        X = eksm(self.P, self.Aksp, b,
                 ksp.max_it, ksp.rtol)
