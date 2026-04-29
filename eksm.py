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
    def __init__(self, A, S, d=None):
        self.A = A
        self.S = S

        if d is None:
            sinv.createVecRight()
            d.array[:] = 1
        self.d = d

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


def eksm(kronmat, Aksp, b, *, kronksp=None, m_krylov=None, atol=None, adaptive_tol=False):
    kron = kronmat.getPythonContext()
    amat, smat = kron.A, kron.S

    if kronksp:
        atol = kronksp.atol
        m_krylov = kronksp.max_it + 1
        kronksp.its = 0
    else:
        assert atol is not None
        assert m_krylov is not None

    n = amat.size[0]
    p = smat.size[0]

    # extract dense numpy array for S
    smat.convert(PETSc.Mat.Type.DENSE)
    S = smat.getDenseArray(readonly=True).copy()
    smat.convert(PETSc.Mat.Type.AIJ)

    darr = kron.d.array_r

    # Set Extended Krylov space quantities
    X = np.zeros((n, p))  # solution matrix
    Varray = np.zeros((n, 2 * m_krylov + 2))  # matrix holding basis vectors
    H = np.zeros((2 * m_krylov + 1, 2 * m_krylov))  # to hold projected problem
    c = np.zeros((2 * m_krylov + 1, 1))  # projected rhs
    y = np.zeros((2 * m_krylov, p))  # projected solution

    # First basis vector (orthogonalised rhs)
    beta = b.norm()
    c[0, 0] = beta
    w = b.copy()
    w /= beta

    V = OrthonormalBasis(w)

    # New vector (multiply previous one by A then orthogonalise)
    with petsctools.inserted_options(Aksp):
        Aksp.solve(V[-1], w)
    _, normwp, hp = V.append(w)
    hp = np.append(hp, normwp)

    if adaptive_tol:
        Aksp_rtol0 = Aksp.rtol

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
            H[:temp, :temp], S,
            np.outer(c[:temp], darr[:p]))

        # Check residual norm
        r = H[temp:temp+1, :temp] @ y
        rnorms = np.zeros((p,1))
        for k in range(p):
            rnorms[k] = norm(r[:,k]) / ((np.abs(darr[k])+2*np.finfo(darr[k]).eps) * beta)
        rnorm = norm(rnorms)

        if kronksp:
            kronksp.monitor(i, rnorm)
            kronksp.logConvergenceHistory(rnorm)
            kronksp.its += 1
            kronksp.norm = rnorm
        else:
            print(f"|r_{i}|: {rnorm:.6e} \\ max |r_{i}|: {max(rnorms)[0]:.6e}")

        # kronksp.callConvergenceTest(i, rnorm)
        if rnorm < atol:
            if kronksp:
                kronksp.setConvergedReason(
                    PETSc.KSP.ConvergedReason.CONVERGED_ATOL)
            else:
                print(f"Residual norm absolute tolerance reached.")
            break

        elif i >= m_krylov - 1:
            if kronksp:
                kronksp.setConvergedReason(
                    PETSc.KSP.ConvergedReason.CONVERGED_ITS)
            else:
                print(f"Maximum iterations reached.")
            break

        if adaptive_tol:
            maxNormR = np.max(rnorms)
            Aksp.setTolerances(rtol=min(Aksp_rtol0/maxNormR, 0.1))

    if adaptive_tol:
        Aksp.setTolerances(rtol=Aksp_rtol0)

    # Solution recovery
    for i, v in enumerate(V.vectors):
        Varray[:, i] = v.array_r
    X = Varray[:, :temp]@y
    return X

def block_eksm(kronmat, Aksp, b, m_krylov, rtol):
    kronctx = kronmat.getPythonContext()
    amat, smat = kronctx.A, kronctx.S

    n = amat.size[0]
    p = smat.size[0]
    bs = len(b)
    if p != bs:
        raise ValueError("Number of columns in b must match size of S. This is to be fixed when we consider adding" \
        "an array d so that we are solving AX+XS = b d^T")

    # extract dense numpy array for S
    smat.convert(PETSc.Mat.Type.DENSE)
    S = smat.getDenseArray(readonly=True).copy()
    smat.convert(PETSc.Mat.Type.AIJ)

    # Set Extended Krylov space quantities
    X = np.zeros((n,p)) # solution matrix
    Varray = np.zeros((n, (2 * m_krylov + 2)*bs)) # matrix holding basis vectors
    H = np.zeros((bs * (2 * m_krylov + 1), bs * (2 * m_krylov))) # to hold projected problem
    L = np.zeros((bs * (2 * m_krylov + 1), bs * (2 * m_krylov))) # to hold projected problem
    c = np.zeros((bs * (2 * m_krylov + 1), bs * (1           ))) # projected rhs
    y = np.zeros((bs * (2 * m_krylov    ), bs * (p           ))) # projected solution

    # First basis vector (orthogonalised rhs)
    beta = b[0].norm()
    c[0,0] = beta
    w = b[0].copy()
    w /= beta

    V = OrthonormalBasis(w)
    for j in range(1, bs):
        _, normwp, h = V.append(b[j])
        c[0:j,j] = h
        c[j,j] = normwp

    # New vector (multiply previous one by A then orthogonalise)
    for j in range(bs):
        with petsctools.inserted_options(Aksp):
            Aksp.solve(V[j], w)
        _, normw, h = V.append(w)
        L[:len(h), j] = h
        L[len(h), j] = normw
        H[j, j] = 1

    Aksprtol = Aksp.rtol
    for i in range(m_krylov):
        # New basis vector (obtained by mult by A)
        for j in range(bs):
            amat.mult(V[-bs+j], w)
            _, normw, h = V.append(w)
            H[:len(h), (2*i+1)*bs+j] = h
            H[len(h), (2*i+1)*bs+j] = normw
            L[(2*i+1)*bs+j, (2*i+1)*bs+j] = 1

        # print(f"L:\n{L[:(2*i+2)*bs+bs, :(2*i+2)*bs]}")
        # print(f"H:\n{H[:(2*i+2)*bs+bs, :(2*i+2)*bs]}")
        # Move on to add next set of basis vectors (obtained by mult by A^-1)
        for j in range(bs):
            with petsctools.inserted_options(Aksp):
                Aksp.solve(V[-bs+j], w)
            _, normw, h = V.append(w)
            L[:len(h), (2*i+2)*bs+j] = h
            L[len(h), (2*i+2)*bs+j] = normw
            H[(2*i+2)*bs+j, (2*i+2)*bs+j] = 1

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
        temp = (2*i+2)*bs
        # K = H[:(2*i+2)*bs, :(2*i+2)*bs]/L[:(2*i+2)*bs, :(2*i+2)*bs]
        K = np.linalg.solve(L[:temp, :temp].T, H[:temp+bs, :temp].T).T
        y = solve_sylvester(
            K[:temp,:temp], S,
            c[:temp,:bs])

        # Check residual norm
        r = K[temp:temp+bs,:temp] @ y
        rnorms = np.zeros((p,1))
        for k in range(p):
            rnorms[k] = norm(r[:,k]) / (norm(c[:bs,:bs]*d[:bs,k]))

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

class SylvesterEKSP:
    """(Is*A + S*Ia)x = b <=> AX + XS = B
    """

    prefix = "sylvester_"

    def setUp(self, ksp):
        self.mat, _ = ksp.getOperators()
        kron = self.mat.getPythonContext()

        prefix = ksp.getOptionsPrefix() or ''
        self.Aksp = PETSc.KSP().create()
        self.Aksp.setOperators(kron.A)

        petsctools.set_from_options(
            self.Aksp, options_prefix=prefix + self.prefix)

        self.adaptive_tol = PETSc.Options().getBool(
            f"{prefix}ksp_{self.prefix}adaptive_tol", False)

    def solve(self, ksp, b, x):
        kron = self.mat.getPythonContext()

        bnest = kron.vec_nest.duplicate()
        b.copy(result=bnest)
        b0 = kron.A.createVecRight()
        bnest.getNestSubVecs()[0].copy(result=b0)

        b0.scale(1/kron.d.array_r[0])

        X = eksm(self.mat, self.Aksp, b0, kronksp=ksp,
                 adaptive_tol=self.adaptive_tol)

        xnest = kron.vec_nest.duplicate()
        xsubs = xnest.getNestSubVecs()
        for k, xsub in enumerate(xsubs):
            xsub.array[:] = X[:, k]
        xnest.setNestSubVecs(xsubs)
        xnest.copy(result=x)

    def view(self, ksp, viewer=None):
        if viewer is None:
            return
        if viewer.type != PETSc.Viewer.Type.ASCII:
            return
        viewer.printfASCII(
            "Extended Krylov method for solving a Sylvester equation AX+SX=B.\n")
        if self.adaptive_tol:
            viewer.printfASCII(
                "  Using an adaptive tolerance for the KSP for A.\n")
        viewer.printfASCII(
            "  The KSP for solving A is:\n")
        viewer.pushASCIITab()
        self.Aksp.view(viewer)
        viewer.popASCIITab()
