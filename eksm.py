import numpy as np
from scipy.linalg import solve_sylvester, norm
import petsctools
from petsc4py import PETSc
Print = PETSc.Sys.Print

__all__ = ["OrthonormalBasis", "eksm", "KroneckerProductMat"]


def matfree2dense(mat):
    array = np.zeros(mat.sizes[0])
    x, y = mat.createVecs()
    for i in range(array.shape[0]):
        x.array[:] = 0
        x.array[i] = 1
        mat.mult(x, y)
        array[:, i] = y.array_r
    dense = PETSc.Mat().createDense(
        size=mat.sizes, array=array)
    return dense


def vecs2numpy(vecs, arr=None):
    if arr is None:
        arr = np.zeros((vecs[0].size, len(vecs)))
    for i, v in enumerate(vecs):
        arr[:, i] = v.array_r
    return arr


def numpy2vecs(vecs, arr):
    for i, v in enumerate(vecs):
        v.array[:] = arr[:, i]
    return vecs


def numpy2vecnest(vec, arr):
    subvecs = vec.getNestSubVecs()
    numpy2vecs(subvecs, arr)
    vec.setNestSubVecs(subvecs)
    return vec


def mdot(x, y):
    n = len(x)
    r = np.ndarray(n)
    for i in range(n):
        r[i] = y.dot(x[i])
    return r


def dmult(x, y, result=None):
    if result is None:
        result = x[0].duplicate()
    result.zeroEntries()
    for i, alpha in enumerate(y):
        result.axpy(alpha, x[i])
    return result


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


def extend_basis(op, V, vecs, nprev, H, L=None, wrk=None):
    if wrk is None:
        wrk = vecs[0].duplicate()
    wrk.zeroEntries()

    offset = nprev*len(vecs)

    for j, v in enumerate(vecs):
        op(v, wrk)
        _, normw, h = V.append(wrk)
        if h is not None:
            H[:len(h), offset+j] = h
            H[len(h), offset+j] = normw
        else:
            H[0, 0] = normw
        if L is not None:
            L[offset+j, offset+j] = 1


class KroneckerProductMat:
    """Apply: S*M + I_{s}*A  (* is kronecker product)

    M defaults to I_{a}
    """
    def __init__(self, A, S, M=None, d=None):
        self.A = A
        self.S = S

        if d is None:
            d = S.createVecRight()
            d.array[:] = 1
        self.d = d

        self.S.convert(PETSc.Mat.Type.DENSE)
        self.Sa = S.getDenseArray(readonly=True).copy()
        self.S.convert(PETSc.Mat.Type.AIJ)

        self.M = M or PETSc.Mat().createConstantDiagonal(
            size=A.sizes, diag=1.0, comm=A.comm)

        self.vec_nest = PETSc.Vec().createNest(
            vecs=[A.createVecRight()
                  for _ in range(S.sizes[0][0])],
            comm=A.comm
        )
        self.vec_nest.setUp()
        self.vec_nest.assemble()

    def mult(self, mat, x, y):
        xn = self.vec_nest.duplicate()
        yn = self.vec_nest.duplicate()
        Mx = self.vec_nest.duplicate()
        Ax = self.vec_nest.duplicate()

        x.copy(result=xn)
        y.zeroEntries()

        xsubs = xn.getNestSubVecs()
        ysubs = yn.getNestSubVecs()
        Mxs = Mx.getNestSubVecs()
        Axs = Ax.getNestSubVecs()

        for xi, Mxi, Axi in zip(xsubs, Mxs, Axs):
            self.A.mult(xi, Axi)
            self.M.mult(xi, Mxi)

        Sa = self.Sa
        for i in range(Sa.shape[0]):
            ysubs[i] += Axs[i]
            for j in range(Sa.shape[1]):
                ysubs[i].axpy(float(Sa[i, j]), Mxs[j])

        yn.setNestSubVecs(ysubs)
        PETSc.Vec.concatenate(ysubs)[0].copy(result=y)

    # def view(self, mat, viewer=None):
    #     pass


def eksm(kronmat, Aksp, b, *, kronksp=None, m_krylov=None, atol=None, adaptive_tol=False, xvec=None):
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
    S = smat.getDenseArray(readonly=True).copy().T
    smat.convert(PETSc.Mat.Type.AIJ)

    darr = kron.d.array_r

    # Set Extended Krylov space quantities
    X = np.zeros((n, p))  # solution matrix
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
            eps = np.finfo(darr[k]).eps
            rnorms[k] = norm(r[:,k]) / ((np.abs(darr[k])+2*eps) * beta)
        rnorm = norm(rnorms)

        if kronksp:
            kronksp.monitor(i, rnorm)
            kronksp.logConvergenceHistory(rnorm)
            kronksp.its += 1
            kronksp.norm = rnorm
            kronksp.callConvergenceTest(i, rnorm)
        else:
            Print(f"|r_{i}|: {rnorm:.6e} \\ max |r_{i}|: {max(rnorms)[0]:.6e}")

        if rnorm < atol:
            if kronksp:
                kronksp.setConvergedReason(
                    PETSc.KSP.ConvergedReason.CONVERGED_ATOL)
            else:
                Print(f"Residual norm absolute tolerance reached.")
            break

        elif i >= m_krylov - 1:
            if kronksp:
                kronksp.setConvergedReason(
                    PETSc.KSP.ConvergedReason.CONVERGED_ITS)
            else:
                Print(f"Maximum iterations reached.")
            break

        if adaptive_tol:
            maxNormR = np.max(rnorms)
            Aksp.setTolerances(rtol=min(Aksp_rtol0/maxNormR, 0.1))

    if adaptive_tol:
        Aksp.setTolerances(rtol=Aksp_rtol0)

    # Solution recovery
    X = vecs2numpy(V[:-2])@y
    if xvec:
        X = numpy2vecnest(xvec, X)
    return X


def block_eksm(kronmat, Aksp, b, d, m_krylov=None, rtol=None, xvec=None, kronksp=None, adaptive_tol=False):
    kronctx = kronmat.getPythonContext()
    amat, smat = kronctx.A, kronctx.S

    if kronksp:
        rtol = kronksp.atol
        m_krylov = kronksp.max_it + 1
        kronksp.its = 0

    n = amat.size[0]
    p = smat.size[0]
    bs = len(b)
    # if p != bs:
    #     raise ValueError("Number of columns in b must match size of S. This is to be fixed when we consider adding" \
    #     "an array d so that we are solving AX+XS = b d^T")

    # extract dense numpy array for S
    smat.convert(PETSc.Mat.Type.DENSE)
    S = smat.getDenseArray(readonly=True).copy().T
    smat.convert(PETSc.Mat.Type.AIJ)

    # Set Extended Krylov space quantities
    X = np.zeros((n,p)) # solution matrix
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

    if adaptive_tol:
        Aksp_rtol0 = Aksp.rtol

    for i in range(m_krylov):
        # New basis vector (obtained by mult by A)
        for j in range(bs):
            amat.mult(V[-bs], w)
            # amat.mult(V[(2*i+1)*bs+j], w)
            _, normw, h = V.append(w)
            H[:len(h), (2*i+1)*bs+j] = h
            H[len(h), (2*i+1)*bs+j] = normw
            L[(2*i+1)*bs+j, (2*i+1)*bs+j] = 1

        # Move on to add next set of basis vectors (obtained by mult by A^-1)
        for j in range(bs):
            with petsctools.inserted_options(Aksp):
                Aksp.solve(V[-bs], w)
                # Aksp.solve(V[(2*i+2)*bs+j], w)
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
            c[:temp,:bs] @ d[:bs,:])

        # Check residual norm
        r = K[temp:temp+bs,:temp] @ y
        rnorms = np.zeros((p,1))
        for k in range(p):
            eps = np.finfo(norm(d[:bs,k])).eps
            rnorms[k] = norm(r[:,k]) / (norm(c[:bs,:bs] @ d[:bs,k]) * (norm(d[:bs,k])+2*eps))
        rnorm = norm(rnorms)

        if kronksp:
            kronksp.monitor(i, rnorm)
            kronksp.logConvergenceHistory(rnorm)
            kronksp.its += 1
            kronksp.norm = rnorm
            kronksp.callConvergenceTest(i, rnorm)
        else:
            Print(f"|r_{i}|: {rnorm:.6e} \\ max |r_{i}|: {max(rnorms)[0]:.6e}")

        if rnorm < rtol:
            if kronksp:
                kronksp.setConvergedReason(
                    PETSc.KSP.ConvergedReason.CONVERGED_ATOL)
            else:
                Print(f"Residual norm absolute tolerance reached.")
            break

        elif i >= m_krylov - 1:
            if kronksp:
                kronksp.setConvergedReason(
                    PETSc.KSP.ConvergedReason.CONVERGED_ITS)
            else:
                Print(f"Maximum iterations reached.")
            break

        if adaptive_tol:
            maxNormR = np.max(rnorms)
            Aksp.setTolerances(
                rtol=max(Aksp_rtol0, min(Aksp_rtol0/maxNormR, 0.1)))

    if adaptive_tol:
        Aksp.setTolerances(rtol=Aksp_rtol0)

    # Solution recovery
    X = vecs2numpy(V[:temp])@y
    if xvec:
        X = numpy2vecnest(xvec, X)
    return X


class SylvesterEKSP:
    """(S*M + Is*A)x = b <=> (XS^T + Ahat*X) = M^{-1}B; Ahat=M^{-1}A
    PETSc Options
    -------------

    -ksp_sylvester_adaptive_tol : adaptive ksp_rtol for A?
    -ksp_sylvester_aksp_max_rtol : maximum adaptive ksp_rtol for A
    -sylvester_A_ ... : options for A^{-1}
    -sylvester_M_ ... : options for M^{-1}
    """

    prefix = "sylvester_"

    def setUp(self, ksp):
        kron = ksp.mat_op.getPythonContext()

        prefix = ksp.getOptionsPrefix() or ''

        self.Aksp = PETSc.KSP().create()
        self.Aksp.setOperators(kron.A)
        self.Aksp.setOptionsPrefix(prefix + self.prefix + "A_")
        self.Aksp.setFromOptions()
        self.Aksp.incrementTabLevel(1, parent=ksp)

        self.Mksp = PETSc.KSP().create()
        self.Mksp.setOperators(kron.M)
        self.Mksp.setOptionsPrefix(prefix + self.prefix + "M_")
        self.Mksp.setFromOptions()
        self.Mksp.incrementTabLevel(1, parent=ksp)

        self.adaptive_tol = PETSc.Options().getBool(
            f"{prefix}ksp_{self.prefix}adaptive_tol", False)

        self.aksp_max_rtol = PETSc.Options().getReal(
            f"{prefix}ksp_{self.prefix}aksp_max_rtol", 0.1)

        self.xnest = kron.vec_nest.duplicate()
        self.bnest = kron.vec_nest.duplicate()

    def solve(self, ksp, b, x):
        bnest, xnest = self.bnest, self.xnest

        kronmat = ksp.mat_op.getPythonContext()
        Amat, Mmat, Smat = kronmat.A, kronmat.M, kronmat.S
        Aksp, Mksp = self.Aksp, self.Mksp

        if self.adaptive_tol:
            Aksp_rtol0 = Aksp.rtol

        # extract dense numpy array for S
        Smat.convert(PETSc.Mat.Type.DENSE)
        S = Smat.getDenseArray(readonly=True).copy().T
        Smat.convert(PETSc.Mat.Type.AIJ)

        wnest = bnest.duplicate()
        b.copy(result=wnest)

        # place M^{-1}B into bnest
        ws = wnest.getNestSubVecs()
        bs = bnest.getNestSubVecs()
        for wi, bi in zip(ws, bs):
            Mksp.solve(wi, bi)
        
        nb = len(bs)

        n = Amat.size[0]
        p = Smat.size[0]
        m_krylov = ksp.max_it + 1

        # Set Extended Krylov space quantities
        H = np.zeros((nb*(2*m_krylov+1), nb*(2*m_krylov))) # to hold projected problem
        L = np.zeros((nb*(2*m_krylov+1), nb*(2*m_krylov))) # to hold projected problem
        c = np.zeros((nb*(2*m_krylov+1), nb*(1         ))) # projected rhs
        y = np.zeros((nb*(2*m_krylov  ), nb*(p         ))) # projected solution

        # initialise ksp convergence
        ksp.its = 0
        ksp.setConvergedReason(PETSc.KSP.ConvergedReason.ITERATING)

        V = OrthonormalBasis()
        self._V = V
        w = Amat.createVecRight()

        def Amult(v, y):
            Amat.mult(v, w)
            Mksp.solve(w, y)

        def Asolve(v, y):
            Mmat.mult(v, w)
            Aksp.solve(w, y)

        # First basis vectors (orthogonalised rhs)
        extend_basis(lambda v, w: v.copy(w), V, bs, 0, c)

        # New basis vectors (obtained by mult by A^-1)
        extend_basis(Asolve, V, V[-nb:], 0, L, H)

        for i in range(m_krylov):
            # New basis vectors (obtained by mult by A then A^-1)
            extend_basis(Amult, V, V[-nb:], 2*i+1, H, L)
            extend_basis(Asolve, V, V[-nb:], 2*i+2, L, H)

            # Solve projected problem
            nvec = (2*i+2)*nb
            K = np.linalg.solve(L[:nvec, :nvec].T, H[:nvec+nb, :nvec].T).T

            y = solve_sylvester(
                K[:nvec,:nvec], S,
                c[:nvec,:nb])
            self._y = y

            # Check residual norm
            r = K[nvec:nvec+nb, :nvec] @ y
            rnorms = np.zeros((p,1))
            eps = np.finfo(c.dtype).eps
            for k in range(p):
                rnorms[k] = norm(r[:,k]) / (norm(c[:nb,:nb][:,k])*(1 + 2*eps))
            rnorm = norm(rnorms)

            # Monitor convergence
            ksp.monitor(i, rnorm)
            self._convergence_test(ksp, i, rnorm)
            if ksp.getConvergedReason() != 0:
                break

            if self.adaptive_tol:
                maxNormR = np.max(rnorms)
                Aksp.setTolerances(
                    rtol=max(Aksp_rtol0, min(Aksp_rtol0/maxNormR, self.aksp_max_rtol)))

        if self.adaptive_tol:
            Aksp.setTolerances(rtol=Aksp_rtol0)

        self.buildSolution(ksp, x)

    def buildSolution(self, ksp, x):
        xs = self.xnest.getNestSubVecs()
        for i, xi in enumerate(xs):
            dmult(self._V, self._y[:,i], result=xi)
        self.xnest.setNestSubVecs(xs)
        PETSc.Vec.concatenate(xs)[0].copy(result=x)

    def view(self, ksp, viewer=None):
        if viewer is None:
            return
        if viewer.type != PETSc.Viewer.Type.ASCII:
            return
        viewer.printfASCII(
            "Extended Krylov method for solving a Sylvester equation AX+SX=B.\n")
        viewer.printfASCII(
            f"maximum iterations={ksp.max_it}, initial guess is zero\n")
        viewer.printfASCII(
            f"tolerances: relative={ksp.rtol}, absolute={ksp.atol}, divergence={ksp.divtol}.\n")
        if self.adaptive_tol:
            viewer.printfASCII(
                "Using an adaptive tolerance for the KSP for A.\n")
        viewer.printfASCII(
            "The KSP for solving A is:\n")
        viewer.pushASCIITab()
        self.Aksp.view(viewer)
        viewer.popASCIITab()
        viewer.printfASCII(
            "The KSP for solving M is:\n")
        viewer.pushASCIITab()
        self.Mksp.view(viewer)
        viewer.popASCIITab()

    def _convergence_test(self, ksp, it, rnorm):
        if it == 0:
            self._rnorm0 = rnorm

        ksp.norm = rnorm
        ksp.its += 1

        ksp.logConvergenceHistory(rnorm)

        if it >= ksp.max_it:
            ksp.setConvergedReason(
                PETSc.KSP.ConvergedReason.CONVERGED_ITS)
        elif rnorm < ksp.atol:
            ksp.setConvergedReason(
                PETSc.KSP.ConvergedReason.CONVERGED_ATOL)
        elif rnorm/self._rnorm0 < ksp.rtol:
            ksp.setConvergedReason(
                PETSc.KSP.ConvergedReason.CONVERGED_RTOL)
        elif rnorm/self._rnorm0 > ksp.divtol:
            ksp.setConvergedReason(
                PETSc.KSP.ConvergedReason.DIVERGED_DTOL)


class IRKKroneckerPC(petsctools.PCBase):
    """
    PETSc Options
    -------------

    -pc_irkkron_shift_type (none|diag|eigmin) : default eigmin
    -pc_irkkron_shift_amount float : only for shift_type diag
    -irk_kron_ ... : options for kronecker ksp
    """
    prefix = "irkkron_"

    def initialize(self, pc):
        # from firedrake import derivative, replace
        from firedrake import derivative, replace, TrialFunction
        from firedrake.assemble import get_assembler
        from firedrake.dmhooks import get_appctx as get_snesctx
        from irksome import Dt
        from irksome.ufl.manipulation import split_time_derivative_terms

        ctx = get_snesctx(pc.getDM())
        stepper = ctx.appctx["stepper"]

        outer_prefix = pc.getOptionsPrefix() or ''
        prefix = f"{outer_prefix}{self.prefix}"
        pc_prefix = f"{outer_prefix}pc_{self.prefix}"

        V = stepper.u0.function_space()

        stage_bcs = stepper.orig_bcs
        butcher = stepper.butcher_tableau

        split_form = split_time_derivative_terms(
            stepper.F, stepper.t, timedep_coeffs=(stepper.u0,)
        )

        Mform = derivative(
            replace(split_form.time, {Dt(stepper.u0): stepper.u0}), stepper.u0)
        M_assembler = get_assembler(Mform, bcs=stage_bcs, mat_type="aij",
            options_prefix=f"{pc_prefix}M_")
        self.M = M_assembler.allocate()
        self._assemble_M = M_assembler.assemble
        self._assemble_M(tensor=self.M)
        # self.M.convert(PETSc.Mat.Type.DENSE)
        # assert np.allclose(np.eye(V.dim()), M.getDenseArray())
        # M.convert(PETSc.Mat.Type.AIJ)

        Kform = derivative(split_form.remainder, stepper.u0)

        dt_f = float(stepper.dt)
        A1, A2 = stepper.splitting(butcher.A)
        A = dt_f*butcher.A
        Ainv_array = np.linalg.inv(A)

        shift_options = PETSc.Options(pc_prefix + "shift_")

        shift_type = shift_options.getString("type", "eigmin")
        valid_shift_types = ("none", "diag", "eigmin")
        if shift_type not in valid_shift_types:
            raise ValueError(
                f"{pc_prefix}shift_type must be one of"
                f" {valid_shift_types}, not {shift_type}."
            )

        if shift_type == "diag":
            shift = shift_options.getScalar("amount", 1.0)
        elif shift_type == "eigmin":
            shift = np.min(np.linalg.eigvals(Ainv_array).real)
        else:
            shift = 0
        self.shift_type = shift_type
        self.shift = shift

        print(f"{shift= }")

        if shift != 0:
            Ainv_array -= shift*np.eye(butcher.num_stages)
            Kform += shift*Mform

        # K_assembler = get_assembler(
        #     Kform, bcs=stage_bcs, form_compiler_parameters=ctx.fcp,
        #     mat_type=ctx.mat_type, sub_mat_type=ctx.sub_mat_type,
        #     options_prefix=prefix, appctx=ctx.appctx
        # )
        K_assembler = get_assembler(
            Kform, bcs=stage_bcs, mat_type="aij",
            options_prefix=f"{pc_prefix}K_")
        self.K = K_assembler.allocate()
        self._assemble_K = K_assembler.assemble
        self._assemble_K(tensor=self.K)
        # K is defined over a single stage so needs
        # the DM for a single stage.
        # self.K.petscmat.setDM(stepper.u0.function_space().dm)

        Ainv = PETSc.Mat().createDense(
            size=(A.shape, A.shape),
            array=Ainv_array,
        )
        Ainv.convert(PETSc.Mat.Type.AIJ)

        full_size = pc.getOperators()[0].sizes

        kronmat = PETSc.Mat().createPython(
            size=full_size,
            context=KroneckerProductMat(
                self.K.petscmat, Ainv, self.M.petscmat),
            comm=pc.comm,
        )

        # IRK builds IxM + AxK but for sylvester we need
        # (A^-1)xM + IxK, so premultiply with (A^-1)xI
        A1inv = PETSc.Mat().createDense(
            size=(A.shape, A.shape),
            array=np.linalg.inv(dt_f*A1),
        )
        A1inv.convert(PETSc.Mat.Type.AIJ)
        zero_mat = PETSc.Mat().createConstantDiagonal(
            size=self.K.petscmat.sizes,
            diag=0., comm=pc.comm)
        self.kron_a1inv = PETSc.Mat().createPython(
            size=full_size,
            context=KroneckerProductMat(zero_mat, A1inv),
            comm=pc.comm,
        )

        kronksp = PETSc.KSP().create(comm=pc.comm)
        # This KSP is over the monolithic space so we
        # need the monolithic DM, but we are providing
        # the operators manually so deactivate the dm.
        # kronksp.setDM(pc.getDM())
        # kronksp.setDMActive(PETSc.KSP.DMActive.ALL, False)
        # Now we can finish setting up the KSP
        kronksp.setOperators(kronmat)
        kronksp.setOptionsPrefix(prefix)
        # default to looking like a pc
        kronksp.setType(PETSc.KSP.Type.PREONLY)
        kronksp.setFromOptions()

        kronksp.incrementTabLevel(1, parent=pc)
        kronksp.pc.incrementTabLevel(1, parent=pc)
        self.kronksp = kronksp

    def update(self, pc):
        self._assemble_M(tensor=self.M)
        self._assemble_K(tensor=self.K)

    def apply(self, pc, x, y):
        w = y.duplicate()
        self.kron_a1inv.mult(x, w)
        self.kronksp.solve(w, y)

    def view(self, pc, viewer=None):
        if viewer is None:
            return
        if viewer.type != PETSc.Viewer.Type.ASCII:
            return
        viewer.printfASCII(
            "Preconditioner to solve the Kronecker product Jacobian (IxM + AxK).\n")
        viewer.printfASCII(
            f"Shift type: {self.shift_type}, shift amount: {self.shift}.\n")
        viewer.printfASCII(
            "The KSP for solving the Kronecker product Jacobian is:\n")
        viewer.pushASCIITab()
        self.kronksp.view(viewer)
        viewer.popASCIITab()
