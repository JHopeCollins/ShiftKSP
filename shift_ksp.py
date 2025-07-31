
def orthogonalise(x, y):
        xty = y.dot(x)
        y -= x*xty
        ynorm = y.norm()
        y /= ynorm
        return y, ynorm, xty
    

class ShiftedComplexKSP:
    prefix = "shift_"
    def setFromOptions(self, ksp):
        pass

    def setUp(self, ksp, b, x):
        Akron, Pkron = ksp.getOperators()

        A = Akron.getMat(0, 1)
        S = Akron.getMat(1, 0)

        self.n, _ = A.getSizes()
        self.p, _ = S.getSizes()

        self.s = S.getDiagonal().array_r

        Ap = Pkron.getMat(0, 1)

        prefix = ksp.getOptionsPrefix()
        self.Aksp = PETSc.KSP().create()
        self.Aksp.setOperators(A, Ap)
        self.Aksp.setPrefix(prefix+self.prefix)
        self.Aksp.setFromOptions()
        self.Aksp.setUp()

        self.xs = tuple(
            A.createVecRight()
            for _ in range(self.p))

        m, n = self.max_it, self.n
        self.V = np.array((n, 2*m))
        self.H = np.array((2*m, 2*m-2))

        self.b = A.createVecLeft()
        self.Ab = A.createVecRight()
        self.Ainvb = A.createVecRight()

    def solve(self, ksp, ball, xall):
        n, p = self.n, self.p
        b = self.b
        V = self.V
        Ainvb = self.Ainvb
        Ab = self.Ab

        # 1a) split b into bi
        b.array[:] = ball.array[:n]
        bnorm = b.norm()
        b /= bnorm
        V[:,0] = b.array_r[:]

        Aksp.solve(b, Ainvb)

        _, A1bnorm, bA1b = orthogonalise(b, Ainvb)

        V[:,1] = Ainvb.array_r[:]

        norms = [A1bnorm]
        dots = [bA1b]

        # 2) loop
        for k in range(self.max_it):
            Voff = 2*k+1

            # a) A^k b
            b.array[:] = V[:,Voff]
            self.A.mult(b, Ab)

            for j in range(Voff - 1):
                b.array[:] = V[:, j]
                _, norm, dot = orthogonalise(b, Ab)
                norms.append(norm)
                dots.append(dot)

            self.V[:,Voff+1] = Ab.array_r[:]

            # b) A^-k b
            Voff += 1
            b.array[:] = V[:,Voff]
            self.Aksp.solve(b, Ainvb)

            for j in range(Voff - 1):
                b.array[:] = V[:, j]
                _, norm, dot = orthogonalise(b, Ainvb)
                norms.append(norm)
                dots.append(dot)

            self.V[:,Voff+1] = Ainvb.array_r[:]

        #   c) H calculation over self.V[:2*k]
        #   d) residual estimation
        #   e) monitors & convergence test

        # 3) reassemble x from xi
        for i in range(p):
            x.array[i*n:(i+1)*n] = self.xs[i].array[:]


class IRKDiagonalisationPC(firedrake.PCBase):
    prefix = "diag_"

    def setUp(self, pc):
        stepper = self.get_appctx(pc).appctx['stepper']

        # Define IA + SM
        # A from single timestep form F (with averaging over stages)
        # S from Butcher tableau eigenvalues

        self.ksp = PETSc.KSP()
        self.ksp.setOperators(IA+SM)
        self.ksp.setPrefix(pc.getOptionsPrefix()+self.prefix)

    def apply(pc, x, y):
        # diagonalise
        ...

        self.ksp.solve(xd, yd)

        # undiagonalise 

params = {
    'pc_type': 'python',
    'pc_python_type': 'irksome.IRKDiagonalisationPC',
    'diag_ksp_type': 'python',
    'diag_ksp_python_type': 'ShiftedComplexKSP',
    'diag_ksp_shift_Aksp_ksp_type': 'gmres',
    'diag_ksp_shift_Aksp_pc_type': 'hppdm',
    'diag_ksp_shift_Mksp_ksp_type': 'chebyshev',
    'diag_ksp_shift_Mksp_pc_type': 'ilu',
}
