
def orthogonalise(x, y):
        xty = y.dot(x)
        y -= x*xty
        return y, xty
    

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
        self.V = np.array((n, 2*m+2))
        self.H = np.array((2*m+1, 2*m))
        self.c = np.array((2*m+1, 1))
        self.y = np.array((2*m, self.p))
        self.rnorms = np.zeros((m,p))

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
        beta = b.norm()
        b /= beta
        V[:,0] = b.array_r[:]
        self.c[0,0] = beta
        
        Aksp.solve(b, Ainvb)

        _, hp = orthogonalise(b, Ainvb)
        normwp = Ainvb.norm()
        Ainvb /= normwp

        V[:,1] = Ainvb.array_r[:]

        ind = 0
        # 2) loop
        for k in range(self.max_it):
            ind += 1

            # a) A^k b
            b.array[:] = V[:,ind]
            self.A.mult(b, Ab)

            h = []
            for j in range(ind+1):
                b.array[:] = V[:, j]
                _, dot = orthogonalise(b, Ab)
                h.append(dot)
            normw = Ab.norm()
            Ab /= normw

            self.V[:,Voff+1] = Ab.array_r[:]
            
            self.H[:ind+1, ind] = np.array(h)
            self.H[ind+1, ind] = normw

            self.H[:ind, ind - 1] = -self.H[:ind, :ind - 1] @ hp[:ind - 1]
            self.H[ind - 1, ind - 1] += 1
            self.H[:ind + 2, ind - 1] -= np.append(h, normw) * normwp
            self.H[:ind + 2, ind - 1] /= hp[ind - 1]
            
            # b) A^-k b
            ind += 1
            b.array[:] = V[:,ind]
            self.Aksp.solve(b, Ainvb)

            hp = []
            for j in range(ind + 1):
                b.array[:] = V[:, j]
                _, dot = orthogonalise(b, Ainvb)
                hp.append(dot)
            normwp = Ainvb.norm()
            Ainvb /= normwp

            self.V[:,ind+1] = Ainvb.array_r[:]

            #   d) residual estimation
            temp = 2*k+2
            self.y[:temp,:] = scipy.linalg.solve_sylvester(self.H[:temp,:temp], 
                                self.S,
                                np.outer(self.c[:temp],np.ones(p)))
            r = self.H[temp:temp+self.p,:temp] @ self.y[:temp,:]
        
            for j in range(self.p):
                self.rnorms[k,j] = np.linalg.norm(r[:,j])
            
        #   e) monitors & convergence test

        # 3) reassemble x from xi
        
        for i in range(p):
            self.xs[i].array[:] = self.V[:,:temp]@ self.y[:temp,i]
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
