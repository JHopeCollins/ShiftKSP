from numpy import ndarray

__all__ = ["OrthonormalBasis"]


def mdot(x, y):
    n = len(x)
    r = ndarray(n)
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
    def __init__(self, A, S):
        self.A = A
        self.S = S

        Ia = PETSc.Mat().createConstantDiagonal(
            size=A.sizes, diag=1, comm = A.comm)

        Is = PETSc.Mat().createConstantDiagonal(
            size=S.sizes, diag=1, comm = S.comm)

        IA = Ia.kron(A)
        SI = S.kron(Is)

        self.kronecker_mat = IA + SI

    def mult(self, pc, x, y):
        self.kronecker_mat.mult(x, y)
