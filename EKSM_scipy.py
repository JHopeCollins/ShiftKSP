from math import sqrt, sin, cos, pi
import numpy as np
from scipy.sparse import eye, spdiags, kron
from scipy.sparse.linalg import splu, LinearOperator, gmres
from scipy.linalg import solve_sylvester, norm
import butchertableau as bt

np.random.seed(6)
   
# Initialize
m = 10 # grid size
n = m * m # number of unknowns
symmetric = True
re = 10
angle = pi/3
I = eye(m)
L = spdiags([[1]*m, [-1]*m], [0, 1], m, m)
D = spdiags([[-1]*m, [2]*m, [-1]*m], [-1,0,1], m, m)
A =  (1/re)*(m+1)**2 * (kron(D,I) + kron(I,D))
if not symmetric:
    A +=  (m+1) * (cos(angle)*kron(L,I) + sin(angle)*kron(I,L))
A += eye(n)

# Factor the matrix A
lu = splu(A.tocsc())
dA = LinearOperator(A.shape, matvec=lu.solve)

# Generate Butcher tableau matrix
order = 8
tableau = bt.butcher(order, 15)
S, _, _ = tableau.radau() 
Sinv = np.array(tableau.inv(S),np.float64)
p = Sinv.shape[0]
# print(f"{Sinv = }")

# Create rhs matrix
b = np.random.randn(n, 1)

# Set Extended Krylov space quantities
m_krylov = 160 # maximum iteration count
X = np.zeros((n,p)) # solution matrix
V = np.zeros((n, 2 * m_krylov + 2)) # matrix holding basis vectors
H = np.zeros((2 * m_krylov + 1, 2 * m_krylov)) # to hold projected problem
c = np.zeros((2 * m_krylov + 1, 1)) # projected rhs
y = np.zeros((2 * m_krylov, p)) # projected solution

# First basis vector (orthogonalised rhs)
beta = norm(b)
c[0,0] = beta
V[:, 0] = b.flatten()/beta

# New vector (multiply previous one by A then orthogonalise)
w = dA @ V[:,0]
hp = V[:, 0].T @ w
w = w - V[:, 0] * hp
h1p = V[:, 0].T @ w
w = w - V[:, 0] * h1p
hp += h1p
normwp = np.linalg.norm(w)
hp = np.append(hp, normwp)
V[:, 1] = w / normwp

# index for last vector
ind = 0

for i in range(m_krylov):
    ind += 1

    # New basis vector (obtained by mult by A)
    w = A @ V[:, ind]
    h = V[:, :ind+1].T @ w
    w = w - V[:, :ind+1] @ h
    h1 = V[:, :ind+1].T @ w
    w = w - V[:, :ind+1] @ h1
    h += h1
    normw = np.linalg.norm(w)
    V[:, ind + 1] = w / normw

    # Put projection data in H
    H[:ind+1, ind] = h
    H[ind+1, ind] = normw

    # Put projection data for previous set, see note below
    H[:ind, ind - 1] = -H[:ind, :ind - 1] @ hp[:ind - 1]
    H[ind - 1, ind - 1] += 1
    H[:ind + 2, ind - 1] -= np.append(h, normw) * normwp
    H[:ind + 2, ind - 1] /= hp[ind - 1]

    # Move on to add next set of basis vectors (obtained by mult by A^-1)
    ind += 1
    w = dA @ V[:, ind]
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
    hp = V[:, :ind+1].T @ w
    w = w - V[:, :ind+1] @ hp
    h1p = V[:, :ind+1].T @ w
    w = w - V[:, :ind+1] @ h1p
    hp += h1p
    normwp = np.linalg.norm(w)
    V[:, ind + 1] = w / normwp
    
    # Solve projected problem
    temp = 2*i+2
    y = solve_sylvester(H[:temp,:temp],Sinv,np.outer(c[:temp],np.ones(p)))
    
    # Check residual norm
    r = H[temp:temp+p,:temp] @ y
    rnorms = np.zeros((p,1))
    for k in range(p):
        rnorms[k] = norm(r[:,k])/beta
        
    # for k in range(p):
    #     if use_least_squares:
    #         VtAV = H[0:temp+1,0:temp].copy()
    #         VtAV[0:temp,0:temp] += s[k]*np.eye(temp)
    #         y[0:temp,k] = np.linalg.lstsq(VtAV,c[0:temp+1,0])[0]
    #     else:
    #         VtAV = H[0:temp,0:temp].copy()
    #         VtAV[0:temp,0:temp] += s[k]*np.eye(temp)
    #         y[0:temp,k] = np.linalg.solve(VtAV,c[0:temp,0])
    #     X[:,k] = V[:,0:temp]@y[0:temp,k]
    #     norms[k] = np.linalg.norm(A@X[:,k]+s[k]*X[:,k]-b.T)/beta
    print(f"max r_{i}: {max(rnorms)[0]:.6e}")
    if(np.max(rnorms) < 1e-8):
        break

# Solution recovery    
X = V[:,:temp]@y

# Check true residual norms
norms = np.zeros((p,1))
R = A@X+X@Sinv
for k in range(p):
    norms[k] = np.linalg.norm(b.T-R[:,k])/beta
print(f"Maximum of true residual norms: {max(np.abs(norms))[0]:.6e}")

