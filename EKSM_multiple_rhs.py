import numpy as np
from scipy.sparse import eye, spdiags, kron
from scipy.sparse.linalg import splu, LinearOperator
from scipy.linalg import solve_sylvester, norm, solve
import butchertableau as bt

   
# Initialize
m = 100 # grid size
n = m * m # number of unknowns
symmetric = False
I = eye(m)
L = spdiags([[1]*m, [-1]*m], [0, 1], m, m)
D = spdiags([[-1]*m, [2]*m, [-1]*m], [-1,0,1], m, m)
A =  (m+1)**2 * (kron(D,I) + kron(I,D))
if not symmetric:
    A +=  (m+1) * (kron(L,I))
        
# Factor the matrix A
lu = splu(A.tocsc())
dA = LinearOperator(A.shape, matvec=lu.solve)

# Generate Butcher tableau matrix
order = 20
tableau = bt.butcher(order, 15)
S, _, _ = tableau.radau() 
Sinv = np.array(tableau.inv(S),np.float64)
p = Sinv.shape[0]

# Create rhs matrix
b = np.random.randn(n, p)

# Set Extended Krylov space quantities
m_krylov = 160 # maximum iteration count
X = np.zeros((n,p)) # solution matrix
V = np.zeros((n, p * (2 * m_krylov + 2))) # matrix holding basis vectors
H = np.zeros((p * (2 * m_krylov + 1), p * 2 * m_krylov)) # to hold projected problem
c = np.zeros((p * (2 * m_krylov + 1), p)) # projected rhs
y = np.zeros((p * 2 * m_krylov, p)) # projected solution

# First basis vectors (orthogonalised rhs)
V[:,:p], beta = np.linalg.qr(b,'reduced')
c[:p,:p] = beta

# New set of vectors (multiply previous set by A then orthogonalise)
w = dA @ V[:,:p]
hp = V[:, :p].T @ w
w = w - V[:, :p] @ hp
h1p = V[:, :p].T @ w
w = w - V[:, :p] @ h1p
hp += h1p
V[:, p : 2*p], normwp = np.linalg.qr(w,'reduced')

# index set for last set
ind = np.array(range(p))

for i in range(m_krylov):
    ind += p
    max_ind = max(ind)+1

    # New set of basis vectors (obtained by mult by A)
    w = A @ V[:, ind]
    h = V[:, :max_ind].T @ w
    w = w - V[:, :max_ind] @ h
    h1 = V[:, :max_ind].T @ w
    w = w - V[:, :max_ind] @ h1
    h += h1
    normw = np.linalg.norm(w)
    V[:, ind + p], normw = np.linalg.qr(w,'reduced')

    # Put projection data in H
    H[:max_ind, ind] = h
    H[max_ind:max_ind+p, ind] = normw

    # Put projection data for previous set, see note below
    H[:max_ind, ind - p] = -H[:max_ind, :max_ind - p] @ hp[:max_ind - p]
    H[max_ind - 2*p:max_ind-p, max_ind - 2*p:max_ind-p] += np.eye(p)
    H[:max_ind, ind - p] -= h @ normwp
    H[max_ind : max_ind + p, ind - p] -= normw @ normwp
    H[:max_ind + p, ind - p] = solve(hp[ind - p].T,H[:max_ind + p, ind - p].T).T

    # Move on to add next set of basis vectors (obtained by mult by A^-1)
    ind += p
    max_ind += p
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
    hp = V[:, :max_ind].T @ w
    w = w - V[:, :max_ind] @ hp
    h1p = V[:, :max_ind].T @ w
    w = w - V[:, :max_ind] @ h1p
    hp += h1p
    V[:, ind + p], normwp = np.linalg.qr(w,'reduced')
    
    # Solve projected problem
    temp = (2*i+2)*p
    norms = np.zeros((p,1))
    y[:temp,:] = solve_sylvester(H[:temp,:temp],Sinv,c[:temp,:p])
    
    # Check residual norm
    r = H[temp:temp+p,:temp] @ y[:temp,:]
    rnorms = np.zeros((p,1))
    for k in range(p):
        rnorms[k] = norm(r[:,k])/norm(b[:,k])
         
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
    print("max r_",i, ":", max(rnorms))
    if(np.max(rnorms) < 1e-8):
        break
    
# Solution recovery    
X = V[:,:temp]@y[:temp,:]

# Check true residual norms
norms = np.zeros((p,1))
R = A@X+X@Sinv - b
for k in range(p):
    norms[k] = norm(R[:,k])/norm(b[:,k])
print("Maximum of true residual norms:", max(np.abs(norms)))
