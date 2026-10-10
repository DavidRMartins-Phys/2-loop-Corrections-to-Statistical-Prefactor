import numpy as np
import scipy.special as spec
from scipy.interpolate import CubicSpline
from tqdm.notebook import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import warnings
from Shooting import shooting, clean_profile
from Riccati_Solutions import Riccati_Solution, Riccati_Solver


def meta_analytical_matrix(r, d, ell):
    "Determine G_meta(r,r') as a matrix of values."
    alpha = 0.5 * np.sqrt((d-2)**2 + 4*ell*(ell+d-2))
    rr, rp = r[:, None], r[None,:]
    lo, hi = np.minimum(rr, rp), np.maximum(rr, rp)
    with np.errstate(all = "ignore"):
        pref = (rr * rp)**(1-d/2)
        G = pref * spec.ive(alpha, lo) * spec.kve(alpha, hi) * np.exp(lo-hi)
        #exponential at the end converts the exponential to regular bessels
        #ive = e^(-lo)*iv, kve = e^(hi)*kv
    return G


def RG_low_ell(r_eval, phi, lambdap, d, ell, bounce=True):
    """The low ell propagators, both in the bounce and metastable backgrounds, are determined vie the matrix inverse of the operator,\
    rather than via the differential equation solutions. This is due to the different eigenvalues associated. For the ell=1,\
    the pseudoinverse is used, in order to remove the zero modes of the operator. As inputs, takes the ell values and whether it is the \
    bounce or metastable background."""
    N=len(r_eval)
    #lattice spacing
    h = r_eval[1] - r_eval[0]
    #Operator initialisation
    O = np.zeros((N,N))
    #Definition of V''(phi) based on the background
    if bounce:
        Vpp = 1 - phi(r_eval) + lambdap * 0.5 * phi(r_eval)**2
    else:
        Vpp = np.ones(N)
    if ell==1:
        for i in range(1, N-1):
            r = r_eval[i]
            #Second derivative
            O[i, i-1] = -1/h**2
            O[i,i] = 2/h**2
            O[i, i+1] = -1/h**2
            #First derivative
            O[i, i+1]+= (1-d)/r * 0.5/h
            O[i, i-1]+= (d-1)/r * 0.5/h
            #Constants
            O[i,i]+= (d-1)/r**2 + Vpp[i]
        
        # Explicitly set boundary rows so they aren't identically zero:
        O[0, 0] = 1.0 / h**2
        O[-1, -1] = 1.0 / h**2
        #Symmetrise the Operator with respect to the integration weight
        w = r_eval**(d-1)
        sqrt = np.sqrt(w)
        O_sym = (sqrt[:,None] * O)/sqrt[None,:]
        O_sym = 0.5 * (O_sym + O_sym.T)

        # Determine the pseudoinverse of the matrix, i.e. the inverse removing the zero eigvals
        O_sym_inv = np.linalg.pinv(O_sym, 1e-7)

        #undo symmetrisation via weights
        G_bar_mat = O_sym_inv/(h*sqrt[:, None]*sqrt[None, :])
        return G_bar_mat
    else:
        for i in range(1, N):
            r = r_eval[i]
            O[i, i-1] = -1/h**2 + (d-1)/r * 0.5/h
            O[i,i] = 2/h**2 + Vpp[i]
            if i<N-1:
                O[i, i+1] = -1/h**2 + (1-d)/r * 0.5/h

        # Explicitly set boundary rows so they aren't identically zero:
        O[0, 0] =  2*d/h**2 + Vpp[0]
        O[0, 1] = -2*d/h**2
        O[-1, -1] = 1.0 / h**2
        #integration weight
        w = r_eval**(d-1)
        #Determine the inverse of the matrix
        Oinv = np.linalg.inv(O)

        #Normalise with the delta function
        G=Oinv/(h*w[None, :])
        return G

def compute_stack_entry(l, r_eval, phi_vals, r_min, r_max, d, phi_0, lambdap):
    phi = CubicSpline(r_eval, phi_vals)
    N = len(r_eval)
    G_phi = np.zeros((N, N))
    #determines the difference for low l values
    if l in (0,1):
        return l, RG_low_ell(r_eval, phi, lambdap, d, l) - RG_low_ell(r_eval, phi, lambdap, d, l, False)
    #determines the metastable propagator from riccati integration
    elif l>30:
        G_meta = np.zeros((N,N))
        IS_met, OS_met = Riccati_Solver(r_min, r_max, d, l, phi_0, lambdap, phi, False)
        for i in range(N):
            rp = r_eval[i]
            G_meta[:,i] = Riccati_Solution(r_eval, rp, d, l, IS_met, OS_met, phi_0,lambdap, False)
    else:
        G_meta = meta_analytical_matrix(r_eval, d, l)
    
    IS, OS = Riccati_Solver(r_min, r_max, d, l, phi_0, lambdap, phi, True)
    for i in range(N):
        rp = r_eval[i]
        G_phi[:,i] =Riccati_Solution(r_eval, rp, d, l, IS, OS, phi_0,lambdap, True)
    return l, G_phi - G_meta
    
def build_propagator_difference_stack(d, lambdap, r_eval, L_max, r_min, r_max, max_workers = None):
    #1) Shooting solution
    R, solution, phi_0 = shooting(d, lambdap)
    phi = clean_profile(r_eval,CubicSpline(R, solution))
    #2) Initialisation of the stack
    N = len(r_eval)
    Stack = np.empty((L_max, N, N))
    with ThreadPoolExecutor(max_workers = max_workers) as executor:
        futures = [executor.submit(compute_stack_entry,l, r_eval, phi, r_min, r_max, d, phi_0, lambdap) for l in range(L_max)]
        for future in tqdm(as_completed(futures), total=L_max, desc="Computing Stack"):
            l_res, res = future.result()
            Stack[l_res] = res
    return phi, Stack