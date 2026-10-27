import numpy as np
import scipy.special as spec
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from Shooting import shooting

def Riccati_eq(r, y, d, lambdap, ell, phi, Bounce):
    """
    Differential equation in the Riccati form, with the integration variable as G'/G.
    
    Parameters
    ----------
    r : ndarray
        radial coordinate spanning r_min to r_max
    y : differential equation variable
       
    d : float
        Number of dimensions
    lambdap : float
        dimensionless coupling value
    ell : float/integer
        Orbital angular momentum number, quantifying the hyper-spherical harmonics
    phi : spline object
        bounce profile
    Bounce : Boolean
        If True, the effective mass takes into accound the non-constant bounce profile, allowing the solution to the 
        bounce propagator. If False, the equation for the metastable propagator is obtained.
    """
    
    phi_at_r = phi(r)
    if Bounce:
        msq_eff = 1 - phi_at_r + lambdap * 0.5 * phi_at_r**2 + ell*(ell+d-2)/r**2
    else:
        msq_eff = 1+ ell*(ell+d-2)/r**2
    R = y[0]
    Rp = -(R**2 + (d-1)/r * R) + msq_eff
    return [Rp]

def Riccati_left_BC(r_min, d, ell, phi0, lambdap, Bounce):
    """
    Defines the boundary condition at the r_min value given the analytical solution in the limit.

    Parameters
    ----------
    r_min : float
        Lower integration bound.
    d : float
        Number of dimensions
    ell : float/integer
        Orbital angular momentum number, quantifying the hyper-spherical harmonics
    phi0 : float
        Bounce solution in the r->0 limit.
    lambdap : float
        dimensionless coupling
    Bounce: Boolean
        If True, the effective mass takes into accound the non-constant bounce profile, allowing the solution to the 
        bounce propagator. If False, the equation for the metastable propagator is obtained.

    Returns
    -------
    ndarray
        Boundary conditions at r_min for ell value.

    """
    alpha = 0.5 * np.sqrt((d-2)**2 + 4*ell*(ell + d - 2))
    if Bounce:
        b2 = (1 - phi0 + lambdap/2 * phi0**2)
    else:
        b2=1
    return np.array([(1 - 0.5*d + alpha)/r_min + b2*r_min/(2*(alpha + 1))])


def Riccati_right_BC(r_max, d, ell):
    """
    Defines the boundary condition at the r_max value given the analytical
    (modified Bessel function) solution in the large-r limit, where the
    effective mass approaches that of the metastable vacuum (b = 1).
    
    The Riccati variable is the logarithmic derivative of the radial function,
    y = d ln(psi)/dr, with psi(r) = r^(1 - d/2) K_alpha(r) decaying as r -> infinity.
    The boundary value is therefore y = (1 - d/2)/r + K_alpha'(r)/K_alpha(r).
    
    Parameters
    ----------
    r_max : float
        Upper integration bound.
    d : float
        Number of dimensions.
    ell : float/integer
        Orbital angular momentum number, quantifying the hyper-spherical harmonics.
    
    Returns
    -------
    ndarray
        Boundary condition at r_max for the given ell value.
    
    Notes
    -----
    For alpha < 10, the logarithmic derivative of K_alpha is evaluated exactly
    using the exponentially scaled Bessel function kve(alpha, r) = e^r K_alpha(r),
    which avoids underflow at large r. The derivative follows from the recurrence
    K_alpha' = -(K_{alpha-1} + K_{alpha+1})/2, and the e^r scaling is removed
    by subtracting 1 from the scaled logarithmic derivative.
    
    For alpha >= 10, kve becomes unreliable (overflow in the order), so the
    uniform large-order asymptotic (Debye) expansion of K_alpha is used instead,
    giving d ln K/dr = -s/r - r/(2 s^2) with s = sqrt(alpha^2 + r^2).
    """
    alpha = 0.5*np.sqrt((d-2)**2 + 4*ell*(ell+d-2))
    if alpha < 10:
        K  = spec.kve(alpha, r_max)
        Kp = -0.5*(spec.kve(alpha-1, r_max) + spec.kve(alpha+1, r_max)) + K
        dlnK = Kp/K - 1        # kve = e^r K, so K'/K = kve'/kve - 1
    else:
        s = np.sqrt(alpha**2 + r_max**2)
        dlnK = -s/r_max - r_max/(2*s**2)
    return np.array([(1 - 0.5*d)/r_max + dlnK])


#Solver yielding the solutions for R(r)
def Riccati_Solver(r_min, r_max, d, ell, phi0, lambdap, phi, Bounce, r_eval):
    """
    Solves the Riccati equation for R(r) = G'(r)/G(r), the logarithmic
    derivative of the radial part G(r) of the propagator for partial wave ell,
    by integrating from both ends of the radial domain with the analytical
    boundary conditions in the r -> 0 and r -> infinity limits.
    
    The solution regular at the origin is integrated outwards from r_min to
    r_max, and the solution decaying at infinity is integrated inwards from
    r_max to r_min. The two are then available to be matched, e.g. at an
    intermediate point, to build the propagator for the given ell.
    
    Parameters
    ----------
    r_min : float
        Lower integration bound.
    r_max : float
        Upper integration bound.
    d : float
        Number of dimensions.
    ell : float/integer
        Orbital angular momentum number, quantifying the hyper-spherical harmonics.
    phi0 : float
        Bounce solution in the r -> 0 limit, used for the left boundary condition.
    lambdap : float
        Dimensionless coupling.
    phi : callable
        Bounce profile phi(r), passed through to Riccati_eq to build the
        effective mass term.
    Bounce : bool
        If True, the effective mass takes into account the non-constant bounce
        profile, giving the bounce propagator. If False, the equation for the
        metastable (constant-mass) propagator is solved.
    
    Returns
    -------
    sol1 : OdeResult
        Outward solution R(r) = G'/G (r_min -> r_max) from
        scipy.integrate.solve_ivp, evaluated at r_eval, with dense output.
    sol2 : OdeResult
        Inward solution R(r) = G'/G (r_max -> r_min), evaluated at
        np.flip(r_eval), with dense output.
    
    Notes
    -----
    Both integrations use the implicit Radau method (atol = 1e-10,
    rtol = 1e-8), which is well suited to the stiff behaviour of the Riccati
    equation at large ell. Integrating each solution in its stable direction
    (outwards for the regular one, inwards for the decaying one) avoids
    contamination from the exponentially growing mode.
    
    The grid r_eval is read from the global scope rather than passed as an
    argument. sol2.t is ordered from r_max down to r_min, so it must be
    flipped before being compared point by point with sol1.t.
    """
    ICI = Riccati_left_BC(r_min, d, ell, phi0, lambdap, Bounce)
    ICO = Riccati_right_BC(r_max, d, ell)
    sol1 = solve_ivp(Riccati_eq, [r_min, r_max], ICI, t_eval = r_eval, method = 'Radau', dense_output = True, \
                     atol = 1e-10, rtol = 1e-8, args = (d, lambdap,ell,  phi, Bounce))
    sol2 = solve_ivp(Riccati_eq, [r_max, r_min], ICO, t_eval = np.flip(r_eval), method = 'Radau', dense_output = True, \
                     atol = 1e-10, rtol = 1e-8, args = (d, lambdap,ell, phi, Bounce))
    return sol1, sol2


def Riccati_Splicer(sol_left, sol_right, r, r_prime, d, r_eval):
    """
    Builds the radial propagator G(r, r') for a single partial wave ell by
    splicing the outward and inward Riccati solutions at the source point r'.
    
    The Riccati solutions R_L = G_L'/G_L (regular at the origin) and
    R_R = G_R'/G_R (decaying at infinity) are integrated to recover the
    ratios G_L(r)/G_L(r') and G_R(r)/G_R(r'). These are joined at r = r'
    and normalised by the jump condition of the delta-function source, so the
    full solution is continuous at r' and its derivative jumps there.
    
    Parameters
    ----------
    sol_left : OdeResult
        Outward solution R_L(r) from Riccati_Solver (r_min -> r_max), with dense output.
    sol_right : OdeResult
        Inward solution R_R(r) from Riccati_Solver (r_max -> r_min), with dense output.
    r : ndarray
        Radial points at which the propagator is returned.
    r_prime : float
        Source point r' of the propagator. Must lie within the range of r_eval.
    d : float
        Number of dimensions.
    r_eval : ndarray
        Grid on which the Riccati solutions are sampled and splined. Should
        match the grid used in Riccati_Solver.
    
    Returns
    -------
    G : ndarray
        Radial propagator G(r, r') for the given ell, evaluated at r. It equals
        prefactor * G_L(r)/G_L(r') for r <= r' and prefactor * G_R(r)/G_R(r')
        for r > r', where prefactor = 1/(r'^(d-1) W).
    """
    #Extraction of R from solution
    R_left = sol_left.sol(r_eval).real[0]
    R_right = sol_right.sol(r_eval).real[0]
    
    R_L = CubicSpline(r_eval, R_left)
    R_R = CubicSpline(r_eval, R_right)
    
    W = R_L(r_prime) - R_R(r_prime)
    prefactor = 1/(r_prime**(d-1) * W)
    
    #Antiderivatives of R
    R_L_int = R_L.antiderivative()
    R_R_int = R_R.antiderivative()
    
    #masks working as a heavy sidestep function
    mask_l = r<=r_prime
    mask_r = ~ mask_l
    
    #Determining the log of the ratios between g_l/r and g_l/r(r')
    ln_ratio_l = R_L_int(r[mask_l]) - R_L_int(r_prime)
    ln_ratio_r = R_R_int(r[mask_r]) - R_R_int(r_prime)
    
    #Setup of solution array
    G = np.zeros_like(r)
    
    #final solution
    if r[mask_l].size > 0:
      G[mask_l] = prefactor * np.exp(ln_ratio_l)
    if r[mask_r].size > 0:
      G[mask_r] = prefactor * np.exp(ln_ratio_r)
    
    return G

def Greens_Test(IS, OS, r_eval, d, ell, lambdap, phi, Bounce,trim=3):
    R_L = CubicSpline(r_eval, IS.sol(r_eval).real[0])
    R_R = CubicSpline(r_eval, OS.sol(r_eval).real[0])

    if Bounce:
        p=phi(r_eval)
    else:
        p=0
    m2 = 1-p+0.5*lambdap*p**2 + ell * (ell+d-2)/r_eval**2
    out = {}
    for name, R_spline in [("L", R_L), ("R", R_R)]:
        R = R_spline(r_eval)
        R_p = R_spline(r_eval, 1)

        res = R_p + R**2 + (d-1)/r_eval * R - m2

        scale = abs(R_p) + R**2 + abs((d-1)*R/r_eval)+abs(m2)

        rel_error = np.abs(res/scale)

        error = rel_error[trim:-trim]
        out["Residual_" + name] = np.max(error)
    return out


#Defining the function which obtains the final solution using Riccati
#resolves the general r' solutions per specific r' and sets normalisation
def Riccati_Solution(r, rprime, d, ell, IS, OS, phi0, lambdap, Bounce,phi,r_eval,tol = 1e-5):
    """
    Returns the radial propagator G(r, r') for partial wave ell, but only if
    the underlying Riccati solutions pass a set of numerical consistency checks.

    The propagator is built by Riccati_Splicer from the outward and inward
    Riccati solutions. Before it is returned, Greens_Test evaluates how well
    each solution satisfies its ODE and how constant the Wronskian-like
    quantity is across the domain. If any metric exceeds the tolerance, a
    warning is printed and the propagator is withheld.

    Parameters
    ----------
    r : ndarray
        Radial points at which the propagator is evaluated.
    rprime : float
        Source point r' of the propagator.
    d : float
        Number of dimensions.
    ell : float/integer
        Orbital angular momentum number, quantifying the hyper-spherical harmonics.
    IS : OdeResult
        Outward (inner) Riccati solution from Riccati_Solver (r_min -> r_max).
    OS : OdeResult
        Inward (outer) Riccati solution from Riccati_Solver (r_max -> r_min).
    phi0 : float
        Bounce solution in the r -> 0 limit.
    lambdap : float
        Dimensionless coupling.
    Bounce : bool
        If True, the effective mass takes into account the non-constant bounce
        profile (bounce propagator). If False, the metastable (constant-mass)
        propagator is used.
    tol : float, optional
        Tolerance applied to each of the three checks: the ODE residuals of
        the left and right solutions, and the largest spread of the Wronskian.
        Default is 1e-3.

    Returns
    -------
    ndarray or str
        The propagator G(r, r') from Riccati_Splicer if all checks pass.
        Otherwise the string "Failed Test", after printing a warning.
        """
    
    RG = Riccati_Splicer(IS, OS, r, rprime, d, r_eval)
    metrics = Greens_Test(IS, OS, r, d, ell, lambdap, phi, Bounce)
    
    ode_L_ok = metrics["Residual_L"] < tol
    ode_R_ok = metrics["Residual_R"] < tol
    
    if ode_L_ok and ode_R_ok:
      return RG
    else:
       raise ValueError("Riccati Solution failed convergence checks: ")
