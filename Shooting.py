from scipy.differentiate import derivative
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.integrate import solve_ivp, quad
import numpy as np

def Vp(phi, lambdap):
    """
    First derivative of the scalar potential, 
    V'(phi) = phi - phi^2/2 + lambdap * phi^3/6.

    Parameters
    ----------
    phi : float or ndarray
        Field value(s).
    lambdap : float
        Quartic coupling (dimensionless).

    Returns
    -------
    float or ndarray
        V'(phi), same shape as `phi`.
    """
    V = phi - 0.5 * phi**2 + lambdap/6 * phi**3
    return V

def Initialderiv(d, lambdap, r_min, phi_0_guess):
    """
    Due to the divergence of the first derivative term [(d-1)/r d/dr ]in the ODE for r->0, 
    use a small radius close to r=0 (r_min) and approximate the field as a taylor series.
    phi(r) ≈ phi_0 + B r^2,   so   phi'(r) ≈ 2 B r
    The bounce has the boundary condition
    [dphi/dr]_{r=0} = 0. This sets the value of B, based on the phi(0) ansatz.

    Parameters
    ----------
    d: float
        number of dimensions
    lambdap: float
        dimensionless coupling
    r_min: float
        small distance away from 0 where the numerics begin
    phi_0_guess: float
        ansatz for the initial field configuration value

    Returns:
    phip_0: float
        dphi/dr at r_min
    """
    B = 1/(2*d) * Vp(phi_0_guess, lambdap)
    phip_0 = 2* B * r_min
    return phip_0

def Initial_Phi(phi_0_guess, d, lambdap, r_min):
    """
    Determines the value of the bounce at r_min, given an Ansatz phi_0_guess at r=0.
    This is done via the Taylor expansion phi(r) = phi_0 + B r^2.
    B is determined by enforcing that the derivative of the bounce vanishes at r=0.

    Parameters
    ----------
    d: float
        number of dimensions
    lambdap: float
        dimensionless coupling
    r_min: float
        small distance away from 0 where the numerics begin
    phi_0_guess: float
        ansatz for the initial field configuration value

    Returns
    -------
    phi(r_min)
    """
    B = 1/(2*d) * Vp(phi_0_guess, lambdap)
    Value = phi_0_guess + B * r_min**2
    return Value

def fun(r, y,d, lambdap):
    """
    Right-hand side of the bounce equation as a first-order system.
    phi'' + (d-1)/r * phi' = V'(phi)  ->  y = (phi, chi) with chi = dphi/dr.
    
    Parameters
    ----------
    r : float
        Radial coordinate.
    y : sequence of float
        (phi, chi) at r.
    d : float
        Number of dimensions.
    lambdap : float
        Dimensionless coupling.

    Returns
    -------
    list of float
        [dphi/dr, dchi/dr].
    """
    phi, chi = y
    return [chi, -(d-1)/r * chi + Vp(phi, lambdap)]


def Solver(phi_0_guess, d, r_min, r_max, lambdap, r_eval):
    """
    Integrate the bounce equation outwards from r_min to r_max for a trial phi(0).

    Starts from the Taylor-expanded initial conditions at r_min and solves
    with DOP853 at tight tolerances.

    Parameters
    ----------
    phi_0_guess : float
        Trial value of phi(0).
    d : float
        Number of dimensions.
    r_min : float
        Radius where the numerics begin.
    r_max : float
        Radius where the integration stops.
    lambdap : float
        Dimensionless coupling.
    r_eval : ndarray
        array between r_min and r_max

    Returns
    -------
    radial : ndarray
        Radial points (equal to `r_eval`).
    phi : ndarray
        Field values at those points.
    """
    y0 = np.array([Initial_Phi(phi_0_guess,d , lambdap,r_min), Initialderiv(d, lambdap, r_min, phi_0_guess)])
    sol = solve_ivp(fun, [r_min, r_max], y0, t_eval = r_eval, method = 'DOP853',dense_output = True, atol = 1e-15, rtol = 1e-13, args = (d, lambdap))
    phi = sol.y[0]
    chi = sol.y[1]
    radial = sol.t
    return radial, phi


def Undershoot(phi):
    """
    Checks the field value at the boundary r_max. From the BCs the field should decay to zero. 
    If the bounce value becomes negative, it has overshot. If it remains positive, it has undershot.

    Parameters
    ----------
    phi: spline object
        field config

    Returns
    -------
    Bool: 
        True if undershoot, False if overshoot.
    """
    if phi[-1]< 0 :
        return False
    else:
        return True


def Bracket(lambdap,d, r_min, r_max, r_eval):
    """
    To find the appropriate phi_0 for a given dimensionless coupling, use ansatzs combined
    with the bisection method. For the initial values to use, define two extremes, one that 
    overshoots and one that undershoots.
    Iterate until both are found to define the bracket.

    Parameters
    ----------
    lambdap: float
        dim-less coupling
    d: float
        number of dimensions
    r_min, r_max: floats
        upper and lower radial bounds for the integrator.
    r_eval: ndarray
        array from r_min to r_max of N datapoints.
    
    Returns
    -------
    phi_0_under: float
        lower value for the bracket. Undershoots.
    phi_0_over: float
        upper value for the bracket. Overshoots.
    """
    phi_0_under = 0
    phi_guess = phi_0_under
    step = 0.01
    while Undershoot(Solver(phi_guess, d, r_min, r_max, lambdap, r_eval)[1]):
        phi_guess += step
        phi_0_over = phi_guess
    return phi_0_under, phi_0_over


def shooting(d, lambdap, r_min, r_max, r_eval):
    """
    Solves for the bounce background.
    1) Determines the bracket values.
    2) Initialises the radial (R) and solution (solution) arrays with None. Ensures that the loop breaks if a solution is found.
    3) In the while loop, take the bisection of the bracket value. Determine if the given phi_0_guess under- or over-shoots.
        a) If overshoot: replace the upper bracket value with the phi_0_guess suggested.
        b) If undershoot: replace the lower bracket value with the phi_0_guess suggested.
    4) Repeat until the overshoot and undershoot values in the update bracket match to within tolerance.
    5) Test the solution found. If the test is passed the solution is accepted.

     Parameters
    ----------
    lambdap: float
        dim-less coupling
    d: float
        number of dimensions
    r_min, r_max: floats
        upper and lower radial bounds for the integrator.
    r_eval: ndarray
        array from r_min to r_max of N datapoints.
    
    Returns
    -------
    R: dense output object
        radial array
    solution: dense output object
        phi(r)
    phi0: 
        value of the bounce at r=0
    """
    phi_0_under, phi_0_over = Bracket(lambdap, d, r_min, r_max, r_eval)

    R, solution = None, None

    tolerance = 1e-10
    max_iterations = 1000
    i = 0

    while i < max_iterations:
        phi_0_guess = 0.5 * (phi_0_over + phi_0_under)
        t, phi = Solver(phi_0_guess, d, r_min, r_max, lambdap, r_eval)

        R = t
        solution = phi

        # Check for convergence
        if abs(phi_0_over - phi_0_under) < tolerance:
          phi0 = phi_0_guess
          break

        if Undershoot(phi):
          phi_0_under = phi_0_guess
        else:
          phi_0_over = phi_0_guess

        i += 1
    Test = Shooting_Test(d, lambdap, R, solution)
    if Test:
        return R, solution, phi0
    else:
        return None, None, None

def Shooting_Test(d, lambdap, R, phi, tol=1e-3):
    """
    Determines if the shooting solution determined via the solver IS the solution ot the differential equation via the maximum residual
    across the r domain. The edges, namerly r_min and r_max are dropped to avoid artifacts on the derivatives of phi.
    If the maximal residual across the domain is lower than the set tolerance of 1e-3, the solution is accepted.

    Parameters
    ---------
    lambdap: float
        dim-less coupling
    d: float
        number of dimensions
    R: dense output object
        radial dense output array
    phi: dense output object
        bounce solution as dense output
    tol: float
        tolerance for test on residuals

    Returns
    ------
    Test: Bool
        True if test is passed, False if test is failed.
    """ 
   
    phi_of_r = CubicSpline(R, phi)
    phi_p = phi_of_r.derivative(1)
    phi_pp = phi_of_r.derivative(2)

    r_test = R[1:-1]          # drop edges
    DE = -phi_pp(r_test) - (d-1)/r_test * phi_p(r_test) + Vp(phi_of_r(r_test), lambdap)

    max_residual = float(np.max(np.abs(DE)))
    Test = np.all(max_residual <= tol)
    return Test


def clean_profile(r_eval, phi, threshold=1e-5):
    """
    To avoid large r artifacts, force the shooting solution to zero once a threshold close to zero is passed.
    Bounce is monotonic, so it should decay to zero smoothly. Determines the index at which the threshold is crossed
    and sets all values thereafter to zero.

    Parameters
    ----------
    r_eval: ndarray
        array of radial coordinates.
    phi: Spline object
        Continuous bounce solution.

    Returns
    -------
    phi(r), the bounce evaluated on the points of r_eval.
    """
    phi = phi(r_eval)
    
    # Find the index where the field first drops below the threshold
    idx = np.where(np.abs(phi) < threshold)[0]
    if len(idx) > 0:

        first_zero_idx = idx[0]
        # Force the field to zero from that point onward
        phi[first_zero_idx:] = 0
    return phi