"""
Numerical solver for the high-dimensional set of self-consistent equations.

This file implements the routines used to solve the system of
scalar equations arising in the high-dimensional analysis of
extensive-width quadratic neural networks, and to compute the
corresponding learning curves for the MSE, training loss, in-sample error,
as functions of parameters alpha, kappa, kappastar, lambda, Delta.
"""

import pickle
import numpy as np
import torch
from scipy.integrate import quad
from scipy.optimize import brentq
from numpy.polynomial import Polynomial

# Some functions (solve_poly, rho, edges_rho) are adapted from:
# https://github.com/SPOC-group/ExtensiveWidthQuadraticSamples,
# code for the paper “Bayes-optimal learning of an extensive-width neural network from quadratically many samples” (NeurIPS 2024).



def edges_MP(kappastar):
    """Edges of the Marchenko-Pastur distribution with aspect ratio kappastar"""
    return [(1 - 1 / np.sqrt(kappastar))**2, (1 + 1 / np.sqrt(kappastar))**2]


def solve_poly(z, xi, kappastar):
    """Solve the cubic polynomial equation for the Stieltjes transform of the additive free convolution between:
        - Marchenko-Pastur distribution with aspect ratio kappastar
        - Semicircular distribution with variance xi"""
    p = [xi, -(z + kappastar * xi), 1 - kappastar + kappastar * z, -kappastar]
    return np.roots(p)


def rho(x, xi, kappastar):
    """Density of the free additive convolution between Marchenko-Pastur with parameter kappastar
    and semicircular density with variance xi
    Uses the Stieltjes-Perron inversion formula"""
    return np.max(np.imag(solve_poly(x + 1e-6j * min(xi**2, 1), xi, kappastar))) / np.pi


def edges_rho(sigma, kappa):
    """Returns the edges of the support of rho. Depending on the value of sigma, kappa, the support may be:
        - one interval (2 edges)
        - two disjoint intervals (4 edges)"""
    edges_poly = Polynomial([-4 * sigma ** 2 + 12 * kappa * sigma ** 2 - 12 * kappa ** 2 * sigma ** 2 + 4 * kappa ** 3 * sigma ** 2 - 8 * kappa ** 2 * sigma ** 4 - 20 * kappa ** 3 * sigma ** 4 + kappa ** 4 * sigma ** 4 - 4 * kappa ** 4 * sigma ** 6,
                             8 * kappa * sigma ** 2 + 2 * kappa ** 2 * sigma ** 2 - 10 * kappa ** 3 * sigma ** 2 + 8 * kappa ** 3 * sigma ** 4 - 2 * kappa ** 4 * sigma ** 4,
                             1 - 2 * kappa + kappa ** 2 - 2 * kappa ** 2 * sigma ** 2 + 8 * kappa ** 3 * sigma ** 2 + kappa ** 4 * sigma ** 4,
                             -2 * kappa - 2 * kappa ** 2 - 2 * kappa ** 3 * sigma ** 2,
                             kappa ** 2])
    roots_all = edges_poly.roots()
    real_roots = np.real(roots_all[np.abs(np.imag(roots_all)) < 1e-6])
    return np.sort(real_roots)


def hilbert(x, xi, kappastar):
    """Hilbert transform of rho at point x"""
    z = solve_poly(x + 1e-6j * min(xi**2, 1), xi, kappastar)
    re, im = np.real(z), np.imag(z)
    return re[np.argmax(im)]


def cdf(u, xi, kappastar):
    """Cumulative distribution function of rho
    Computed by numerical integration of rho"""
    edges = edges_rho(np.sqrt(xi), kappastar)
    result = integrate(lambda x: rho(x, xi, kappastar), u, edges, eps=1e-6)
    return min(max(result, 0), 1)


def inv_cdf(p, xi, kappastar):
    """Inverse CDF of rho
    Solves for x such that CDF(x) = p"""
    def objective(x):
        return cdf(x, xi, kappastar) - p
    edges = edges_rho(np.sqrt(xi), kappastar)
    sol = brentq(objective, edges[0], edges[-1])
    return sol


def compute_omega(xi, kappa, kappastar):
    """Compute the selection threshold omega defined as:
        int_omega d rho = kappa"""
    return inv_cdf(kappa, xi, kappastar)


def integrate(func, b, edges, eps=1e-4):
    """Integrate a function against the spectral density rho from x = b to the upper edge of the support.
    Different cases correspond to different possible lengths of edges and different positions of b with respect to these edges"""
    if b >= edges[-1]:
        return 0
    if len(edges) == 2:
        if b <= edges[0]:
            return quad(lambda x: func(x), edges[0], edges[-1], epsabs=eps, epsrel=eps)[0]
        else:
            return quad(lambda x: func(x), b, edges[-1], epsabs=eps, epsrel=eps)[0]
    elif len(edges) == 4:
        if b <= edges[0]:
            return quad(lambda x: func(x), edges[0], edges[1], epsabs=eps, epsrel=eps)[0] + quad(lambda x: func(x), edges[2], edges[3], epsabs=eps, epsrel=eps)[0]
        elif edges[0] < b <= edges[1]:
            return quad(lambda x: func(x), b, edges[1], epsabs=eps, epsrel=eps)[0] + quad(lambda x: func(x), edges[2], edges[3], epsabs=eps, epsrel=eps)[0]
        elif edges[1] < b <= edges[2]:
            return quad(lambda x: func(x), edges[2], edges[3], epsabs=eps, epsrel=eps)[0]
        elif b > edges[2]:
            return quad(lambda x: func(x), b, edges[3], epsabs=eps, epsrel=eps)[0]


def integral_hilbert(b, edges, q, kappastar, xi, eps):
    """Compute the integral involving the Hilbert transform in the system of equations
        int_b rho(x) H(x) (x-q) dx, where H is the Hilbert transform of rho"""
    return integrate(lambda x: rho(x, xi, kappastar) * hilbert(x, xi, kappastar) * (x-q), b, edges, eps)


def integral_square(b, edges, q, kappastar, xi, eps):
    """Compute the integral involving the squares in the system of equations
        int_b rho(x) (q^2 - x^2) dx"""
    return integrate(lambda x: rho(x, xi, kappastar) * (q**2 - x**2), b, edges, eps)


def compute_rank(b, edges, kappastar, xi, eps):
    """Compute the effective rank of the solution
        int_b rho(x) dx"""
    return integrate(lambda x: rho(x, xi, kappastar), b, edges, eps)


def functional(q, omega, edges, kappastar, xi, Delta, lam, eps, MSE_store):
    """Scalar functional whose root defines the fixed-point equation for q
    MSE_store is used to cache the last computed MSE during root finding, in order to avoid recomputing it after convergence"""
    b = max(q, omega)

    # Integrals appearing in the self-consistent equations
    integral = integral_hilbert(b, edges, q, kappastar, xi, eps)
    square = integral_square(b, edges, q, kappastar, xi, eps)

    # Mean-squared error associated with the current q
    MSE = 1 + 1 / kappastar + square + 4 * xi * integral
    MSE_store.append(MSE)

    if lam > 0:
        return 2 * xi * integral / (MSE + Delta / 2) + lam / q - 1
    return 2 * xi * integral / (MSE + Delta / 2) - 1


def bracket_upward(qmin, qmax, func, max_expand=60, expand_factor=1.5):
    """Find a bracketing interval [qmin, qmax] such that func(qmin) and func(qmax) have opposite signs
    Starting from an initial interval, the upper bound qmax is expanded geometrically until a sign change is found or max_expand is reached."""
    fa = func(qmin)
    fb = func(qmax)

    # Check whether the initial bracket already contains a root
    if fa * fb < 0:
        return qmin, qmax, fa, fb
    it = 0
    while it < max_expand:
        # Shift the interval upward: old qmax becomes new qmin
        qmin, fa = qmax, fb

        # Expand the upper bound multiplicatively
        qmax = qmax * expand_factor
        fb = func(qmax)

        # Stop as soon as a sign change is detected
        if fa * fb < 0:
            return qmin, qmax, fa, fb
        it += 1

    # Failed to find a valid bracket
    return None, None, fa, fb


def numeric_derivatives(q, xi_val, omega, edges, kappastar, Delta, lam, eps, h_rel=1e-6):
    """Computes partial derivatives of the functional with respect to q, xi
    using centered finite differences"""

    # Adaptive step sizes to avoid underflow near zero
    hq = max(abs(q) * h_rel, 1e-10)
    hxi = max(abs(xi_val) * h_rel, 1e-10)

    # Partial derivative with respect to q
    f_qp = functional(q + hq, omega, edges, kappastar, xi_val, Delta, lam, eps, [])
    f_qm = functional(q - hq, omega, edges, kappastar, xi_val, Delta, lam, eps, [])
    derivative_q = (f_qp - f_qm) / (2.0 * hq)

    # Partial derivative with respect to xi
    f_xip = functional(q, omega, edges, kappastar, xi_val + hxi, Delta, lam, eps, [])
    f_xim = functional(q, omega, edges, kappastar, xi_val - hxi, Delta, lam, eps, [])
    derivative_xi = (f_xip - f_xim) / (2.0 * hxi)

    return derivative_q, derivative_xi


def find_root(qmin, qmax, omega, edges, kappastar, xi_val, Delta, lam, eps, max_expand=80):
    """Find the solution q of the fixed-point equation by:
        - Expanding an initial bracket upward until a sign change is found,
        - Applying Brent's method on the resulting interval.
    Returns the solution q and the corresponding value of the MSE, or (nan, nan) if no valid bracket was found"""
    MSE_store = []

    # Find a bracketing interval for the root
    qmin, qmax, f_qmin, f_qmax = bracket_upward(qmin, qmax,
                                                lambda x: functional(x, omega, edges, kappastar, xi_val, Delta, lam, eps, MSE_store),
                                                max_expand=max_expand, expand_factor=1.5)

    # Apply Brent's method if a sign change was found
    if qmin is not None and f_qmax * f_qmin < 0:
        q_val = compute_q(kappastar, xi_val, lam, Delta, qmin, qmax, omega, edges, eps, MSE_store)
        # Last stored MSE corresponds to the converged solution
        MSE_val = MSE_store[-1]
        return q_val, MSE_val
    else:
        # No valid bracket
        return float('nan'), float('nan')


def compute_q(kappastar, xi, lam, Delta, qmin, qmax, omega, edges, eps, MSE_store):
    """Solve F(q) = 0 for q in the interval [qmin, qmax] using Brent's method.
    Tolerance adapted to the scale of xi"""
    tol = min(xi / 100, 1e-6)
    return brentq(lambda x: functional(x, omega, edges, kappastar, xi, Delta, lam, eps, MSE_store), qmin, qmax, xtol=tol, rtol=tol)


def update_variables(xi, kappa, kappastar):
    """Precomputes quantities for the root-finding procedure
        - omega: selection threshold from the top fraction kappa
        - edges: spectral support of rho
        - eps: numerical integration tolerance adapted to xi"""
    omega = 0.0 if kappa >= 1 else compute_omega(xi, kappa, kappastar)
    edges = edges_rho(np.sqrt(xi), kappastar)
    eps = max(min(xi, 0.01), 1e-10) / 100.0
    return omega, edges, eps


def simulate_simple(kappa, kappastar, xi, lam, Delta, qstart=None):
    """Simulate the self-consistent system in the regime where the solution q(xi) is unique for all xi (no double-descent / no bifurcation).
    Parameters:
        kappa, kappastar, lam, Delta : nonnegative scalars
        xi : 1D torch tensor, sorted in increasing order
        qstart : optional initial guess for q at the first xi"""

    mse = torch.zeros(len(xi))
    alpha = torch.zeros(len(xi))
    q = torch.zeros(len(xi))
    rank = torch.zeros(len(xi))

    # Initial bracketing interval for q at the first xi
    if qstart is None:
        qmin = lam
        qmax = 2 * lam if lam > 0 else 0.1
    else:
        qmin = qstart
        qmax = 1.5 * qstart

    max_iters = 50

    for i in range(len(xi)):
        xi_val = xi[i].item()
        omega, edges, eps = update_variables(xi_val, kappa, kappastar)

        # Solve the fixed-point equation for q at current xi
        q_val, MSE_val = find_root(qmin, qmax, omega, edges, kappastar, xi_val, Delta, lam, eps, max_iters)
        if not np.isnan(q_val):
            # Warm-start for the next xi: shift the bracket around the new root
            qmin = q_val
            qmax = max(2 * q_val, qmax)
            rank_val = compute_rank(max(q_val, omega), edges, kappastar, xi_val, eps)
        else:
            rank_val = float('nan')

        # Compute the corresponding alpha from the closed-form relation with the MSE
        alpha_val = (MSE_val + Delta / 2) / (2 * xi_val)

        mse[i] = MSE_val
        alpha[i] = alpha_val
        q[i] = q_val
        rank[i] = rank_val

        print(f"\r[step {i:04d}] xi={xi_val:.8f}   alpha={alpha_val:.8f}   q={q_val:.8f}   MSE={MSE_val:.8f}", end="", flush=True)

    print()
    loss = lam ** 2 * alpha * xi / q ** 2
    ise = loss + Delta / 4 * (1 - 2 * lam / q)

    ind = ~torch.isnan(q)
    return mse[ind], loss[ind], ise[ind], alpha[ind], q[ind], xi[ind], rank[ind], None


def simulate_singular(kappa, kappastar, Delta, xi):
    """Simulate the self-consistent system in the regime where lam=0, Delta > 0
    and q=0 is imposed as a solution.
    In this case the system is evaluated at q=0,
    and the variable 'test' is used to check the validity of the solution"""

    mse = torch.zeros(len(xi))
    alpha = torch.zeros(len(xi))
    test = torch.zeros(len(xi))
    rank = torch.zeros(len(xi))

    # Loop over increasing xi (no continuation needed since q is fixed to zero)
    for i in range(len(xi)):
        xi_val = xi[i].item()
        omega, edges, eps = update_variables(xi_val, kappa, kappastar)
        cutoff = max(omega, 0.0)

        # Integrals evaluated at q = 0
        integral = integral_hilbert(cutoff, edges, 0, kappastar, xi_val, eps)
        square = integral_square(cutoff, edges, 0, kappastar, xi_val, eps)
        MSE = 1 + 1 / kappastar + square + 4 * xi_val * integral
        mse[i] = MSE
        alpha[i] = (MSE + Delta / 2) / (2 * xi_val)

        # Validity test for the q = 0 branch, the solution is meaningful only when test > 0
        test[i] = alpha[i] - integral

        # Rank = mass of rho above cutoff
        rank[i] = compute_rank(cutoff, edges, kappastar, xi_val, eps)

        print(f"\r[step {i:04d}] xi={xi_val:.8f}   alpha={alpha[i]:.8f}   MSE={mse[i]:.8f}", end="", flush=True)

    print()
    loss = xi * test ** 2 / alpha
    ise = xi * test ** 2 / alpha + Delta / 4 * (1 - 2 * test / alpha)
    q = torch.zeros(len(xi))

    return mse, loss, ise, alpha, q, xi, rank, test


def simulate_noisy(kappa, kappastar, xi_start, xi_end, lam, Delta):
    """Simulate the self-consistent system in the presence of double descent, where the equation
    F(q, xi) = 0 may admit multiple solutions q for the same value of xi.
    Pseudo–arclength continuation method in the plane (y, q) with y = log(xi) to track
    a continuous solution branch across turning points, where standard continuation in xi would fail.

    Phase I  : pseudo–arclength continuation starting from xi_start,
               following the branch until alpha drops below a threshold or until xi reaches xi_end.

    Phase II : once the branch becomes single-valued again, switch back
               to simple continuation in xi using simulate_simple.

    Returns: Concatenated solution from both phases"""

    xi = []
    q = []
    mse = []
    alpha = []
    rank = []

    # Arclength step size control
    ds_min = 5e-4
    ds_max = 0.01
    ds_init = ds_max

    # Newton parameters
    newton_tol = 1e-6
    newton_max_iter = 20

    # Threshold on alpha to decide when to exit Phase I
    alpha_switch = 0.15

    qmin = lam
    qmax = 2 * lam

    # ------------------------------------------------------------------
    # Initialization: compute first two points to define a tangent
    # ------------------------------------------------------------------
    xi0 = xi_start.item()
    omega0, edges0, eps0 = update_variables(xi0, kappa, kappastar)

    q0, mse0 = find_root(qmin, qmax, omega0, edges0, kappastar, xi0, Delta, lam, eps0)
    alpha0 = (mse0 + Delta / 2) / (2 * xi0)
    rank0 = compute_rank(max(q0, omega0), edges0, kappastar, xi0, eps0)

    xi.append(xi0)
    q.append(q0)
    mse.append(mse0)
    alpha.append(alpha0)
    rank.append(rank0)

    # Update bracket for next root
    qmin = q0
    qmax = max(2 * q0, qmax)

    # Second point: small step in xi to initialize the tangent direction
    xi1 = xi0 * 1.001
    omega1, edges1, eps1 = update_variables(xi1, kappa, kappastar)

    q1, mse1 = find_root(qmin, qmax, omega1, edges1, kappastar, xi1, Delta, lam, eps1)
    alpha1 = (mse1 + Delta / 2) / (2 * xi1)
    rank1 = compute_rank(max(q1, omega1), edges1, kappastar, xi1, eps1)

    xi.append(xi1)
    q.append(q1)
    mse.append(mse1)
    alpha.append(alpha1)
    rank.append(rank1)

    # Initial tangent vector in (y, q), with y = log(xi)
    dy = np.log(xi1) - np.log(xi0)
    dq = q1 - q0
    norm = np.hypot(dy, dq)
    t_y = dy / norm
    t_q = dq / norm

    # Initial arclength step size, rescaled by alpha
    ds = min(max(ds_init, ds_min), ds_max / alpha1)

    step = 0
    # ------------------------------------------------------------------
    # Phase I: pseudo–arclength loop
    # ------------------------------------------------------------------
    while step < 5e3:
        # Prediction: advance along the tangent by arclength ds
        xi_pred = xi[-1] * np.exp(t_y * ds)
        q_pred = q[-1] + t_q * ds
        newton_its = 0
        converged = False

        xi_curr = xi_pred
        q_curr = q_pred

        # Solve the system enforcing orthogonality to the tangent (pseudo–arclength condition)
        #   F(q, xi) = 0
        #   (y - y_pred, q - q_pred) · (t_y, t_q) = 0
        while newton_its < newton_max_iter:
            newton_its += 1

            omega, edges, eps = update_variables(xi_curr, kappa, kappastar)

            Fval = functional(q_curr, omega, edges, kappastar, xi_curr, Delta, lam, eps, [])
            F_q, F_xi = numeric_derivatives(q_curr, xi_curr, omega, edges, kappastar, Delta, lam, eps)
            F_y = F_xi * xi_curr

            # Linear system for (dy_corr, dq_corr)
            A = np.array([[F_y, F_q], [t_y, t_q]])
            rhs = np.array([-Fval, 0.0])

            try:
                dy_corr, dq_corr = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                # Jacobian is singular: abort this step
                break

            corr_norm = abs(dy_corr) + abs(dq_corr)

            # If correction is too large, we may jump to another branch: reject
            if corr_norm > 0.5 * ds:
                break

            # Apply correction
            xi_curr *= np.exp(dy_corr)
            q_curr += dq_corr

            # Convergence test
            if abs(dy_corr) + abs(dq_corr) < newton_tol:
                converged = True
                break

        # If Newton failed, reduce arclength and retry
        if not converged:
            ds *= 0.65
            if ds < ds_min:
                # Step too small: probably stuck near a singular point
                # Breaking and going to phase II
                break
            continue

        # Accept the corrected point and compute observables
        omega, edges, eps = update_variables(xi_curr, kappa, kappastar)
        mse_val = []
        _ = functional(q_curr, omega, edges, kappastar, xi_curr, Delta, lam, eps, mse_val)
        mse_val = mse_val[-1]

        alpha_val = (mse_val + Delta / 2) / (2 * xi_curr)
        rank_val = compute_rank(max(q_curr, omega), edges, kappastar, xi_curr, eps)

        print(f"\r[step {step:04d}] xi={xi_curr:.8f}   alpha={alpha_val:.8f}   q={q_curr:.8f}   MSE={mse_val:.8f}",
              end="", flush=True)

        xi.append(xi_curr)
        q.append(q_curr)
        mse.append(mse_val)
        alpha.append(alpha_val)
        rank.append(rank_val)

        # Exit Phase I when alpha becomes small: branch is single-valued again
        if alpha_val < alpha_switch:
            break

        # Adaptive arclength control based on Newton convergence
        # Small newton_its: OK to take bigger steps
        # Large newton_its: should reduce arclength for stability
        if newton_its <= 2:
            ds *= 1.05
        elif newton_its >= 5:
            ds *= 0.9

        if newton_its <= 3 and alpha_val < 0.25:
            ds *= 1.2

        # Update tangent using the last two accepted points
        dy_new = np.log(xi[-1]) - np.log(xi[-2])
        dq_new = q[-1] - q[-2]
        norm_new = np.sqrt(dy_new ** 2 + dq_new ** 2)

        t_y_new = dy_new / norm_new
        t_q_new = dq_new / norm_new

        # Enforce step size bounds
        ds = np.clip(ds, ds_min, ds_max / alpha_val)
        t_y, t_q = t_y_new, t_q_new

        # Stop if xi exceeds the target range
        if xi[-1] > xi_end:
            break

        step += 1

    xi_P1 = torch.tensor(xi)
    q_P1 = torch.tensor(q)
    mse_P1 = torch.tensor(mse)
    alpha_P1 = torch.tensor(alpha)
    rank_P1 = torch.tensor(rank)

    loss_P1 = lam ** 2 * alpha_P1 * xi_P1 / q_P1 ** 2
    ise_P1 = loss_P1 + Delta / 4 * (1 - 2 * lam / q_P1)

    print()

    # ------------------------------------------------------------------
    # Phase II: simple continuation once the branch is single-valued again
    # ------------------------------------------------------------------
    xi_new = xi[-1] * 1.001

    if alpha[-1] > alpha_switch:
        # Phase I exited early: use a denser grid to stabilize Phase II
        xi_phase2 = torch.cat([torch.logspace(np.log10(xi_new), np.log10(5 * xi_new), 100),
                               torch.logspace(np.log10(5 * xi_new), np.log10(xi_end), 100)[1:]])
    else:
        xi_phase2 = torch.logspace(np.log10(xi_new), np.log10(xi_end), 100)

    mse_P2, loss_P2, ise_P2, alpha_P2, q_P2, xi_P2, rank_P2, _ = simulate_simple(kappa, kappastar, xi_phase2, lam, Delta, qstart=q[-1])

    # Concatenate both phases
    mse = torch.cat((mse_P1, mse_P2))
    loss = torch.cat((loss_P1, loss_P2))
    ise = torch.cat((ise_P1, ise_P2))
    alpha = torch.cat((alpha_P1, alpha_P2))
    q = torch.cat((q_P1, q_P2))
    xi_new = torch.cat((xi_P1, xi_P2))
    rank = torch.cat((rank_P1, rank_P2))

    ind = ~torch.isnan(q)
    return mse[ind], loss[ind], ise[ind], alpha[ind], q[ind], xi_new[ind], rank[ind], None


def interpolation_test(xi, kappa, kappastar, Delta, alpha_store):
    """Compute the solution of the system of equations at q = 0
    The existence of a solution is guaranteed for alpha > alpha_inter
    Returns a quantity that is positive for alpha > alpha_inter and zero at alpha_inter
    The corresponding value of alpha is appended to alpha_store"""
    omega = 0 if kappa >= 1 else compute_omega(xi, kappa, kappastar)
    cutoff = max(omega, 0)
    edges = edges_rho(np.sqrt(xi), kappastar)
    integral = integral_hilbert(cutoff, edges, 0, kappastar, xi, eps=1e-4)
    square = integral_square(cutoff, edges, 0, kappastar, xi, eps=1e-4)
    MSE = 1 + 1 / kappastar + square + 4 * xi * integral
    alpha = (MSE + Delta / 2) / (2 * xi)
    alpha_store.append(alpha)
    return alpha - integral


def interpolation_threshold(kappa, kappastar, Delta):
    """ Finds xi such that interpolation_test(xi) = 0, and returns:
        - xi_inter  : value of xi at the interpolation threshold
        - alpha_inter : corresponding value of alpha"""
    alpha_store = []

    def objective(x):
        return interpolation_test(x, kappa, kappastar, Delta, alpha_store)

    lims = [Delta, max(100, 100 * Delta)]
    sol = brentq(objective, lims[0], lims[1])
    return sol, alpha_store[-1]


def simulate(kappa, kappastar, xi, lam, Delta, noisy):
    """Dispatch to the appropriate simulation regime.
        - noisy == False : Unique-solution regime (no double descent) → simulate_simple
        - noisy == True and lam > 0 : Double-descent regime with noise → simulate_noisy
        - noisy == True and lam == 0 : Singular regime with q = 0 branch → simulate_singular"""
    if not noisy:
        return simulate_simple(kappa, kappastar, xi, lam, Delta)
    elif lam > 0 and noisy:
        return simulate_noisy(kappa, kappastar, xi[0], xi[-1], lam, Delta)
    else:
        return simulate_singular(kappa, kappastar, Delta, xi)


class Simulator:
    """Class to run and store batches of simulations over grids of parameters"""
    def __init__(self, kappa, kappastar, xi, lam, Delta, noisy, ID):

        self.Nxi = xi.shape[-1]
        self.kappa = kappa
        self.kappastar = kappastar
        self.lam = lam
        self.Delta = Delta
        self.xi = xi
        
        self.noisy = noisy

        self.id = ID

        self.mse = None
        self.loss = None
        self.ise = None
        self.alpha = None
        self.q = None
        self.xi_new = None
        self.rank = None
        self.interpolation_test = None

    def simulate(self):
        """Runs simulations over all combinations of (kappa, lam)
            - broadcasts all scalar/vector parameters to a common grid,
            - reshapes xi to shape (Nkappa, Nlam, Nxi),
            - loops over all parameter pairs,
            - stores each solution as an object entry in 2D arrays."""

        # broadcast kappa and kappastar
        kappa = np.atleast_1d(self.kappa)
        kappastar = np.atleast_1d(self.kappastar)

        # repeat kappastar if scalar
        if kappastar.size == 1:
            kappastar = np.repeat(kappastar, kappa.size)

        # broadcast lam and Delta into matching 1D grids
        lam = np.atleast_1d(self.lam)
        Delta = np.atleast_1d(self.Delta)

        if lam.size > 1:
            # Multiple values of lam, single Delta
            lam_g = lam
            Delta_g = np.full(lam.size, Delta.item())
        elif Delta.size > 1:
            # Multiple values of Delta, single lam
            Delta_g = Delta
            lam_g = np.full(Delta.size, lam.item())
        else:
            # Both scalar
            lam_g = np.array([lam.item()])
            Delta_g = np.array([Delta.item()])

        # broadcast noisy
        noisy = np.asarray(self.noisy)
        if noisy.ndim == 0:
            noisy = np.full(lam_g.size, noisy.item())

        # Reshape xi to a 3D grid: (Nkappa, Nlam, Nxi)
        xi = self.xi
        if xi.ndim == 1:
            xi = xi[None, None, :]
        elif xi.ndim == 2:
            xi = xi[None, :, :] if xi.shape[0] == lam_g.size else xi[:, None, :]

        xi = xi.expand(kappa.size, lam_g.size, xi.size(-1))

        shape = (kappa.size, lam_g.size)

        mse = np.empty(shape, dtype=object)
        loss = np.empty(shape, dtype=object)
        ise = np.empty(shape, dtype=object)
        alpha = np.empty(shape, dtype=object)
        q = np.empty(shape, dtype=object)
        rank = np.empty(shape, dtype=object)
        xi_new = np.empty(shape, dtype=object)
        test = np.empty(shape, dtype=object)

        # Main loop over parameter grid
        total = kappa.size * lam_g.size
        count = 0

        for i in range(kappa.size):
            for j in range(lam_g.size):
                count += 1
                print(f"[{count}/{total}] kappa={kappa[i]}, kappastar={kappastar[i]}, lam={lam_g[j]}, Delta={Delta_g[j]}, noisy={noisy[j]}")
                mse[i, j], loss[i, j], ise[i, j], alpha[i, j], q[i, j], xi_new[i, j], rank[i, j], test[i, j] = simulate(kappa[i], kappastar[i], xi[i, j], lam_g[j], Delta_g[j], noisy[j])
                print()

        self.mse = mse.squeeze()
        self.loss = loss.squeeze()
        self.ise = ise.squeeze()
        self.alpha = alpha.squeeze()
        self.q = q.squeeze()
        self.xi_new = xi_new.squeeze()
        self.rank = rank.squeeze()
        self.interpolation_test = test.squeeze()


    def save(self):
        """Serialize the Simulator object to disk using pickle."""
        path = 'Simulators/SimHD_{}.pickle'.format(self.id)
        print("Saving Simulator...")
        print("ID = {}".format(self.id))
        print()
        with open(path, 'wb') as file:
            pickle.dump(self, file)
