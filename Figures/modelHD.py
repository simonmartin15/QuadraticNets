import pickle
import numpy as np
import torch
from numpy import exp, log
from scipy.integrate import quad
from scipy.optimize import brentq
from numpy.polynomial import Polynomial



def edges_MP(kappastar):
    return [(1 - 1 / np.sqrt(kappastar))**2, (1 + 1 / np.sqrt(kappastar))**2]


def solve_poly(z, xi, kappastar):
    p = [xi, -(z + kappastar * xi), 1 - kappastar + kappastar * z, -kappastar]
    return np.roots(p)


def rho(x, xi, kappastar):
    return np.max(np.imag(solve_poly(x + 1e-6j * min(xi**2, 1), xi, kappastar))) / np.pi


def edges_rho(sigma, kappa):
    edges_poly = Polynomial([-4 * sigma ** 2 + 12 * kappa * sigma ** 2 - 12 * kappa ** 2 * sigma ** 2 + 4 * kappa ** 3 * sigma ** 2 - 8 * kappa ** 2 * sigma ** 4 - 20 * kappa ** 3 * sigma ** 4 + kappa ** 4 * sigma ** 4 - 4 * kappa ** 4 * sigma ** 6,
                             8 * kappa * sigma ** 2 + 2 * kappa ** 2 * sigma ** 2 - 10 * kappa ** 3 * sigma ** 2 + 8 * kappa ** 3 * sigma ** 4 - 2 * kappa ** 4 * sigma ** 4,
                             1 - 2 * kappa + kappa ** 2 - 2 * kappa ** 2 * sigma ** 2 + 8 * kappa ** 3 * sigma ** 2 + kappa ** 4 * sigma ** 4,
                             -2 * kappa - 2 * kappa ** 2 - 2 * kappa ** 3 * sigma ** 2,
                             kappa ** 2])
    roots_all = edges_poly.roots()
    real_roots = np.real(roots_all[np.abs(np.imag(roots_all)) < 1e-6])
    return np.sort(real_roots)


def hilbert(x, xi, kappastar):
    z = solve_poly(x + 1e-6j * min(xi**2, 1), xi, kappastar)
    re, im = np.real(z), np.imag(z)
    return re[np.argmax(im)]


def cdf(u, xi, kappastar):
    edges = edges_rho(np.sqrt(xi), kappastar)
    result = integrate(lambda x: rho(x, xi, kappastar), u, edges, eps=1e-6)
    return min(max(result, 0), 1)


def inv_cdf(p, xi, kappastar):
    def objective(x):
        return cdf(x, xi, kappastar) - p
    edges = edges_rho(np.sqrt(xi), kappastar)
    sol = brentq(objective, edges[0], edges[-1])
    return sol


def compute_omega(xi, kappa, kappastar):
    return inv_cdf(kappa, xi, kappastar)


def integrate(func, b, edges, eps=1e-4):
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
    return integrate(lambda x: rho(x, xi, kappastar) * hilbert(x, xi, kappastar) * (x-q), b, edges, eps)


def integral_square(b, edges, q, kappastar, xi, eps):
    return integrate(lambda x: rho(x, xi, kappastar) * (q**2 - x**2), b, edges, eps)


def compute_rank(b, edges, kappastar, xi, eps):
    return integrate(lambda x: rho(x, xi, kappastar), b, edges, eps)


def functionnal(q, omega, edges, kappastar, xi, Delta, lam, eps, MSE_store):
    b = max(q, omega)
    integral = integral_hilbert(b, edges, q, kappastar, xi, eps)
    square = integral_square(b, edges, q, kappastar, xi, eps)
    MSE = 1 + 1 / kappastar + square + 4 * xi * integral
    MSE_store.append(MSE)

    return 2 * xi * integral / (MSE + Delta / 2) + lam / q - 1


def functionnal_zero(q, omega, edges, kappastar, xi, Delta, eps, MSE_store):
    b = max(q, omega)
    integral = integral_hilbert(b, edges, q, kappastar, xi, eps)
    square = integral_square(b, edges, q, kappastar, xi, eps)
    MSE = 1 + 1 / kappastar + square + 4 * xi * integral

    MSE_store.append(MSE)
    return 2 * xi * integral / (MSE + Delta / 2) - 1


def eval_functional(q, omega, edges, kappastar, xi_val, Delta, lam, eps, MSE_store):
    try:
        del MSE_store[:]
    except Exception:
        MSE_store = []

    f = functionnal(q, omega, edges, kappastar, xi_val, Delta, lam, eps, MSE_store)
    mse = MSE_store[-1] if len(MSE_store) > 0 else float('nan')
    return f, mse


def bracket_upward(q_left, q_right, eval_f, max_expand=60, expand_factor=1.5):
    fa = eval_f(q_left)
    fb = eval_f(q_right)
    if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
        return q_left, q_right, fa, fb
    it = 0
    while it < max_expand:
        q_left, fa = q_right, fb
        q_right = q_right * expand_factor
        fb = eval_f(q_right)
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
            return q_left, q_right, fa, fb
        it += 1
    return None, None, fa, fb


def numeric_derivatives(q, xi_val, omega, edges, kappastar, Delta, lam, eps, h_rel=1e-6):
    hq = max(abs(q) * h_rel, 1e-10)
    hxi = max(abs(xi_val) * h_rel, 1e-10)

    tmp = []
    fqph, _ = eval_functional(q + hq, omega, edges, kappastar, xi_val, Delta, lam, eps, tmp)
    tmp = []
    fqmh, _ = eval_functional(q - hq, omega, edges, kappastar, xi_val, Delta, lam, eps, tmp)
    F_q = (fqph - fqmh) / (2.0 * hq)

    tmp = []
    f_xip, _ = eval_functional(q, omega, edges, kappastar, xi_val + hxi, Delta, lam, eps, tmp)
    tmp = []
    f_xim, _ = eval_functional(q, omega, edges, kappastar, xi_val - hxi, Delta, lam, eps, tmp)
    F_xi = (f_xip - f_xim) / (2.0 * hxi)

    return F_q, F_xi


def find_initial_root(qmin, qmax, omega, edges, kappastar, xi_val, Delta, lam, eps, max_expand=80):
    def f_only(q):
        tmp = []
        f, _ = eval_functional(q, omega, edges, kappastar, xi_val, Delta, lam, eps, tmp)
        return f

    bracket = bracket_upward(qmin, qmax, f_only, max_expand=max_expand, expand_factor=1.5)
    if bracket[0] is None:
        return float('nan'), float('nan')
    a, b, fa, fb = bracket
    MSE_store = []
    try:
        q0 = brentq(lambda qq: f_only(qq), a, b, xtol=1e-8, rtol=1e-10, maxiter=200)
    except Exception as e:
        print(f"[find_initial_root] brentq failed xi={xi_val}: {e}")
        return float('nan'), float('nan')
    _, mse0 = eval_functional(q0, omega, edges, kappastar, xi_val, Delta, lam, eps, MSE_store)
    return q0, mse0


def compute_q_zero(kappastar, xi, Delta, qmin, qmax, omega, edges, eps, MSE_store):
    tol = min(xi / 100, 1e-6)
    try:
        sol = brentq(lambda x: functionnal_zero(x, omega, edges, kappastar, xi, Delta, eps, MSE_store), qmin, qmax,
                     xtol=tol, rtol=tol)
    except Exception as e:
        print(f"Root finding failed at xi={xi}: {e}")
        sol = float('nan')
    return sol


def compute_q(kappastar, xi, lam, Delta, qmin, qmax, omega, edges, eps, MSE_store):
    tol = min(xi / 100, 1e-6)
    try:
        sol = brentq(lambda x: functionnal(x, omega, edges, kappastar, xi, Delta, lam, eps, MSE_store), qmin, qmax,
                     xtol=tol, rtol=tol)
    except Exception as e:
        print(f"Root finding failed at xi={xi}: {e}")
        sol = float('nan')
    return sol


def simulate_zeroreg_simple(kappa, kappastar, xi, Delta, max_iters=50):
    mse = []
    loss = []
    alpha = []
    q = []
    rank = []
    xi_track = []

    qmin = 0
    qmax = 0.1

    for i in range(len(xi)):
        xi_val = xi[i].item()
        omega = 0 if kappa >= 1 else compute_omega(xi_val, kappa, kappastar)
        edges = edges_rho(xi_val ** 0.5, kappastar)
        MSE_store = []

        eps = max(min(xi_val, 0.01), 1e-10) / 100

        f_qmin = functionnal_zero(qmin, omega, edges, kappastar, xi_val, Delta, eps, MSE_store)
        f_qmax = functionnal_zero(qmax, omega, edges, kappastar, xi_val, Delta, eps, MSE_store)

        count = 0

        if f_qmin < 0:
            qmin = 0
            qmax = 0.1
            f_qmin = functionnal_zero(qmin, omega, edges, kappastar, xi_val, Delta, eps, MSE_store)

        while f_qmax > 0 and count < max_iters:
            qmin = qmax
            qmax *= 2
            f_qmax = functionnal_zero(qmax, omega, edges, kappastar, xi_val, Delta, eps, MSE_store)
            count += 1

        if f_qmax < 0 < f_qmin:
            q_val = compute_q_zero(kappastar, xi_val, Delta, qmin, qmax, omega, edges, eps, MSE_store)
            MSE_val = MSE_store[-1]
            alpha_val = (MSE_val + Delta / 2) / (2 * xi_val)
            rank_val = compute_rank(max(q_val, omega), edges, kappastar, xi_val, eps)

            q.append(q_val)
            mse.append(MSE_val)
            loss.append(0)
            alpha.append(alpha_val)
            rank.append(rank_val)
            xi_track.append(xi_val)

            qmin = q_val

        print(f"\r[step {i:04d}] xi={xi_val:.8f}   alpha={alpha_val:.8f}   q={q_val:.8f}   MSE={MSE_val:.8f}", end="",
              flush=True)

    alpha, ind = torch.sort(torch.tensor(alpha))

    return (torch.tensor(mse)[ind], torch.tensor(loss)[ind], alpha, torch.tensor(q)[ind],
            torch.tensor(xi_track)[ind], torch.tensor(rank)[ind])


def simulate_simple(kappa, kappastar, xi, lam, Delta, max_iters=50):
    mse = torch.zeros(len(xi))
    loss = torch.zeros(len(xi))
    alpha = torch.zeros(len(xi))
    q = torch.zeros(len(xi))
    rank = torch.zeros(len(xi))

    qmin = lam
    qmax = 2 * lam

    for i in range(len(xi)):
        xi_val = xi[i].item()
        omega = 0 if kappa >= 1 else compute_omega(xi_val, kappa, kappastar)
        edges = edges_rho(xi_val ** 0.5, kappastar)
        MSE_store = []

        eps = max(min(xi_val, 0.01), 1e-10) / 100

        f_qmin = functionnal(qmin, omega, edges, kappastar, xi_val, Delta, lam, eps, MSE_store)
        f_qmax = functionnal(qmax, omega, edges, kappastar, xi_val, Delta, lam, eps, MSE_store)
        count = 0
        if f_qmin < 0:
            qmin = lam
            qmax = 2 * lam
            f_qmin = functionnal(qmin, omega, edges, kappastar, xi_val, Delta, lam, eps, MSE_store)

        while f_qmax > 0 and count < max_iters:
            qmin = qmax
            qmax *= 2
            f_qmax = functionnal(qmax, omega, edges, kappastar, xi_val, Delta, lam, eps, MSE_store)
            count += 1

        if f_qmax < 0 < f_qmin:
            q_val = compute_q(kappastar, xi_val, lam, Delta, qmin, qmax, omega, edges, eps, MSE_store)
            MSE_val = MSE_store[-1]
            qmin = q_val
            rank_val = compute_rank(max(q_val, omega), edges, kappastar, xi_val, eps)
        else:
            q_val = float('nan')
            MSE_val = float('nan')
            rank_val = float('nan')

        alpha_val = (MSE_val + Delta / 2) / (2 * xi_val)

        mse[i] = MSE_val
        loss[i] = lam ** 2 * alpha_val * xi_val / q_val ** 2 + Delta / 4 * (1 - 2 * lam / q_val)
        alpha[i] = alpha_val
        q[i] = q_val
        rank[i] = rank_val

        print(f"\r[step {i:04d}] xi={xi_val:.8f}   alpha={alpha_val:.8f}   q={q_val:.8f}   MSE={MSE_val:.8f}", end="", flush=True)

    print()

    ind = ~torch.isnan(q)
    return mse[ind], loss[ind], alpha[ind], q[ind], xi[ind], rank[ind]


def simulate_singular(kappa, kappastar, Delta, xi, check=True):
    mse = torch.zeros(len(xi))
    alpha = torch.zeros(len(xi))
    crit = torch.zeros(len(xi))
    for i in range(len(xi)):
        omega = 0 if kappa >= 1 else compute_omega(xi[i], kappa, kappastar)
        cutoff = max(omega, 0)
        edges = edges_rho(np.sqrt(xi[i]), kappastar)
        integral = integral_hilbert(cutoff, edges, 0, kappastar, xi[i], eps=1e-4)
        square = integral_square(cutoff, edges, 0, kappastar, xi[i], eps=1e-4)
        MSE = 1 + 1 / kappastar + square + 4 * xi[i] * integral
        mse[i] = MSE
        alpha[i] = (MSE + Delta / 2) / (2 * xi[i])
        crit[i] = alpha[i] - integral
        print('\rStep[{0}/{1}], xi = {2}, alpha = {3}, MSE = {4}, crit = {5}'.format(i + 1, len(xi),
                                                                                     round(xi[i].item(), 3),
                                                                                     round(alpha[i].item(), 3),
                                                                                     round(mse[i].item(), 3),
                                                                                     crit[i] > 0), end="")

    print('')
    loss = mse / (2 * alpha ** 2) * crit + Delta / 4 * (1 - crit / alpha) ** 2
    if check:
        return mse, loss, alpha, None, xi, crit
    return mse, alpha


def simulate_noisy(kappa, kappastar, xi_start, xi_end, lam, Delta):
    Y_MAX = 700.0
    DY_MAX = 5.0
    XI_MIN = float(exp(-Y_MAX))
    XI_MAX = float(exp(Y_MAX))

    max_steps = int(5e3)
    ds_init = 1e-3
    ds_min = 1e-6
    ds_max = 2e-2

    newton_tol = 1e-6
    newton_max_iter = 10

    curv_up_thresh = 0.02
    curv_down_thresh = 0.2
    curv_up_factor = 1.4
    curv_down_factor = 0.6
    curv_stable_steps = 3
    curv_stable_factor = 1.3

    GAMMA = 1.0

    alpha_switch = 0.15

    qmin0 = lam
    qmax0 = 2 * lam

    y0 = float(log(float(xi_start)))
    y_target = float(log(float(xi_end)))

    def update_cache(xi):
        omega = 0.0 if kappa >= 1 else compute_omega(xi, kappa, kappastar)
        edges = edges_rho(np.sqrt(xi), kappastar)
        eps = max(min(xi, 0.01), 1e-10) / 100.0
        return omega, edges, eps

    xi0 = float(xi_start)
    omega_cache, edges_cache, eps_cache = update_cache(xi0)

    q0, mse0 = find_initial_root(qmin0, qmax0, omega_cache, edges_cache, kappastar, xi0, Delta, lam, eps_cache)

    xi_list = [xi0]
    q_list = [q0]
    mse_list = [mse0]
    alpha_list = [(mse0 + Delta / 2) / (2 * xi0)]
    rank_list = [compute_rank(max(q0, omega_cache), edges_cache, kappastar, xi0, eps_cache)]

    y1 = y0 + 1e-3
    y1 = float(np.clip(y1, -Y_MAX, Y_MAX))
    xi1 = float(exp(y1))
    omega1, edges1, eps1 = update_cache(xi1)

    q_left = max(1e-12, q0 * 0.9)
    q_right = max(q0 * 1.2, q_left * 1.1)

    def f_for_xi1(q):
        tmp = []
        f, _ = eval_functional(q, omega1, edges1, kappastar, xi1, Delta, lam, eps1, tmp)
        return f

    br = bracket_upward(q_left, q_right, f_for_xi1, max_expand=30, expand_factor=1.3)
    if br[0] is None:
        q1, mse1 = q0, mse0
        y1, xi1 = y0, xi0
        omega1, edges1, eps1 = omega_cache, edges_cache, eps_cache
    else:
        a1, b1, _, _ = br
        try:
            q1 = brentq(lambda qq: f_for_xi1(qq), a1, b1, xtol=1e-8, rtol=1e-10)
            tmp = []
            _, mse1 = eval_functional(q1, omega1, edges1, kappastar, xi1, Delta, lam, eps1, tmp)
        except Exception:
            q1, mse1 = q0, mse0
            y1, xi1 = y0, xi0
            omega1, edges1, eps1 = omega_cache, edges_cache, eps_cache

    xi_list.append(xi1)
    q_list.append(q1)
    mse_list.append(mse1)
    alpha_list.append((mse1 + Delta / 2) / (2 * xi1))
    rank_list.append(compute_rank(max(q1, omega1), edges1, kappastar, xi1, eps1))

    dy = y1 - y0
    dq = q1 - q0
    norm = np.hypot(dy, dq)
    t_y = dy / norm
    t_q = dq / norm
    t_y_prev, t_q_prev = t_y, t_q

    ds = float(ds_init)
    ds = min(max(ds, ds_min), ds_max)

    ds_max_soft = float(ds_max)
    ds_max_hard = 5.0 * float(ds_max)

    stable_curv_count = 0
    step = 2

    while step < max_steps:
        y_prev = float(log(xi_list[-1]))
        q_prev = q_list[-1]

        y_pred = y_prev + t_y * ds
        y_pred = float(np.clip(y_pred, -Y_MAX, Y_MAX))
        q_pred = q_prev + t_q * ds

        if (y_target > y0 and y_pred > y_target) or (y_target < y0 and y_pred < y_target):
            ds = (y_target - y_prev) / t_y if abs(t_y) > 1e-16 else 0.0
            y_pred = y_target
            q_pred = q_prev + t_q / GAMMA * ds
            y_pred = float(np.clip(y_pred, -Y_MAX, Y_MAX))

        y_curr = float(y_pred)
        y_curr = float(np.clip(y_curr, -Y_MAX, Y_MAX))
        xi_curr = float(exp(y_curr))
        xi_curr = float(np.clip(xi_curr, XI_MIN, XI_MAX))
        q_curr = float(q_pred)

        newton_its = 0
        converged = False

        while newton_its < newton_max_iter:
            newton_its += 1

            omega_c, edges_c, eps_c = update_cache(xi_curr)

            tmp = []
            try:
                Fval, _ = eval_functional(q_curr, omega_c, edges_c, kappastar, xi_curr, Delta, lam, eps_c, tmp)
            except Exception:
                Fval = np.nan

            last_Fval = Fval
            if not np.isfinite(Fval):
                break

            try:
                F_q, F_xi = numeric_derivatives(q_curr, xi_curr, omega_c, edges_c, kappastar, Delta, lam, eps_c)
            except Exception:
                break

            F_y = F_xi * xi_curr

            A = np.array([[F_y, F_q], [t_y, t_q]], dtype=float)
            rhs = np.array([-Fval, 0.0], dtype=float)

            try:
                dy_corr, dq_corr = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                break

            if not np.isfinite(dy_corr):
                break
            dy_corr = float(np.clip(dy_corr, -DY_MAX, DY_MAX))
            if not np.isfinite(dq_corr):
                break
            dq_corr = float(dq_corr)

            y_curr += dy_corr
            q_curr += dq_corr

            y_curr = float(np.clip(y_curr, -Y_MAX, Y_MAX))
            xi_curr = float(exp(y_curr))
            xi_curr = float(np.clip(xi_curr, XI_MIN, XI_MAX))

            if abs(dy_corr) + abs(dq_corr) < newton_tol:
                converged = True
                omega_cache, edges_cache, eps_cache = omega_c, edges_c, eps_c
                break

        if not converged:
            ds *= 0.5
            ds_max_soft = max(ds_max_soft * 0.7, ds_max)
            if ds < ds_min:
                print("\nds too small; stopping.")
                xi_phase2_start = xi_curr
                q_phase2_start = q_curr
                break
            continue

        tmp = []
        try:
            _, mse_corr = eval_functional(q_curr, omega_cache, edges_cache, kappastar, xi_curr, Delta, lam, eps_cache,
                                          tmp)
        except Exception:
            ds *= 0.5
            ds_max_soft = max(ds_max_soft * 0.7, ds_max)
            if ds < ds_min:
                print("\nds too small; stopping (post-eval error).")
                xi_phase2_start = xi_curr
                q_phase2_start = q_curr
                break
            continue

        omega_curr, edges_curr, eps_curr = update_cache(xi_curr)
        alpha_curr = (mse_corr + Delta / 2) / (2 * xi_curr)
        rank_curr = compute_rank(max(q_curr, omega_curr), edges_curr, kappastar, xi_curr, eps_curr)

        print(f"\r[step {step:04d}] xi={xi_curr:.8f}   alpha={alpha_curr:.8f}   q={q_curr:.8f}   MSE={mse_corr:.8f}",
              end="", flush=True)

        xi_list.append(xi_curr)
        q_list.append(q_curr)
        mse_list.append(mse_corr)
        alpha_list.append(alpha_curr)
        rank_list.append(rank_curr)

        if alpha_curr < alpha_switch:
            xi_phase2_start = xi_curr
            q_phase2_start = q_curr
            break

        if newton_its <= 2:
            ds *= 1.15
            ds_max_soft = min(ds_max_soft * 1.05, ds_max_hard)
        elif newton_its >= newton_max_iter - 1:
            ds *= 0.7
            ds_max_soft = max(ds_max_soft * 0.8, ds_max)

        ds = min(ds, ds_max_soft)

        dy_new = float(log(xi_list[-1])) - float(log(xi_list[-2]))
        dq_new = q_list[-1] - q_list[-2]

        norm_new = np.sqrt(dy_new * dy_new + (GAMMA * dq_new) ** 2)

        t_y_new = dy_new / norm_new
        t_q_new = GAMMA * dq_new / norm_new

        curv = np.hypot(t_y_new - t_y_prev, t_q_new - t_q_prev)

        if curv < curv_up_thresh:
            ds *= curv_up_factor
            ds_max_soft = min(ds_max_soft * 1.10, ds_max_hard)
            stable_curv_count += 1
        else:
            stable_curv_count = 0

        if curv > curv_down_thresh:
            ds *= curv_down_factor
            ds_max_soft = max(ds_max_soft * 0.7, ds_max)

        if stable_curv_count >= curv_stable_steps:
            ds *= curv_stable_factor
            ds_max_soft = min(ds_max_soft * 1.20, ds_max_hard)
            stable_curv_count = 0

        ds = float(np.clip(ds, ds_min, ds_max_soft))

        t_y_prev, t_q_prev = t_y_new, t_q_new
        t_y, t_q = t_y_new, t_q_new

        y_last = float(log(xi_list[-1]))
        if (y_target > y0 and y_last >= y_target - 1e-12) or \
                (y_target < y0 and y_last <= y_target + 1e-12):
            break

        step += 1

    print()
    N_PHASE2 = 60

    y_start = np.log(xi_phase2_start)
    y_end = np.log(xi_end)

    y_vals = np.linspace(y_start, y_end, N_PHASE2 + 1)[1:]

    q_prev = q_phase2_start

    for i, y in enumerate(y_vals, 1):

        y = float(np.clip(y, -Y_MAX, Y_MAX))
        xi = float(np.exp(y))
        xi = float(np.clip(xi, XI_MIN, XI_MAX))

        omega, edges, eps = update_cache(xi)

        q_left = max(1e-12, q_prev * 0.8)
        q_right = q_prev * 1.25
        try:
            for _ in range(10):
                fL, _ = eval_functional(q_left, omega, edges, kappastar, xi, Delta, lam, eps, [])
                fR, _ = eval_functional(q_right, omega, edges, kappastar, xi, Delta, lam, eps, [])
                if np.isfinite(fL) and np.isfinite(fR) and fL * fR < 0:
                    break
                q_left *= 0.7
                q_right *= 1.4
            else:
                continue

            q = brentq(lambda qq: eval_functional(qq, omega, edges, kappastar, xi, Delta, lam, eps, [])[0],
                       q_left, q_right, xtol=1e-8, rtol=1e-8)

            _, mse = eval_functional(q, omega, edges, kappastar, xi, Delta, lam, eps, [])

        except Exception:
            continue

        alpha = (mse + Delta / 2) / (2 * xi)
        print(f"\r[step {i:04d}] xi={xi:.8f}   alpha={alpha:.8f}   q={q:.8f}   MSE={mse:.8f}", end="", flush=True)

        xi_list.append(xi)
        q_list.append(q)
        mse_list.append(mse)
        alpha_list.append(alpha)
        rank_list.append(compute_rank(max(q, omega), edges, kappastar, xi, eps))

        q_prev = q

    print()

    mse = np.array(mse_list)
    alpha = np.array(alpha_list)
    q = np.array(q_list)
    xi_new = np.array(xi_list)
    rank = np.array(rank_list)

    loss = lam ** 2 * alpha * xi_new / q ** 2 + Delta / 4 * (1 - 2 * lam / q)

    return mse, loss, alpha, q, xi_new, rank


def criterion(xi, kappa, kappastar, Delta, alpha_store):
    omega = 0 if kappa >= 1 else compute_omega(xi, kappa, kappastar)
    cutoff = max(omega, 0)
    edges = edges_rho(np.sqrt(xi), kappastar)
    integral = integral_hilbert(cutoff, edges, 0, kappastar, xi, eps=1e-4)
    square = integral_square(cutoff, edges, 0, kappastar, xi, eps=1e-4)
    MSE = 1 + 1 / kappastar + square + 4 * xi * integral
    alpha = (MSE + Delta / 2) / (2 * xi)
    alpha_store.append(alpha)
    return alpha - integral


def interpolation_treshold(kappa, kappastar, Delta):
    alpha_store = []

    def objective(x):
        return criterion(x, kappa, kappastar, Delta, alpha_store)

    lims = [Delta, max(100, 100 * Delta)]
    sol = brentq(objective, lims[0], lims[1])
    return sol, alpha_store[-1]


def simulate(kappa, kappastar, xi, lam, Delta, noisy):
    if lam > 0 and noisy:
        return simulate_noisy(kappa, kappastar, xi[0], xi[-1], lam, Delta)
    elif lam > 0 and not noisy:
        return simulate_simple(kappa, kappastar, xi, lam, Delta)
    elif lam == 0 and not noisy:
        return simulate_zeroreg_simple(kappa, kappastar, xi, Delta)
    else:
        return simulate_singular(kappa, kappastar, Delta, xi)



class Simulator:
    def __init__(self, kappa, kappastar, xi, lam, Delta, noisy, ID):

        self.Nxi = xi.shape[-1]
        self.kappa = kappa
        self.kappastar = kappastar
        self.lam = lam
        self.Delta = Delta
        self.xi = xi
        
        self.noisy = noisy

        self.cutoff = 1e-6

        self.id = ID

        self.mse = None
        self.loss = None
        self.alpha = None
        self.q = None
        self.xi_new = None
        self.rank = None

    def simulate(self):
        kappa = np.atleast_1d(self.kappa)
        kappastar = np.atleast_1d(self.kappastar)
        lam = np.atleast_1d(self.lam)
        Delta = np.atleast_1d(self.Delta)

        nk = len(kappa)
        nk_star = len(kappastar)
        nl = len(lam)
        nd = len(Delta)

        k_scalar = nk == 1
        kstar_scalar = nk_star == 1
        l_scalar = nl == 1
        d_scalar = nd == 1

        nk_eff = max(nk, nk_star)
        noisy = np.asarray(self.noisy)

        if noisy.ndim == 0:
            noisy = np.full((nl, nd), noisy)

        elif noisy.ndim == 1:
            if noisy.size == nl:
                noisy = noisy[:, None]
            elif noisy.size == nd:
                noisy = noisy[None, :]
            else:
                raise ValueError("1D noisy must match len(lam) or len(Delta)")

        elif noisy.ndim == 2:
            if noisy.shape != (nl, nd):
                raise ValueError("2D noisy must have shape (len(lam), len(Delta))")

        else:
            raise ValueError("noisy must be scalar, 1D, or 2D")

        xi = self.xi

        if xi.ndim == 1:
            def get_xi(i, j, k):
                return xi
        else:
            expected_shape = []
            if not k_scalar or not kstar_scalar:
                expected_shape.append(nk_eff)
            if not l_scalar:
                expected_shape.append(nl)
            if not d_scalar:
                expected_shape.append(nd)

            def get_xi(i, j, k):
                idx = []
                if not k_scalar or not kstar_scalar:
                    idx.append(i)
                if not l_scalar:
                    idx.append(j)
                if not d_scalar:
                    idx.append(k)
                return xi[tuple(idx)]

        def get_kappa(i):
            return kappa[0] if k_scalar else kappa[i]

        def get_kappastar(i):
            return kappastar[0] if kstar_scalar else kappastar[i]

        shape = (nk_eff, nl, nd)

        mse = np.empty(shape, dtype=object)
        loss = np.empty(shape, dtype=object)
        alpha = np.empty(shape, dtype=object)
        q = np.empty(shape, dtype=object)
        rank = np.empty(shape, dtype=object)
        xi_new = np.empty(shape, dtype=object)

        total = np.prod(shape)
        count = 0

        for i, j, k in np.ndindex(shape):
            count += 1

            kappa_i = get_kappa(i)
            kappastar_i = get_kappastar(i)
            xi_ = get_xi(i, j, k)
            noise = noisy[j, k]

            print(f"[{count}/{total}] kappa={kappa_i}, kappastar={kappastar_i}, lam={lam[j]}, Delta={Delta[k]}")
            mse[i, j, k], loss[i, j, k], alpha[i, j, k], q[i, j, k], xi_new[i, j, k], rank[i, j, k] = simulate(kappa_i, kappastar_i, xi_, lam[j], Delta[k], noise)

        self.mse = mse.squeeze()
        self.loss = loss.squeeze()
        self.alpha = alpha.squeeze()
        self.q = q.squeeze()
        self.xi_new = xi_new.squeeze()
        self.rank = rank.squeeze()


    def save(self):
        """Saves the simulator in a pickle file. Updates the log with parameters"""
        path = 'HDSimulators/SimHD_{}.pickle'.format(self.id)
        print("Saving Simulator...")
        print("ID = {}".format(self.id))
        with open(path, 'wb') as file:
            pickle.dump(self, file)

