import numpy as np
import torch
import matplotlib.colors as mcolors


class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()


def darken_color(hex_color, amount=0.15):
    rgb = mcolors.to_rgb(hex_color)
    darkened = [max(0, c * (1 - amount)) for c in rgb]
    return mcolors.to_hex(darkened)


def print_to_log(filename, text):
    with open(filename, "a") as f:
        print(text, file=f)


def concatenate(a):
    if len(a) == 0:
        return np.array([])
    else:
        return np.concatenate(a)


def cat(a):
    if len(a) == 0:
        return torch.tensor([])
    else:
        return torch.cat(a)


def stack(inputs):
    if not inputs:
        return torch.empty(0)
    return torch.stack(inputs)


def derivative(x, y):
    yprime = []
    xnew = x[:-1]
    for i in range(len(x) - 1):
        yprime.append((y[i+1] - y[i]) / (x[i+1] - x[i]))
    return xnew, yprime


def horizontal_intersections(x, y, y0):
    dy = y - y0
    idx = np.where(np.diff(np.sign(dy)) != 0)[0]
    x_intersections = []
    for i in idx:
        x1, x2 = x[i], x[i+1]
        y1, y2 = y[i], y[i+1]
        xi = x1 + (y0 - y1) * (x2 - x1) / (y2 - y1)
        x_intersections.append(xi)

    return np.array(x_intersections)


def alpha_list(alphaPR, Nout, Nin):
    if alphaPR is None:
        return torch.linspace(0.01, 0.55, Nin+Nout)
    bounds = (max(alphaPR - 0.05, 0), min(alphaPR + 0.05, 0.55))
    length = 0.55 - bounds[1] + bounds[0]
    num_left = np.floor(Nout * bounds[0] / length).astype(np.int32)
    num_right = np.ceil(Nout * (0.55 - bounds[1]) / length).astype(np.int32)
    left = torch.tensor(np.linspace(0.01, bounds[0], num_left, endpoint=False))
    right = torch.tensor(np.linspace(bounds[1], 0.55, num_right))
    mid = torch.tensor(np.linspace(bounds[0], bounds[1], round(Nin), endpoint=False))
    return torch.cat((left, mid, right))


def log_steps(start, stop, num):
    x = (torch.logspace(0, np.log10(stop - start), num) + start - 1).int()
    return torch.unique(x)


def generate_Wishart(s):
    """s = tuple of the form (..., d, m). Returns a Wishart matrix with parameter m of shape (..., d, d)"""
    W = torch.randn(s) / np.sqrt(s[-1])
    return W @ W.mT


def generate_GOE(s):
    """s = tuple of the form (..., d). Returns a GOE matrix with dimension d of shape (..., d, d)"""
    if type(s) is int:
        H = torch.randn(size=(s, s)) / np.sqrt(2*s)
    else:
        d = s[-1]
        s_new = tuple(list(s) + [d])
        H = torch.randn(size=s_new) / np.sqrt(2*d)
    return H + H.mT


def generate_Hermite2(s):
    if type(s) is int:
        x = torch.randn(s)
        X = torch.ger(x, x)
        return (X - torch.eye(s)) / np.sqrt(s)
    else:
        *batch_shape, d = s
        x = torch.randn(*batch_shape, d)
        X = x[..., :, None] * x[..., None, :]
        return (X - torch.eye(d)) / np.sqrt(d)


def compute_MSE(A, B):
    assert A.shape[-1] == A.shape[-2] == B.shape[-1] == B.shape[-2]
    dim = A.shape[-1]
    return torch.sum((A - B)**2, dim=(-1, -2)) / dim


def compute_overlap(A, B):
    assert A.shape[-1] == A.shape[-2] == B.shape[-1] == B.shape[-2]
    dim = A.shape[-1]
    return torch.sum(A * B, dim=(-1, -2)) / dim


def compute_prediction(A, m):
    """A: tensor with shape (..., d, d)"""
    assert A.shape[-1] == A.shape[-2]
    d = A.shape[-1]
    L, Q = torch.linalg.eigh(A)
    f = torch.cat((torch.zeros(d-m), torch.ones(m))) if m < d else torch.ones(d)
    E = f * torch.clamp(L, min=0)
    return Q @ torch.diag_embed(E) @ Q.mT


def compute_eigenvalues_prediction(A, m):
    """A: tensor with shape (..., d, d)"""
    assert A.shape[-1] == A.shape[-2]
    d = A.shape[-1]
    L, Q = torch.linalg.eigh(A)
    f = torch.cat((torch.zeros(d-m), torch.ones(m))) if m < d else torch.ones(d)
    return f * torch.clamp(L, min=0)


def PR_threshold(kappa, kappastar):
    if kappa < min(1, kappastar):
        raise ValueError('No PR threshold for kappa < kappa*')
    elif kappastar >= 1:
        return 1/2
    elif 2 * kappa > 1 + kappastar:
        return (1/2 + kappastar - kappastar**2 / 2) / 2
    else:
        y = ((1 + kappastar) / 2 - kappa) / (1 - kappastar)
        v = np.linspace(0, 1, 1000)
        ind = np.argmin((y - fPR(v))**2)
        a = (kappastar + kappa * (1 - kappastar)) / 2 + (1 - kappastar)**2 / np.pi * v[ind] * (1 - v[ind]**2)**(3/2)
        return min(a, kappa - kappa**2 / 2)


def alpha_star(kappa, kappastar):
    if kappa < kappastar:
        raise ValueError('No PR threshold for kappa < kappa*')
    elif kappastar >= 1:
        return 1/2
    elif 2 * kappa > 1 + kappastar:
        return (1/2 + kappastar - kappastar**2 / 2) / 2
    else:
        y = ((1 + kappastar) / 2 - kappa) / (1 - kappastar)
        v = np.linspace(0, 1, 1000)
        ind = np.argmin((y - fPR(v))**2)
        a = (kappastar + kappa * (1 - kappastar)) / 2 + (1 - kappastar)**2 / np.pi * v[ind] * (1 - v[ind]**2)**(3/2)
        return a


def PR_region(kappa, kappastar):
    if kappa < kappastar:
        raise ValueError('No PR threshold for kappa < kappa*')
    if kappastar >= 1:
        return 'III'
    elif 2 * kappa > 1 + kappastar:
        return 'III'
    elif kappa == kappastar:
        return 'I'
    else:
        y = ((1 + kappastar) / 2 - kappa) / (1 - kappastar)
        v = np.linspace(0, 1, 1000)
        ind = np.argmin((y - fPR(v)) ** 2)
        a = (kappastar + kappa * (1 - kappastar)) / 2 + (1 - kappastar) ** 2 / np.pi * v[ind] * (1 - v[ind] ** 2) ** (3 / 2)
        if a > kappa - kappa**2 / 2:
            return 'I'
        else:
            return 'II'


def strong_threshold(kappa, kappastar, return_q = False):
    if kappa < min(1, kappastar):
        raise ValueError('No PR threshold for kappa < kappa*')
    elif kappastar >= 1:
        return 1/2
    else:
        v = np.linspace(0, 2, 1000)
        y = (min(1, kappa) - kappastar) / (1 - kappastar)
        ind = np.argmin((y - moment_0(v)) ** 2)
        omega = v[ind]

        q = np.linspace(0.001, 2, 1000)
        b = np.minimum(np.maximum(q, omega), 2)
        F = np.array([moment_1(b[i]) - q[i] * moment_0(b[i]) for i in range(len(q))]) / q
        ind = np.argmin((F - kappastar / (1 - kappastar)) ** 2)
        q_opt = q[ind]
        b_opt = b[ind]

        alphaPR = kappastar - kappastar ** 2 / 2 + (1 - kappastar) ** 2 / 2 * (moment_2(b_opt) - q_opt * moment_1(b_opt))

        if not return_q:
            return alphaPR
        return alphaPR, q_opt


def fPR(v):
    return (v * np.sqrt(1-v**2) + np.arcsin(v)) / np.pi


def moment_0(x):
    return 1/2 - x * np.sqrt(4-x**2) / (4 * np.pi) - np.arcsin(x/2) / np.pi


def moment_1(x):
    return (4-x**2)**(3/2) / (6 * np.pi)


def moment_2(x):
    return 1/2 - np.sqrt(4-x**2) * (x**3 - 2*x) / (8*np.pi) - np.arcsin(x/2) / np.pi


def sigma(x):
    if np.abs(x) <= 2:
        return np.sqrt(4-x**2) / (2 * np.pi)
    return 0


def rho_MP(x, kappastar):
    r1 = (1 - 1 / np.sqrt(kappastar))**2
    r2 = (1 + 1 / np.sqrt(kappastar))**2
    return np.sqrt((r2 - x) * (x - r1)) / (2 * np.pi * x)