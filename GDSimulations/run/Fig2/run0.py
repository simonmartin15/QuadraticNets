import model
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, required=True)
args = parser.parse_args()

# Number of runs = 15


def main():
    d = 100
    kappa = 0.7
    kappastar = 0.5

    a = np.linspace(0.01, 0.6, 30)
    alpha = [a[0:6], a[6:12], a[12:18], a[18:24], a[24:30]]
    rep = 10

    eta = 5e-3
    Tmax = [1000, 1000, 2000]
    lam = [0.5, 0.1, 0.05]
    delta = 0.
    gamma = 1.

    num_save = 10

    grid_shape = (len(lam), len(alpha))
    idx = args.idx
    i, j = np.unravel_index(idx, grid_shape)

    ID = 'Fig2_0{0}{1}'.format(i, j)

    Sim = model.Simulator(d, kappa, kappastar, alpha[j], rep, eta, Tmax[i], lam[i], gamma,
                          delta, num_save, saveZ=False, saveLabels=False, ID=ID)

    Sim.simulate()
    Sim.save()


if __name__ == '__main__':
    main()
