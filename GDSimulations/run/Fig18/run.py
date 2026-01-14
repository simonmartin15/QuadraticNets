import model
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, required=True)
args = parser.parse_args()

# Number of runs = 105


def main():
    d = 100
    kappa = [0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7]
    kappastar = [0.2, 0.3, 0.5, 0.3, 0.5, 0.6, 0.7]

    a = np.linspace(0.01, 0.6, 30)
    alpha = [a[0:6], a[6:12], a[12:18], a[18:24], a[24:30]]
    rep = 10

    eta = 5e-3
    Tmax = [1000, 1000, 2000]
    lam = [0.5, 0.1, 0.05]
    delta = 0.5
    gamma = 1.

    num_save = 10

    grid_shape = (len(kappa), len(lam), len(alpha))
    idx = args.idx
    i, j, k = np.unravel_index(idx, grid_shape)

    ID = 'Fig18_{0}{1}{2}'.format(i, j, k)

    Sim = model.Simulator(d, kappa[i], kappastar[i], alpha[k], rep, eta, Tmax, lam[j], gamma,
                          delta, num_save, saveZ=False, saveLabels=False, ID=ID)

    Sim.simulate()
    Sim.save()


if __name__ == '__main__':
    main()
