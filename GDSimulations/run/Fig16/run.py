import model
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, required=True)
args = parser.parse_args()

# Number of runs = 36


def main():
    d = 100
    kappa = [0.4, 0.35, 0.32, 0.3]
    kappastar = 0.3

    a = np.linspace(0.01, 0.6, 20)
    alpha = [a[0:10], a[10:15], a[15:20]]
    rep = 10

    eta = 5e-3
    Tmax = 5000 
    lam = [0.1, 0.05, 0.01]
    delta = 0.
    gamma = 1.

    num_save = 10

    grid_shape = (len(kappa), len(lam), len(alpha))
    idx = args.idx
    i, j, k = np.unravel_index(idx, grid_shape)

    ID = 'Fig16_{0}{1}{2}'.format(i, j, k)

    Sim = model.Simulator(d, kappa[i], kappastar, alpha[k], rep, eta, Tmax, lam[j], gamma,
                          delta, num_save, saveZ=False, saveLabels=False, ID=ID)

    Sim.simulate()
    Sim.save()


if __name__ == '__main__':
    main()
