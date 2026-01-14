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

    a = np.linspace(0.01, 0.8, 40)
    alpha = [a[0:15], a[15:25], a[25:30], a[30:35], a[35:40]]
    rep = 20

    eta = 5e-3
    Tmax = 2000
    lam = 0.
    delta = 0.5
    gamma = [1., 0.1, 0.01]

    num_save = 10

    grid_shape = (len(gamma), len(alpha))
    idx = args.idx
    i, j = np.unravel_index(idx, grid_shape)

    ID = 'Fig10_{0}{1}'.format(i, j)

    Sim = model.Simulator(d, kappa, kappastar, alpha[j], rep, eta, Tmax, lam, gamma[i],
                          delta, num_save, saveZ=False, saveLabels=False, ID=ID)

    Sim.simulate()
    Sim.save()


if __name__ == '__main__':
    main()
