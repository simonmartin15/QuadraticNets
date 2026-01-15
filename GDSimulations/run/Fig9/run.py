import model
import argparse
import numpy as np
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, required=True)
args = parser.parse_args()


def main():
    d = 100
    kappa = 0.7
    kappastar = 0.5

    alphaPR = utils.PR_threshold(kappa, kappastar)
    a = np.linspace(0.01, alphaPR + 0.1, 30)
    alpha = [a[0:13], a[13:23], a[23:30]]
    rep = 10

    eta = 5e-3
    Tmax = 10000
    lam = 0.
    delta = 0.
    gamma = [1., 0.5, 0.1, 0.01]

    num_save = 10

    grid_shape = (len(gamma), len(alpha))
    idx = args.idx
    i, j = np.unravel_index(idx, grid_shape)

    ID = 'Fig9_{0}{1}'.format(i, j)

    Sim = model.Simulator(d, kappa, kappastar, alpha[j], rep, eta, Tmax, lam, gamma[i],
                          delta, num_save, saveZ=False, saveLabels=False, ID=ID)

    Sim.simulate()
    Sim.save()


if __name__ == '__main__':
    main()
