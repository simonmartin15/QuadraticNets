import model
import argparse
import numpy as np
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, required=True)
args = parser.parse_args()

# Number of runs = 150


def main():
    d = 100
    kappa = [0.2, 0.3, 0.3, 0.36, 0.4, 0.45, 0.5, 0.5,
             0.5, 0.6, 0.6, 0.7, 0.7, 0.7, 0.7, 0.7,
             0.75, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    kappastar = [0.1, 0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.3,
                 0.5, 0.3, 0.5, 0.1, 0.2, 0.3, 0.5, 0.7,
                 0.7, 0.3, 0.3, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    num_alpha = 6
    grid_shape = (len(kappa), num_alpha)
    idx = args.idx
    i, j = np.unravel_index(idx, grid_shape)

    alphaPR = utils.PR_threshold(kappa[i], kappastar[i])
    a = utils.alpha_list(alphaPR, 20, 20)
    alpha = [a[0:10], a[10:18], a[18:25], a[25:31], a[31:36], a[36:40]]
    rep = 10

    eta = 5e-3
    Tmax = 10000
    lam = 0.
    delta = 0.
    gamma = 1.

    num_save = 10

    ID = 'Fig20_{0}{1}'.format(i, j)

    Sim = model.Simulator(d, kappa[i], kappastar[i], alpha[j], rep, eta, Tmax, lam, gamma,
                          delta, num_save, saveZ=False, saveLabels=False, ID=ID)

    Sim.simulate()
    Sim.save()


if __name__ == '__main__':
    main()
