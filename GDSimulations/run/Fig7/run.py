import model
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, required=True)
args = parser.parse_args()

# Number of runs = 4


def main():
    d = 100
    kappa = 0.3
    kappastar = 0.2

    alpha = np.linspace(0.01, 0.35, 30)
    rep = 10

    eta = 5e-3
    Tmax = 3000
    lam = [0.1, 0.05, 0.01, 0.005]
    delta = 0.
    gamma = 1.

    num_save = 10

    i = args.idx

    ID = 'Fig7_{}'.format(i)

    Sim = model.Simulator(d, kappa, kappastar, alpha, rep, eta, Tmax, lam[i], gamma,
                          delta, num_save, saveZ=False, saveLabels=False, ID=ID)

    Sim.simulate()
    Sim.save()


if __name__ == '__main__':
    main()
