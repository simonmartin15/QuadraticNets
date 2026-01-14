import model
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, required=True)
args = parser.parse_args()

# Number of runs = 3


def main():
    d = 100
    kappa = 0.4
    kappastar = 0.3

    alpha = np.linspace(0.01, 0.6, 20)
    rep = 10

    eta = 5e-3
    Tmax = 5000
    lam = 0.01
    delta = [0.1, 0.25, 0.5]
    gamma = 1.

    num_save = 10

    i = args.idx

    ID = 'Fig6_0{}'.format(i)

    Sim = model.Simulator(d, kappa, kappastar, alpha, rep, eta, Tmax, lam, gamma,
                          delta[i], num_save, saveZ=False, saveLabels=False, ID=ID)

    Sim.simulate()
    Sim.save()


if __name__ == '__main__':
    main()
