import sys
import os
import numpy as np
import argparse

# Add project_root/code to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
code_dir = os.path.join(project_root, "code")
sys.path.insert(0, code_dir)

import model

parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, required=True)
args = parser.parse_args()

# Number of runs = 80


def main():
    d = 100
    kappa = [0.7, 0.5, 0.4, 0.35]
    kappastar = 0.3

    a = np.linspace(0.01, 0.6, 20)
    alpha = [a[0:8], a[8:13], a[13:17], a[17:20]]
    rep = 10
    eta = 5e-3
    Tmax = 10000
    lam = [0.1, 0.05, 0.025, 0.005, 0.002]
    delta = 1.0
    gamma = 1.

    num_save = 10

    grid_shape = (len(kappa), len(lam), len(alpha))
    idx = args.idx
    i, j, k = np.unravel_index(idx, grid_shape)

    ID = 'Fig19_{0}{1}{2}'.format(i, j, k)

    Sim = model.Simulator(d, kappa[i], kappastar, alpha[k], rep, eta, Tmax, lam[j], gamma,
                          delta, num_save, saveZ=False, saveLabels=False, ID=ID)

    Sim.simulate()
    Sim.save()


if __name__ == '__main__':
    main()
