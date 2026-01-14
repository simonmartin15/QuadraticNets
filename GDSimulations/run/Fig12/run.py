import model
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, required=True)
args = parser.parse_args()

# Number of runs = 8


def main():
    d = 150
    kappa = 0.5
    kappastar = 0.2

    a = [0.11, 0.14, 0.17, 0.21, 0.26, 0.32, 0.37, 0.42, 0.47]
    alpha = [a[0:6], a[6:9]]
    rep = 10

    eta = 5e-3
    Tmax = 1000
    lam = [0.1, 0.0]
    delta = 0.
    gamma = 1.0

    data = ['hermite', 'gaussian']
    num_save = 5000

    grid_shape = (len(data), len(lam), len(alpha))
    idx = args.idx
    i, j, k = np.unravel_index(idx, grid_shape)

    ID = 'Fig12_{0}{1}{2}'.format(i, j, k)

    Sim = model.Simulator(d, kappa, kappastar, alpha[k], rep, eta, Tmax, lam[j], gamma,
                          delta, num_save, data[i], saveZ=False, saveLabels=False, ID=ID)

    if Sim.device == 'cpu':
        raise ValueError('Error: cpu')

    Sim.simulate()
    Sim.save()


if __name__ == '__main__':
    main()
