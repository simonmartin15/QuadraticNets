import model
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, required=True)
args = parser.parse_args()

# Number of runs = 2


def main():
    d = 100
    kappa = 0.4
    kappastar = 0.3

    alpha = [0.08, 0.13, 0.2, 0.23, 0.26, 0.29,
             0.32, 0.35, 0.38, 0.44, 0.5]
    rep = 10

    eta = 5e-3
    Tmax = 5000
    lam = 0.01
    delta = [0.25, 0.5]
    gamma = 1.

    num_save = 10000

    i = args.idx

    ID = 'Fig5_{}'.format(i)

    Sim = model.Simulator(d, kappa, kappastar, alpha, rep, eta, Tmax, lam, gamma,
                          delta[i], num_save, saveZ=False, saveLabels=False, ID=ID)

    Sim.simulate()
    Sim.save()


if __name__ == '__main__':
    main()
