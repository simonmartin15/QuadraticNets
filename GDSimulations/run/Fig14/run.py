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

    a = [0.02, 0.1, 0.2, 0.23, 0.25, 0.28, 0.31, 
         0.34, 0.36, 0.39, 0.5, 0.6, 0.7, 0.8]
    alpha = [a[:9], a[9:14]]
    rep = 10

    eta = 5e-3
    Tmax = 5000
    lam = 0.01
    delta = 0.
    gamma = 1.

    num_save = 10000

    i = args.idx

    ID = 'Fig14_{0}'.format(i)

    Sim = model.Simulator(d, kappa, kappastar, alpha[i], rep, eta, Tmax, lam, gamma,
                          delta, num_save, saveZ=False, saveLabels=False, ID=ID)

    Sim.simulate()
    Sim.save()


if __name__ == '__main__':
    main()
