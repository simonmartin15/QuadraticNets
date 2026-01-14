import model
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, required=True)
args = parser.parse_args()

# Number of runs = 3


def main():
    d = 100
    kappa = 0.7
    kappastar = 0.5

    alpha = [0.25, 0.4]
    rep = 50

    eta = 5e-3
    Tmax = 1000
    lam = [0.5, 0.1, 0.05]
    delta = 0.
    gamma = 1.

    i = args.idx

    ID = 'Fig2_1{}'.format(i)

    Sim = model.Simulator(d, kappa, kappastar, alpha, rep, eta, Tmax, lam[i], gamma,
                          delta, num_save=0, saveZ=True, saveLabels=False, ID=ID)

    Sim.simulate()
    Sim.save()


if __name__ == '__main__':
    main()
