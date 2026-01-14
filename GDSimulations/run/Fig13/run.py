import modelOJA
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, required=True)
args = parser.parse_args()

# Number of runs = 5


def main():
    d = [200, 400, 600, 800, 1000]
    kappa = [0.8, 0.2]
    rep = 20
    eta = 5e-3
    Tmax = 5000
    gamma = 1.

    num_save = 20000

    i = args.idx

    ID = 'Fig13_{0}'.format(i)

    Sim = modelOJA.Simulator(d[i], kappa, rep, eta, Tmax, gamma, num_save, ID)
    Sim.simulate()
    Sim.save()


if __name__ == '__main__':
    main()
