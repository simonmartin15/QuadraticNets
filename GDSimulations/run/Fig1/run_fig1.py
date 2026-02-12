import sys
import os

# Add project_root/code to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
code_dir = os.path.join(project_root, "code")
sys.path.insert(0, code_dir)


import model

# Number of runs = 1


def main():
    d = 150
    kappa = 0.4
    kappastar = 0.3
    gamma = 1.
    lam = 0.
    delta = 0.

    alpha = [0.25, 0.4]
    rep = 10

    eta = 5e-3
    Tmax = 200

    ID = 'Fig1'

    Sim = model.Simulator(d, kappa, kappastar, alpha, rep, eta, Tmax, lam, gamma,
                          delta, num_save=0, saveZ=False, saveLabels=True, ID=ID)
    Sim.simulate()
    Sim.save()


if __name__ == '__main__':
    main()
