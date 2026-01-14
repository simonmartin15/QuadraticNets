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
