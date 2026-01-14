from time import time
import pickle
from utils import *
import numpy as np
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Simulator:
    def __init__(self, d, k, rep, eta, Tmax, gamma, num_save, ID):
        """
        d (int): input dimension
        k, kstar (int): rescaled number of neurons of student and teacher
        alpha (1D array): rescaled number of observations
        rep (int): number of repetitions for averaging
        eta (float): GD stepsize
        Tmax (float): max time, gives number of steps as nstep = int(Tmax / eta)
        lam (float): intensity of regularization
        gamma (float): variance of the initialization
        delta (float): variance of the label noise
        temp (float): variance of the Langevin noise
        num_save (int): number of points saved
        ID (str): Id of the simulation
        """

        torch.manual_seed(0)

        self.d = d
        self.k = k
        self.Nk = len(self.k)

        self.gamma = gamma
        self.eta = eta
        self.Tmax = Tmax

        self.num_save = num_save
        self.ID = ID

        self.m = [int(self.d * kappa) for kappa in self.k]
        self.rep = rep

        self.Teacher = generate_GOE((self.Nk, self.rep, self.d))
        self.Zinf = torch.stack([compute_prediction(self.Teacher[i], self.m[i]) for i in range(self.Nk)])

        print("")
        print("")
        print("Simulator {}".format(self.ID))
        print("Parameters:")
        print("Dimension: {}".format(self.d))
        print("kappa: {}".format(self.k))
        print("gamma: {}".format(self.gamma))
        print("reps: {}".format(self.rep))
        print("eta: {}".format(self.eta))
        print("Tmax: {}".format(self.Tmax))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Device: {}'.format(self.device))
        print("")

        self.nstep = int(self.Tmax / self.eta)
        self.steps_save = log_steps(0, self.nstep, self.num_save)

        self.MSE = None
        self.Dist = None

    def simulate_one(self, m, Teacher, Lim):
        t0 = time()
        W = (np.sqrt(self.gamma) * torch.randn(size=(self.rep, self.d, m)) / np.sqrt(m)).to(self.device)
        T = Teacher.to(self.device)
        L = Lim.to(self.device)

        MSE = torch.zeros(size=(len(self.steps_save), self.rep)).to(self.device)
        Dist = torch.zeros(size=(len(self.steps_save), self.rep)).to(self.device)

        count_save = 0

        for j in range(self.nstep):
            print('\rStep[{0}/{1}], Distance = {2}'.format(j+1, self.nstep, torch.mean(Dist[count_save-1]).item()), end="")
            Z = W @ W.mT

            if j in self.steps_save:
                MSE[count_save] = torch.sum((Z - T)**2, dim=(1, 2)) / self.d
                Dist[count_save] = torch.sum((Z - L)**2, dim=(1, 2)) / self.d
                count_save += 1

            W += - self.eta * (Z - T) @ W

        runtime = time() - t0
        print('Run Finished in {0} sec, MSE = {1}, Dist = {2}'.format(round(runtime, 2), torch.mean(MSE[-1]), torch.mean(Dist[-1])))
        print("")

        return MSE.to('cpu'), Dist.to('cpu')

    def simulate(self):
        t0 = time()
        MSE = torch.zeros(self.Nk, len(self.steps_save), self.rep)
        Dist = torch.zeros(self.Nk, len(self.steps_save), self.rep)
        
        for i in range(self.Nk):
            print('Run [{0}/{1}], kappa = {2}'.format(i+1, self.Nk, self.k[i]))
            print("")

            with ClearCache():
                MSE[i], Dist[i] = self.simulate_one(self.m[i], self.Teacher[i], self.Zinf[i])

        self.MSE = MSE
        self.Dist = Dist

        t1 = time()

        print('Total time GD: {}'.format(round(t1-t0, 2)))
        print("")

    def save(self):
        """Saves the simulator in a pickle file. Updates the log with parameters"""
        path = 'Simulators/SimGD_{}.pickle'.format(self.ID)
        print("Saving Simulator...")
        print("ID = {}".format(self.ID))
        with open(path, 'wb') as file:
            pickle.dump(self, file)








