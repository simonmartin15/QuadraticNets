from time import time
import pickle
from utils import *
import numpy as np
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Simulator:
    def __init__(self, d, k, kstar, alpha, rep, eta, Tmax, lam, gamma, delta, temp, num_save, ID):
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
        num_save (int): number of points safed
        ID (str): Id of the simulation
        """

        torch.manual_seed(0)

        self.d = d
        self.k = k
        self.kstar = kstar
        self.alpha = np.array(alpha)
        self.gamma = gamma
        self.delta = delta
        self.lam = lam
        self.eta = eta
        self.Tmax = Tmax
        self.lam = lam
        self.gamma = gamma
        self.temp = temp

        self.num_save = num_save
        self.ID = ID

        self.log_file = "logs/logs_{}.txt".format(self.ID)

        self.Nn = len(self.alpha)

        self.m = round(self.d * self.k)
        self.ms = round(self.d * self.kstar)
        self.n = (self.d ** 2 * self.alpha).astype(np.int32)
        self.rep = rep

        self.Teacher = generate_Wishart((self.Nn, self.rep, self.d, self.ms))

        self.print_to_log("")
        self.print_to_log("")
        self.print_to_log("Simulator {}".format(self.ID))
        self.print_to_log("Parameters:")
        self.print_to_log("Dimension: {}".format(self.d))
        self.print_to_log("kappa: {}".format(self.k))
        self.print_to_log("kappa*: {}".format(self.kstar))
        self.print_to_log("alpha: {}".format(self.alpha))
        self.print_to_log("lambda: {}".format(self.lam))
        self.print_to_log("delta: {}".format(self.delta))
        self.print_to_log("gamma: {}".format(self.gamma))
        self.print_to_log("temperature: {}".format(self.temp))
        self.print_to_log("reps: {}".format(self.rep))
        self.print_to_log("eta: {}".format(self.eta))
        self.print_to_log("Tmax: {}".format(self.Tmax))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.print_to_log('Device: {}'.format(self.device))
        self.print_to_log("")

        self.nstep = int(self.Tmax / self.eta)
        self.steps_save = log_steps(0, self.nstep, self.num_save)

        self.ZOverlap = None
        self.yOverlap = None

        self.Z = None
        self.labels = None
        self.labels_Teacher = None

        self.MSE = None
        self.loss = None


    def print_to_log(self, text):
        return print_to_log(self.log_file, text)

    def simulate_one(self, n, Teacher):
        t0 = time()
        W = (np.sqrt(self.gamma) * torch.randn(size=(self.rep, self.d, self.m)) / np.sqrt(self.m)).to(self.device)
        X = generate_GOE((self.rep, n, self.d)).to(self.device)
        T = Teacher.to(self.device)
        Labels_Teacher = torch.einsum('rij,rpij->rp', T, X)

        xi = torch.randn((self.rep, n)).to(self.device)
        Labels_grad = Labels_Teacher + np.sqrt(self.delta) * xi

        grad_teacher = torch.einsum('rp, rpij->rij', Labels_grad, X)

        Z = W @ W.mT

        MSE = torch.zeros(size=(len(self.steps_save), self.rep)).to(self.device)
        Loss = torch.zeros(size=(len(self.steps_save), self.rep)).to(self.device)
        ZOverlap = torch.zeros(size=(len(self.steps_save), self.rep)).to(self.device)
        yOverlap = torch.zeros(size=(len(self.steps_save), self.rep)).to(self.device)

        count_save = 0

        for j in range(self.nstep):
            print('\rStep[{0}/{1}], MSE = {2}'.format(j+1, self.nstep, torch.mean(MSE[count_save-1]).item()), end="")
            Z = W @ W.mT

            if j in self.steps_save:
                Labels = torch.sum(X * Z[:, None], dim=(-1, -2))
                L = Labels - Labels_Teacher
                MSE[count_save] = torch.sum((Z - T)**2, dim=(1, 2)) / self.d
                Loss[count_save] = torch.sum(L**2, dim=1) / (4 * n)
                ZOverlap[count_save] = torch.sum(Z * T, dim=(1, 2)) / self.d
                yOverlap[count_save] = torch.sum(Labels * Labels_Teacher, dim=1) / n
                count_save += 1

            # Optimizer step
            grad = (torch.einsum('rkl,rpkl,rpij->rij', Z, X, X) - grad_teacher) @ W * self.d / n
            grad_noise = 0 if self.temp == 0 else np.sqrt(self.eta * self.temp / self.d) * torch.randn(self.rep, self.d, self.m).to(self.device)

            W += - self.eta * (grad + 2 * self.lam * W) + grad_noise

        Zsave = Z
        Lsave = torch.sum(X * Z[:, None], dim=(-1, -2))

        runtime = time() - t0
        self.print_to_log('Run Finished in {0} sec, MSE = {1}, Loss = {2}'.format(round(runtime, 2), torch.mean(MSE[-1]), torch.mean(Loss[-1])))
        self.print_to_log("")

        return MSE.to('cpu'), Loss.to('cpu'), ZOverlap.to('cpu'), yOverlap.to('cpu'), Zsave.to('cpu'), Lsave.to('cpu'), Labels_Teacher.to('cpu')

    def simulate(self):
        t0 = time()
        MSE = torch.zeros(self.Nn, len(self.steps_save), self.rep)
        Loss = torch.zeros(self.Nn, len(self.steps_save), self.rep)
        ZOverlap = torch.zeros(self.Nn, len(self.steps_save), self.rep)
        yOverlap = torch.zeros(self.Nn, len(self.steps_save), self.rep)
        Z = [0 for _ in range(self.Nn)]
        labels = [0 for _ in range(self.Nn)]
        labels_teacher = [0 for _ in range(self.Nn)]

        for i in range(self.Nn):
            self.print_to_log('Run [{0}/{1}], alpha = {2}'.format(i+1, self.Nn, np.round(self.alpha[i], 6)))
            print('Run [{0}/{1}], alpha = {2}'.format(i+1, self.Nn, np.round(self.alpha[i], 6)))
            print("")
            with ClearCache():
                MSE[i], Loss[i], ZOverlap[i], yOverlap[i], Z[i], labels[i], labels_teacher[i] = self.simulate_one(self.n[i], self.Teacher[i])

        self.MSE = MSE
        self.loss = Loss
        self.ZOverlap = ZOverlap
        self.yOverlap = yOverlap
        self.Z = Z
        self.labels = labels
        self.labels_Teacher = labels_teacher

        t1 = time()
        self.print_to_log('Total time GD: {}'.format(round(t1-t0, 2)))
        self.print_to_log("")

    def save(self):
        """Saves the simulator in a pickle file. Updates the log with parameters"""
        path = 'Simulators/SimGD_{}.pickle'.format(self.ID)
        self.print_to_log("Saving Simulator...")
        self.print_to_log("ID = {}".format(self.ID))
        with open(path, 'wb') as file:
            pickle.dump(self, file)








