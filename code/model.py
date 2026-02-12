"""
Gradient flow simulator for quadratic neural networks.

This file implements numerical simulations of gradient descent dynamics
used to compare with the theoretical high-dimensional predictions.
"""

from time import time
import pickle
from utils import *
import numpy as np
import torch

import os

# Allow duplicate OpenMP libraries (avoids MKL / OpenMP runtime errors on some systems)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Simulator:
    """Simulator for gradient descent dynamics for quadratic neural networks.
    Runs gradient descent for a range of sample sizes (alpha),
    averages over repetitions, and records MSE, training loss and in-sample error along the dynamics."""
    def __init__(self, d, k, kstar, alpha, rep, eta, Tmax, lam, gamma, delta, num_save, saveZ, saveLabels, ID, data='gaussian'):
        """
        d (int): input dimension
        k, kstar (int): rescaled number of neurons of student and teacher
        alpha (1D array): rescaled number of observations
        rep (int): number of repetitions for averaging
        eta (float): stepsize
        Tmax (float): maximum time (number of steps = int(Tmax / eta))
        lam (float): intensity of regularization
        gamma (float): variance of the initialization
        delta (float): variance of the label noise
        num_save (int): number of saved time points
        saveZ (bool): save the limit of gradient flow
        saveLabels (bool): save labels
        ID (str): identifier of the simulation
        data (str): how to generate the observations, either gaussian (GOE) or hermite (centered quadratic activation)
            Default is gaussian
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

        self.data = data

        self.saveZ = saveZ
        self.saveLabels = saveLabels

        self.num_save = num_save
        self.ID = ID

        self.Nn = len(self.alpha)

        self.m = round(self.d * self.k)
        self.ms = round(self.d * self.kstar)
        self.n = (self.d ** 2 * self.alpha).astype(np.int32)
        self.rep = rep

        self.Teacher = generate_Wishart((self.Nn, self.rep, self.d, self.ms))

        print()
        print()
        print("Simulator {}".format(self.ID))
        print("Parameters:")
        print("Dimension: {}".format(self.d))
        print("kappa: {}".format(self.k))
        print("kappa*: {}".format(self.kstar))
        print("alpha: {}".format(self.alpha))
        print("lambda: {}".format(self.lam))
        print("delta: {}".format(self.delta))
        print("gamma: {}".format(self.gamma))
        print("reps: {}".format(self.rep))
        print("eta: {}".format(self.eta))
        print("Tmax: {}".format(self.Tmax))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Device: {}'.format(self.device))
        print()

        self.nstep = int(self.Tmax / self.eta)
        self.steps_save = log_steps(0, self.nstep, self.num_save)

        if self.saveLabels:
            self.steps_save_labels = log_steps(0, self.nstep, 40)

        self.Z = None
        self.labels = None

        self.MSE = None
        self.loss = None
        self.loss_full = None
        self.in_sample_error = None


    def simulate_one(self, n, Teacher):
        """Run one gradient descent simulation for a fixed sample size n and a fixed teacher matrix.
        Returns time series of MSE, loss, full loss, in-sample error, and optionally the final Z matrix and saved labels."""
        t0 = time()

        # Initialize student weights
        W = (np.sqrt(self.gamma) * torch.randn(size=(self.rep, self.d, self.m)) / np.sqrt(self.m)).to(self.device)

        # Generate data matrices
        if self.data == 'hermite':
            X = generate_Hermite2((self.rep, n, self.d)).to(self.device)
        elif self.data == 'gaussian':
            X = generate_GOE((self.rep, n, self.d)).to(self.device)
        else:
            raise ValueError('Not implemented')

        T = Teacher.to(self.device)

        # Teacher outputs and noisy labels
        Labels_Teacher = torch.einsum('rij,rpij->rp', T, X)
        xi = torch.randn((self.rep, n)).to(self.device)
        Labels_Noisy = Labels_Teacher + np.sqrt(self.delta) * xi

        grad_teacher = torch.einsum('rp, rpij->rij', Labels_Noisy, X)

        Z = W @ W.mT

        MSE = torch.zeros(size=(len(self.steps_save), self.rep)).to(self.device)
        Loss = torch.zeros(size=(len(self.steps_save), self.rep)).to(self.device)
        LossFull = torch.zeros(size=(len(self.steps_save), self.rep)).to(self.device)
        InSampleError = torch.zeros(size=(len(self.steps_save), self.rep)).to(self.device)

        Lsave = []
        count_save = 0

        # Gradient descent loop
        for j in range(self.nstep):
            print('\rStep[{0}/{1}]'.format(j+1, self.nstep), end="")
            Z = W @ W.mT

            if j in self.steps_save:
                Labels = torch.sum(X * Z[:, None], dim=(-1, -2))
                L = Labels - Labels_Teacher
                LNoise = Labels - Labels_Noisy
                MSE[count_save] = torch.sum((Z - T)**2, dim=(1, 2)) / self.d
                loss = torch.sum(LNoise**2, dim=1) / (4 * n)
                Loss[count_save] = loss
                LossFull[count_save] = loss + self.lam * torch.sum(W ** 2, dim=(1, 2)) / self.d
                InSampleError[count_save] = torch.sum(L**2, dim=1) / (4 * n)
                count_save += 1

            if self.saveLabels and j in self.steps_save_labels:
                Labels = torch.sum(X * Z[:, None], dim=(-1, -2))
                Lsave.append(Labels)

            grad = (torch.einsum('rkl,rpkl,rpij->rij', Z, X, X) - grad_teacher) @ W * self.d / n
            W += - self.eta * (grad + 2 * self.lam * W)

        Zsave = Z.to('cpu') if self.saveZ else None
        Lsave = stack([y.to('cpu') for y in Lsave]) if self.saveLabels else None

        runtime = time() - t0
        print('Run Finished in {} sec'.format(round(runtime, 2)))
        print()

        return MSE.to('cpu'), Loss.to('cpu'), LossFull.to('cpu'), InSampleError.to('cpu'), Zsave, Lsave

    def simulate(self):
        """Run gradient descent simulations for all values of alpha.
        Aggregates results over repetitions and stores them."""
        t0 = time()
        MSE = torch.zeros(self.Nn, len(self.steps_save), self.rep)
        Loss = torch.zeros(self.Nn, len(self.steps_save), self.rep)
        LossFull = torch.zeros(self.Nn, len(self.steps_save), self.rep)
        InSampleError = torch.zeros(self.Nn, len(self.steps_save), self.rep)
        Z = [0 for _ in range(self.Nn)]
        labels = [0 for _ in range(self.Nn)]

        # Loop over sample sizes
        for i in range(self.Nn):
            print('Run [{0}/{1}], alpha = {2}'.format(i+1, self.Nn, np.round(self.alpha[i], 6)))
            print()
            with ClearCache():
                MSE[i], Loss[i], LossFull[i], InSampleError[i], Z[i], labels[i] = self.simulate_one(self.n[i], self.Teacher[i])

        self.MSE = MSE
        self.loss = Loss
        self.loss_full = LossFull
        self.in_sample_error = InSampleError
        self.Z = Z if self.saveZ else None
        self.labels = labels if self.saveLabels else None

        t1 = time()
        print('Total time GD: {}'.format(round(t1-t0, 2)))
        print()

    def save(self):
        """Saves the simulator in a pickle file"""
        path = 'Simulators/SimGD_{}.pickle'.format(self.ID)
        print("Saving Simulator...")
        print("ID = {}".format(self.ID))
        with open(path, 'wb') as file:
            pickle.dump(self, file)


class SimulatorOja:
    """Simulator for discrete Oja flow.
    Runs gradient descent for a rank-m approximation problem, 
    and tracks convergence to the spectral projector of the target matrix."""
    def __init__(self, d, k, rep, eta, Tmax, gamma, num_save, ID):
        """
        d (int): input dimension
        k (1D array): list of rescaled ranks kappa = m / d
        rep (int): number of repetitions for averaging
        eta (float): stepsize
        Tmax (float): maximum time (number of steps = int(Tmax / eta))
        gamma (float): variance of the initialization
        num_save (int): number of saved time points
        ID (str): identifier of the simulation
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

        # Teacher matrices and their rank-m spectral projections
        self.Teacher = generate_GOE((self.Nk, self.rep, self.d))
        self.Zinf = torch.stack([spectral_cutoff(self.Teacher[i], self.m[i]) for i in range(self.Nk)])

        print()
        print()
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
        print()

        self.nstep = int(self.Tmax / self.eta)
        self.steps_save = log_steps(0, self.nstep, self.num_save)

        self.MSE = None
        self.Dist = None

    def simulate_one(self, m, Teacher, Lim):
        """Run one gradient descent simulation for a fixed rank m and teacher matrix.
        Returns time series of MSE and distance to the limit of the dynamics."""
        t0 = time()

        # Initialize weights
        W = (np.sqrt(self.gamma) * torch.randn(size=(self.rep, self.d, m)) / np.sqrt(m)).to(self.device)
        T = Teacher.to(self.device)
        L = Lim.to(self.device)

        MSE = torch.zeros(size=(len(self.steps_save), self.rep)).to(self.device)
        Dist = torch.zeros(size=(len(self.steps_save), self.rep)).to(self.device)

        count_save = 0

        # Gradient descent loop
        for j in range(self.nstep):
            print('\rStep[{0}/{1}]'.format(j + 1, self.nstep), end="")
            Z = W @ W.mT

            if j in self.steps_save:
                MSE[count_save] = torch.sum((Z - T) ** 2, dim=(1, 2)) / self.d
                Dist[count_save] = torch.sum((Z - L) ** 2, dim=(1, 2)) / self.d
                count_save += 1

            W += - self.eta * (Z - T) @ W

        runtime = time() - t0
        print('Run Finished in {} sec'.format(round(runtime, 2)))
        print()

        return MSE.to('cpu'), Dist.to('cpu')

    def simulate(self):
        """Run gradient descent simulations for all values of kappa.
        Aggregates results over repetitions and stores them."""
        t0 = time()
        MSE = torch.zeros(self.Nk, len(self.steps_save), self.rep)
        Dist = torch.zeros(self.Nk, len(self.steps_save), self.rep)

        # Loops over the ranks kappa
        for i in range(self.Nk):
            print('Run [{0}/{1}], kappa = {2}'.format(i + 1, self.Nk, self.k[i]))
            print()

            with ClearCache():
                MSE[i], Dist[i] = self.simulate_one(self.m[i], self.Teacher[i], self.Zinf[i])

        self.MSE = MSE
        self.Dist = Dist

        t1 = time()

        print('Total time GD: {}'.format(round(t1 - t0, 2)))
        print()

    def save(self):
        """Saves the simulator in a pickle file"""
        path = 'Simulators/SimGD_{}.pickle'.format(self.ID)
        print("Saving Simulator...")
        print("ID = {}".format(self.ID))
        with open(path, 'wb') as file:
            pickle.dump(self, file)
