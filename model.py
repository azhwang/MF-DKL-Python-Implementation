from botorch.models.gp_regression import SingleTaskGP
import numpy as np
import torch
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel, AdditiveKernel
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.acquisition import AcquisitionFunction
from copy import deepcopy
import math

class MultiFidelityModel():
    """A single task multi-fidelity GP model with abstract kernel.
    """

    def __init__(self, num_fidel, total_budget, costs, kernel_name, true_fns,
        acq_fn_name, bounds, num_candidates, num_restarts, raw_samples, 
        train_Xs, train_Ys, k, is_discrete=False, beta=0.1, device='cpu'):
        '''Initializes a model for MF-MI-Greedy Baysian Optimization
        Args:
        - num_fidel (int): number of fidelity levels (including target)
        - total_budget (float): total budget for exploration
        - costs (list): cost of each fidelity level (sorted from lowest to 
            highest fidelity)
        - kernel_name (str): kernel to use for covariance matrix
        - acq_fn_name (str): acq_fn to use for sf_gp_opt
        - beta (float): beta for UCB
        - train_Xs (list of Tensors): list of training input sets for each 
            fidelity
        - train_Ys (list of Tensors): list of training output sets for each 
            fidelity
        '''
        self.bounds = bounds
        self.num_candidates = num_candidates
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.true_fns = true_fns
        self.k = k
        self.is_discrete = is_discrete
        # initialize GP for target function
        kernels = {
            'RBF': ScaleKernel(RBFKernel()),
            'Matern': None # Botorch defaults to Matern
        }
        assert (len(train_Xs) == len(train_Ys), "Unequal points and labels")
        assert (len(train_Xs) == num_fidel, "unspecified training sets for"  
            + "certain fidelity levels")
        self.f_m = SingleTaskGP(train_Xs[-1], train_Ys[-1], 
                covar_module=kernels[kernel_name])

        self.costs = costs
        self.B = total_budget
        self.selected = set()
        self.num_fidel = num_fidel
        # initialize GPs for each fidelity level
        self.e_ls = []
        self.f_ls = []
        for i in range(num_fidel - 1):
            e_l_covar = kernels[kernel_name]
            self.f_ls.append(SingleTaskGP(train_Xs[i], train_Ys[i], covar_module=AdditiveKernel(self.f_m.covar_module, e_l_covar)))
            self.e_ls.append(SingleTaskGP(torch.tensor([]).to(device), 
                    torch.tensor([]).to(device), 
                    covar_module=e_l_covar))
            
        if acq_fn_name == "EI":
            self.acq_fn = ExpectedImprovement(self.f_m, -1000000)
        else:
            self.acq_fn = UpperConfidenceBound(self.f_m, beta)

    def information_gain_single_point(self, x, l):
        return self.f_ls[l].covar_module(x, x) - self.e_ls[l].covar_module(x, x)

    def information_gain_set(self, x, l, L, f_ls, e_ls):
        L.append((x, l))
        cov_1 = self.covariance_across_levels(f_ls, L)
        cov_2 = self.covariance_across_noise(e_ls, L)
        # need to ask about this calculation
        return np.linalg.det(cov_1) - np.linalg.det(cov_2)

    def covariance_across_levels(self, f_ls, points):
        cov = np.zeros((len(points), len(points)))
        for i, (x1, l1) in enumerate(points):
            for j, (x2, l2) in enumerate(points):
                if l1 == l2:
                    cov[i, j] = f_ls[l1].covar_module(x1, x2)
                else:
                    cov[i, j] = self.f_m.covar_module(x1, x2)
        return cov

    def covariance_across_noise(self, e_ls, points):
        cov = np.zeros((len(points), len(points)))
        for i, (x1, l1) in enumerate(points):
            for j, (x2, l2) in enumerate(points):
                if l1 == l2:
                    cov[i, j] = e_ls[l1].covar_module(x1, x2)
                else:
                    cov[i, j] = 0
        return cov

    def explore_lf(self):
        f_ls_initial = deepcopy(self.f_ls)
        e_ls_initial = deepcopy(self.e_ls)

        actions_fidel = []
        action_cost = 0
        threshold = 1/np.sqrt(self.B)
        # TODO argmax information gain
        if self.is_discrete:
            best_ig = -math.inf
            l = None
            x = None
            for row, fidel in enumerate(self.true_fns):
                for x_l, _ in row:
                    ig = self.information_gain_single_point(x_l, fidel)
                    if ig > best_ig:
                        best_ig = ig
                        l = fidel
                        x = x_l

        else:
            # use optimizer to get argmax
            # iterate over each fidelity level
            # use continuous optimization techniques to then optimize over x
            # choose x, l with maximum info gain value
            pass

        if l == None:
            return set(actions_fidel), 0
        if l == self.num_fidel-1:
            return set(actions_fidel), 0
        elif self.information_gain_set(x, l, actions_fidel, f_ls_initial, e_ls_initial) < threshold:
            return set(actions_fidel), 0
        else:
            actions_fidel.append((x, l))
            action_cost += self.costs[l]
            if self.is_discrete:
                self.f_ls[l] = self.f_ls[l].condition_on_observations(x, self.true_fns[l][x])
            else:
                self.f_ls[l] = self.f_ls[l].condition_on_observations(x, self.true_fns[l](x))

        return set(actions_fidel), action_cost

    def sf_gp_opt(self):
        candidate, acq_value = optimize_acqf(self.acq_fn, bounds=self.bounds, q=self.num_candidates, num_restarts=self.num_restarts, raw_samples=self.raw_samples)
        acq_vals = acq_value.cpu().numpy()
        idx, = np.where(acq_vals == np.max(acq_vals))
        return candidate[idx]

    def optimize(self):
        while self.B > 0:
            L, total_cost = self.explore_lf()
            x = self.sf_gp_opt()
            #total_queried = len(L) + 1
            self.selected.update(L)
            self.selected.add((x, self.num_fidel-1))
            self.B -= total_cost + self.costs[-1]
            
            # update posterior of f_m 
            self.f_m = self.f_m.condition_on_observations(x, self.true_fns[self.num_fidel-1](x))
            
            # update hyperparameters
            #if total_queried >= self.k:
                # update hyperparameters
            #    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gpytorch.likelihoods.GaussianLikelihood(), self.f_m)
                

        return x




