from botorch.models.gp_regression import SingleTaskGP
import numpy as np
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim import optimize_acqf


class MultiFidelityModel():
    """A single task multi-fidelity GP model with abstract kernel.
    """

    def __init__(self, num_fidel, total_budget, costs, kernel_name, acq_fn_name, bounds, num_candidates, num_restarts, raw_samples, beta=0.1, train_Xs=None, train_Ys=None, device='cpu'):
        '''Initializes a model for MF-MI-Greedy Baysian Optimization
        Args:
        - num_fidel (int): number of fidelity levels (including target)
        - total_budget (float): total budget for exploration
        - costs (list): cost of each fidelity level (sorted from lowest to highest fidelity)
        - kernel_name (str): kernel to use for covariance matrix
        - acq_fn_name (str): acq_fn to use for sf_gp_opt
        - beta (float): beta for UCB
        - train_Xs (list of Tensors): list of training input sets for each fidelity
        - train_Ys (list of Tensors): list of training output sets for each fidelity
        '''
        self.bounds = bounds
        self.num_candidates = num_candidates
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        # initialize GP for target function
        kernels = {
            'RBF': ScaleKernel(RBFKernel()),
            'Matern': None # Botorch defaults to Matern
        }
        if train_Xs is None:
            self.f_m = SingleTaskGP(torch.tensor([]).to(device), torch.tensor([]).to(device), covar_module=kernels[kernel_name])
        else:
            assert (len(train_Xs) == len(train_Ys), "Unequal points and labels")
            assert (len(train_Xs) == num_fidel, "unspecified training sets for certain fidelity levels")
            self.f_m = SingleTaskGP(train_Xs[-1], train_Ys[-1], covar_module=kernels[kernel_name])

        self.costs = costs
        self.B = total_budget
        self.selected = set()
        self.observed_y = set()
        self.num_fidel = num_fidel
        # initialize GPs for each fidelity level
        self.e_ls = []
        if train_Xs is None:
            for _ in range(num_fidel - 1):
                self.e_ls.append(SingleTaskGP(torch.tensor([]).to(device), torch.tensor([]).to(device), covar_module=kernels[kernel_name]))
        else:
            # TODO
            raise Exception("Still unimplemented")
            # f_l is modeled by train_X, train_Y. Need to calculate gaussian process e_l = f_m - f_l and create a model object

        if acq_fn_name == "EI":
            self.acq_fn = ExpectedImprovement(self.f_m, -1000000)
        else:
            self.acq_fn = UpperConfidenceBound(self.f_m, beta)
        
    def MI(self, y):
        #TODO
        pass
        
    def explore_lf(self):
        actions = []
        action_cost = 0
        threshold = 1/np.sqrt(self.B)
        # TODO
        return set(), total_cost

    def sf_gp_opt(self):
        candidate, acq_value = optimize_acqf(self.acq_fn, bounds=self.bounds, q=self.num_candidates, num_restarts=self.num_restarts, raw_samples=self.raw_samples)
        return torch.max(candidate)

    def optimize(self):
        while self.B > 0:
            L, total_cost = self.explore_lf()
            x = self.sf_gp_opt()
            self.selected.update(L)
            self.selected.add((x, self.num_fidel-1))
            self.B -= total_cost + self.costs[-1]
        return x #?