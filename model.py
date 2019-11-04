from botorch.models.gp_regression import SingleTaskGP
import numpy as np
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.acquisition.analytic import ExpectedImprovement

class MultiFidelityModel():
    """A single task multi-fidelity GP model with abstract kernel.
    """

    def __init__(self, num_fidel, total_budget, costs, kernel_name, acq_fn_name, beta=0.2, train_Xs=None, train_Ys=None, device='cpu'):
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
        
        
    def explore_lf(self):
        actions = []
        action_costs = []
        threshold = 1/np.sqrt(self.B)
        pass

    def sf_gp_opt(self):
        pass