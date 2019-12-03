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
        acq_fn_name, train_Xs, train_Ys, k, is_discrete=False, beta=0.1, device='cpu'):
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
        self.true_fns = true_fns
        self.k = k
        self.is_discrete = is_discrete
        self.bounds = torch.stack([torch.zeros(train_Xs.size(1)), torch.ones(train_Xs.size(1))])
        # initialize GP for target function
        kernels = {
            'RBF': ScaleKernel(RBFKernel()),
            'Matern': None # Botorch defaults to Matern
        }

        self.f_m = SingleTaskGP(train_Xs, train_Ys[:, 0].unsqueeze(1), 
                covar_module=kernels[kernel_name])
        self.device = device
        self.costs = costs
        self.B = total_budget
        self.selected = set()
        self.num_fidel = num_fidel
        # initialize GPs for each fidelity level
        self.e_ls = [0]
        self.f_ls = [0]
        for i in range(1, num_fidel):
            e_l_covar = kernels[kernel_name]
            datax = torch.zeros(train_Xs.size())
            datay = torch.zeros(train_Ys[:, i].unsqueeze(1).size())
            self.f_ls.append(SingleTaskGP(train_Xs, train_Ys[:, i].unsqueeze(1), covar_module=AdditiveKernel(self.f_m.covar_module, e_l_covar)))
            self.e_ls.append(SingleTaskGP(datax, datay, 
                    covar_module=e_l_covar))
        if acq_fn_name == "EI":
            self.acq_fn = ExpectedImprovement(self.f_m, -1000000)
        else:
            self.acq_fn = UpperConfidenceBound(self.f_m, beta)

    def information_gain_single_point(self, x, l):
        return torch.det(self.f_ls[l].covar_module(x, x).evaluate()) - torch.det(self.e_ls[l].covar_module(x, x).evaluate())

    def information_gain_single_point_target(self, x):
        return torch.det(self.f_m.covar_module(x, x).evaluate())

    def information_gain_set(self, x, l, L, f_ls, e_ls):
        L.append((x, l))
        cov_1 = self.covariance_across_levels(f_ls, L)
        cov_2 = self.covariance_across_noise(e_ls, L)
        return torch.det(cov_1) - torch.det(cov_2)

    def covariance_across_levels(self, f_ls, points):
        cov = torch.tensor([])
        for i, (x1, l1) in enumerate(points):
            cov_row = torch.tensor([])
            for j, (x2, l2) in enumerate(points):
                if l1 == l2:
                    cov_row = torch.cat([cov_row, f_ls[l1].covar_module(x1, x2).evaluate()])
                else:
                    cov_row = torch.cat([cov_row, self.f_m.covar_module(x1, x2).evaluate()])
            
            cov = torch.cat([cov, cov_row], dim=1)
        return cov

    def covariance_across_noise(self, e_ls, points):
        cov = torch.tensor([])
        for i, (x1, l1) in enumerate(points):
            for j, (x2, l2) in enumerate(points):
                cov_row = torch.tensor([])
                if l1 == l2:
                    cov_row = torch.cat([cov_row, e_ls[l1].covar_module(x1, x2).evaluate()])
                else:
                    cov_row = torch.cat([cov_row, torch.zeros((len(x1), len(x2)))])
            cov = torch.cat([cov, cov_row], dim=1)
        return cov

    def explore_lf(self):
        f_ls_initial = deepcopy(self.f_ls)
        e_ls_initial = deepcopy(self.e_ls)

        actions_fidel = []
        action_cost = 0
        threshold = 1/np.sqrt(self.B)
        while True:
            if self.is_discrete:
                X, Y = self.true_fns
                best_ig = -math.inf
                l = None
                x = None
                x_idx = None
                for i, x_l in enumerate(X):
                    for fidel in range(self.num_fidel):
                        if fidel == 0:
                            ig = self.information_gain_single_point_target(x_l)
                        else:
                            ig = self.information_gain_single_point(x_l, fidel)
                        if ig > best_ig and (self.B - action_cost - self.costs[fidel]) > 0:
                            best_ig = ig
                            l = fidel
                            x = x_l
                            x_idx = i

            else:
                # use optimizer to get argmax
                # iterate over each fidelity level
                # use continuous optimization techniques to then optimize over x
                # choose x, l with maximum info gain value
                pass

            if l == None:
                print("stopped exploring due to no budget")
                return set(actions_fidel), 0
            elif l == 0:
                print("stopped exploring since querying target is best action")
                return set(actions_fidel), 0
            elif self.information_gain_set(x, l, actions_fidel, f_ls_initial, e_ls_initial) < threshold:
                print("stopped exploring since info gain ratio low")
                return set(actions_fidel), 0
            else:
                actions_fidel.append((x, l))
                action_cost += self.costs[l]
                if self.is_discrete:
                    self.f_ls[l] = self.f_ls[l].condition_on_observations(x, Y[x_idx][l])
                else:
                    self.f_ls[l] = self.f_ls[l].condition_on_observations(x, self.true_fns[l](x))

        return set(actions_fidel), action_cost

    def sf_gp_opt(self):
        if self.is_discrete:
            X, _ = self.true_fns
            acq_vals = self.acq_fn(X.unsqueeze(1))
            argmax = torch.argmax(acq_vals)
            return X[argmax], argmax
        else:
            candidate, acq_value = optimize_acqf(self.acq_fn, bounds=self.bounds, q=1, num_restarts=5, raw_samples=20)
            acq_vals = acq_value.cpu().numpy()
            idx, = np.where(acq_vals == np.max(acq_vals))
            return candidate[idx]

    def optimize(self):
        print("beginning optimization")
        while self.B > 0:
            print("exploring lf")
            L, total_cost = self.explore_lf()
            print("optimizing single fidelity")
            x, x_idx = self.sf_gp_opt()
            #total_queried = len(L) + 1
            self.selected.update(L)
            self.selected.add((x, 0))
            self.B -= total_cost + self.costs[0]
            print("number of queried: {}".format(len(self.selected)))
            print(self.selected)
            print("remaining budget: {}".format(self.B))
            
            # update posterior of f_m 
            print("updating posterior of f_m")
            if self.is_discrete:
                _, Y = self.true_fns
                print(Y[x_idx][0])
                self.f_m = self.f_m.condition_on_observations(x, Y[x_idx][0])
            else:
                self.f_m = self.f_m.condition_on_observations(x, self.true_fns[0](x))
            
            print("current sf opt: {}".format(Y[x_idx][0]))
            # update hyperparameters
            #if total_queried >= self.k:
                # update hyperparameters
            #    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gpytorch.likelihoods.GaussianLikelihood(), self.f_m)
            
        return x




