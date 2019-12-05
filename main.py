import argparse
from scipy.io import loadmat
from model import MultiFidelityModel
import torch
from botorch.utils import standardize

def load_data(dataset):
    if dataset == "astronomy":
        pass
    else: # nanophotonics data
        fomsx = torch.Tensor(loadmat('FOMs_x.mat', mat_dtype=True)['RVs_rnd'])
        fomsy = torch.Tensor(loadmat('FOMs_y.mat', mat_dtype=True)['FOMs'])
        
        # normalize
        minsx = torch.min(fomsx, 1)[0]
        maxsx = torch.max(fomsx, 1)[0]
        fomsx = (fomsx - minsx[:, None]) / maxsx[:, None]
    return fomsx, standardize(fomsy), minsx, maxsx


def main():
    parser = argparse.ArgumentParser(description='MF-MI-Greedy')
    parser.add_argument('--budget', type=float, default=900, help='budget')
    parser.add_argument('--num-fidelity', type=int, default=3, metavar='m', help='number of fidelities')
    parser.add_argument('--fidelity-costs', type=list, default=[9, 2.25, 1], help='cost of each fidelity level')
    parser.add_argument('--dataset', type=str, default='nanophotonics', choices=['nanophotonics', 'astronomy'],
                        help='dataset name')
    parser.add_argument('--device', type=str, default='cpu', help='gpu')
    parser.add_argument('--kernel', type=str, default="RBF", choices=["RBF", "DKL", "Matern"], help='kernel to use')
    parser.add_argument('--acquisition-function', type=str, default="UCB", choices=["UCB", "EI"], help='acquisition function')
    args = parser.parse_args()

    X, Y, _, _ = load_data(args.dataset)
    idx = 2
    # randomize
    
    # fidelity level doesn't need same 10 points
    # debug with 2 or 3
    train_x = X[:idx, :].contiguous()
    train_y = Y[:idx, :].contiguous()

    if args.dataset == 'nanophotonics':
        model = MultiFidelityModel(args.num_fidelity, args.budget, args.fidelity_costs, args.kernel, (X, Y),
                               args.acquisition_function, train_x, train_y, 5, is_discrete=True, device=args.device)

        best_x = model.optimize()
        print(best_x)

if __name__=="__main__":
    main()