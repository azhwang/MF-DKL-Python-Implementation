import argparse
from model import MultiFidelityModel

def main():
    parser = argparse.ArgumentParser(description='MF-MI-Greedy')
    parser.add_argument('--budget', type=float, default=1000, help='budget')
    parser.add_argument('--num-fidelity', type=int, metavar='m', help='number of fidelities')
    parser.add_argument('--fidelity-costs', type=list, help='cost of each fidelity level')
    parser.add_argument('--data', type=str, help='path to data')
    parser.add_argument('--kernel', type=str, choices=["RBF", "DKL", "Matern"], help='kernel to use')
    parser.add_argument('--acquisition-function', type=str, choices=["UCB", "EI"], help='acquisition function')
    args = parser.parse_args()

    
    model = MultiFidelityModel(args.num_fidel, args.total_budget, args.fidelity_costs, args.kernel, args.acquisition_function)