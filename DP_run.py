# load modules and dataset
from ray.tune.progress_reporter import CLIReporter
from DP_server import *
from DP_load_dataset import *
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import pandas as pd

def run_dp(method, model, dataset, prn = True, seed = 123, trial = False, select_round = False,pref_bs=64,local_times=10,n_obj=2,
           weight_merged=None, adaptive_dist=None, **kwargs):
    # choose the model
    if model == 'logistic regression':
        arc = logReg
    elif model == 'multilayer perceptron':

        arc = mlp
    else:
        Warning('Does not support this model!')
        exit(1)

    # set up the dataset
    if dataset == 'synthetic':
        Z, num_features, info = 2, 3, synthetic_info
    elif dataset == 'adult':
        Z, num_features, info = 2, adult_num_features, adult_info
    elif dataset == 'compas':
        Z, num_features, info = compas_z, compas_num_features, compas_info
    elif dataset == 'communities':
        Z, num_features, info = communities_z, communities_num_features, communities_info
    elif dataset == 'bank':
        Z, num_features, info = bank_z, bank_num_features, bank_info
    else:
        Warning('Does not support this dataset!')
        exit(1)

    # set up the server
    if method =='praffl':
        head=Hypernet(n_obj=n_obj,num_classes=2, seed=seed)
        base = PF_MLP(num_features=num_features, num_classes=2, seed=seed)
        server = Server(BaseHeadSplit(base, head), info, train_prn=False, seed=seed, Z=Z,
                        ret=True, prn=prn, trial=trial, select_round=select_round, dataset=dataset, pref_bs=pref_bs, local_times=local_times)
    else:
        server = Server(arc(num_features=num_features, num_classes=2, seed = seed), info, train_prn = False, seed = seed,
                        Z = Z, ret = True, prn = prn, trial = trial, select_round = select_round)
    # execute
    if method == 'equifl':
        result = server.EquiFL(**kwargs)
    elif method == 'uflfb':
        result = server.UFLFB(**kwargs)
    elif method == 'fedfb':
        result = server.FedFB(**kwargs)
    elif method == 'praffl':
        result = server.Praffl(**kwargs)
    elif method == 'cflfb':
        acc, dpdisp, classifier = server.CFLFB(**kwargs)
    elif method == 'fflfb':
        result = server.FFLFB(**kwargs)
    elif method == 'fairfed':
        result = server.FairFed(**kwargs)
    elif method == 'agnosticfair':
        result = server.FAFL(**kwargs)
    else:
        Warning('Does not support this method!')
        exit(1)

    return result


def sim_dp_man(method, model, dataset, num_sim = 5, seed = 0, select_round = False, pref_bs=64, local_times=10,
               weight_merged=None, adaptive_dist=None,  **kwargs):
    global_results = []
    local_results = []
    global_results_adaptmerge = []
    pref_collect = []
    Val_HV = []
    for seed in range(num_sim):
        test_re, test_global, collected_pref, val_hv = run_dp(method, model, dataset, prn = True, seed = seed, trial = False, adaptive_dist=adaptive_dist,
                              select_round = select_round,pref_bs=pref_bs, local_times=local_times,weight_merged=weight_merged, **kwargs)
        local_results.append(test_re)
        global_results.append(test_global)
        pref_collect.append(collected_pref)
        Val_HV.append(val_hv)


if __name__ == "__main__":
    import sys, os
    import argparse
    working_dir = '.'
    # Here insert runing path
    sys.path.insert(1, os.path.join("/home/yerongguang/Fairness/Praffl"))
    os.environ["PYTHONPATH"] = os.path.join("/home/yerongguang/Fairness/Praffl")
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='praffl', type=str)
    parser.add_argument('--model', default='multilayer perceptron', type=str)
    parser.add_argument('--dataset', default='synthetic', type=str)
    parser.add_argument('--num_sim', default=5, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_rounds', default=15, type=int)
    parser.add_argument('--pref_bs', default=8, type=int)
    parser.add_argument('--local_times', default=15, type=int)
    parser.add_argument('--n_obj', default=2, type=int)

    args = parser.parse_args()
    print(args)
    if args.name in ['uflfb', 'cflfb', 'fflfb', 'agnosticfair']:
        sim_dp_man(
            method=args.name,
            model=args.model,
            dataset=args.dataset,
            num_sim=args.num_sim,
            seed=args.seed,

        )
    else:
        sim_dp_man(
            method=args.name,
            model=args.model,
            dataset=args.dataset,
            num_sim=args.num_sim,
            seed=args.seed,
            num_rounds=args.num_rounds,
            pref_bs = args.pref_bs,
            local_times = args.local_times,
            n_obj=args.n_obj,
            weight_merged = args.weight_merged,
            adaptive_dist =args.adaptive_dist
        )


