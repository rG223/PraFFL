import torch, copy, time, random, warnings, os
import numpy as np
import torch.nn.functional as F
from bokeh.core.property.color import Alpha
from dask.array.random import pareto
from flask import request
from numba.core.ir import Global
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
from ray import tune
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.distributions as dist
from torch.nn.utils import clip_grad_norm
from pymoo.indicators.hv import HV
################## MODEL SETTING ########################
DEVICE = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# if torch.cuda.is_available():
#     DEVICE = torch.device('cuda')
# else:
#     DEVICE = torch.device('cpu')
# DEVICE = "cuda:5"
print(DEVICE)
#########################################################

class Server(object):
    def __init__(self, model, dataset_info, seed = 123, num_workers = 4, ret = False,
                train_prn = False, metric = "Demographic disparity", select_round = False, weight_merged=None, adaptive_dist=None,
                batch_size = 128, print_every = 1, fraction_clients = 1, Z = 2, prn = True, trial = False, dataset=None,pref_bs=64, local_times=5):
        """
        Server execution.

        Parameters
        ----------
        model: torch.nn.Module object.

        dataset_info: a list of three objects.
            - train_dataset: Dataset object.
            - test_dataset: Dataset object.
            - clients_idx: a list of lists, with each sublist contains the indexs of the training samples in one client.
                    the length of the list is the number of clients.

        seed: random seed.

        num_workers: number of workers.

        ret: boolean value. If true, return the accuracy and fairness measure and print nothing; else print the log and return None.

        train_prn: boolean value. If true, print the batch loss in local epochs.

        metric: three options, "Risk Difference", "pRule", "Demographic disparity".

        batch_size: a positive integer.

        print_every: a positive integer. eg. print_every = 1 -> print the information of that global round every 1 round.

        fraction_clients: float from 0 to 1. The fraction of clients chose to update the weights in each round.
        """

        self.model = model
        # if torch.cuda.device_count()>1:
        #     self.model = nn.DataParallel(self.model)
        self.model.to(DEVICE)
        self.seed = seed
        self.num_workers = num_workers
        self.dataset_name = dataset
        self.ret = ret
        self.prn = prn
        self.train_prn = False if ret else train_prn
        self.pref_bs = pref_bs
        self.local_times = local_times
        self.metric = metric
        self.disparity = DPDisparity
        self.weight_merged = weight_merged
        self.batch_size = batch_size
        self.print_every = print_every
        self.fraction_clients = fraction_clients
        self.train_dataset, self.test_dataset, self.clients_idx, self.test_clients_idx = dataset_info
        self.num_clients = len(self.clients_idx)
        self.Z = Z

        self.trial = trial
        self.select_round = select_round
        self.adaptive_dist = adaptive_dist
        self.trainloader, self.validloader = self.train_val(self.train_dataset, batch_size)

    def train_val(self, dataset, batch_size, idxs_train_full = None, split = False):
        """
        Returns train, validation for a given local training dataset
        and user indexes.
        """
        torch.manual_seed(self.seed)

        # split indexes for train, validation (90, 10)
        if idxs_train_full == None: idxs_train_full = np.arange(len(dataset))
        idxs_train = idxs_train_full[:int(0.9*len(idxs_train_full))]
        idxs_val = idxs_train_full[int(0.9*len(idxs_train_full)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                    batch_size=batch_size, shuffle=True)

        if split:
            validloader = {}
            for sen in range(self.Z):
                sen_idx = np.where(dataset.sen[idxs_val] == sen)[0]
                validloader[sen] = DataLoader(DatasetSplit(dataset, idxs_val[sen_idx]),
                                        batch_size=max(int(len(idxs_val)/10),10), shuffle=False)
        else:
            validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                     batch_size=max(int(len(idxs_val)/10),10), shuffle=False)
        return trainloader, validloader

    def FedAvg(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, optimizer = "adam"):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        ind = 0
        ref_point = np.array([1] * 2)
        HV_ind = HV(ref_point=ref_point)
        # Training
        train_loss, train_accuracy = [], []
        val_hv = []
        start_time = time.time()
        local_model__ = {i: copy.deepcopy(self.model) for i in range(self.num_clients)}
        weights = self.model.state_dict()
        global_model = copy.deepcopy(self.model)
        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses = [], []
            global_weights = []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            m = max(1, int(self.fraction_clients * self.num_clients)) # the number of clients to be chosen in each round_
            idxs_users = np.random.choice(range(self.num_clients), m, replace=False)

            for idx in idxs_users:
                self.model = local_model__[idx]
                self.model.train()
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx],
                            batch_size = self.batch_size, option = "unconstrained", seed = self.seed, prn = self.train_prn, Z = self.Z)

                w, loss = local_model.standard_update(
                                model=copy.deepcopy(self.model), global_round=round_,
                                    learning_rate = learning_rate, local_epochs = local_epochs,
                                    optimizer = optimizer)
                local_model__[idx].load_state_dict(w)

                weights_only = {key: value for key, value in w.items() if ('weight' in key) and (key != 'linear2.weight')}
                local_weights.append(copy.deepcopy(weights_only))
                global_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            weights = average_weights(global_weights, self.clients_idx, idxs_users)
            global_model.load_state_dict(weights)
            part_weights = average_weights(local_weights, self.clients_idx, idxs_users)
            for c_idx in range(self.num_clients):
                local_model__[c_idx].load_state_dict(part_weights, strict=False)
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z

            Obj_array = []
            Total_hv = 0
            for c in range(m):
                self.model = local_model__[c]
                self.model.eval()
                n_yz = {}
                for y in [0, 1]:
                    for z in range(self.Z):
                        n_yz[(y, z)] = 0
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c],
                            batch_size = self.batch_size, option = "unconstrained", seed = self.seed, prn = self.train_prn, Z = self.Z)
                # validation dataset inference
                acc, loss, n_yz_c, acc_loss, fair_loss, _ = local_model.inference(model = self.model)
                list_acc.append(acc)

                for yz in n_yz:
                    n_yz[yz] += n_yz_c[yz]

                if self.prn:
                    print("Client %d: accuracy loss: %.2f | fairness loss %.2f | %s = %.2f"% (
                            c+1, acc_loss, fair_loss, self.metric[0], self.disparity(n_yz_c)))
                Obj_array.append([1- acc, self.disparity(n_yz)])
                Total_hv += acc * (1-self.disparity(n_yz))


            val_hv.append(Total_hv/3)
            if ind < Total_hv / 3:
                Best_global_model = copy.deepcopy(global_model)
                Best_local_models = copy.deepcopy(local_model__)
                ind = Total_hv / 3
            train_accuracy.append(sum(list_acc)/len(list_acc))

            print(f'Valiation set: HV: {Obj_array}, averaged hv: {Total_hv/m}')

        # Local Test inference after completion of training
        Local_hv = 0
        Local_result = []
        for c in range(self.num_clients):
            try:
                test_acc, n_yz= self.test_inference(model=Best_local_models[c], test_dataset=self.test_dataset,
                                                    idxs=self.test_clients_idx[c] - len(self.train_dataset), local=True)
            except:
                test_acc, n_yz= self.test_inference(model=Best_local_models[c], test_dataset=self.test_dataset,
                                                    idxs=self.test_clients_idx[c], local=True)
            rd_dp = self.disparity(n_yz)
            Local_hv += test_acc * (1-rd_dp)
            Local_result.append([1-test_acc, rd_dp])

        print(f'Local Test dataset Client HV: {Local_hv / 3}')
        global_hv = 0
        test_acc, n_yz= self.test_inference(model=Best_global_model, test_dataset=self.test_dataset,
                                            idxs=self.test_clients_idx, local=False)
        rd_dp = self.disparity(n_yz)
        print(f'Gloal Test dataset Client {c} HV: {test_acc * (1-rd_dp)}')
        Global_result = [1-test_acc, rd_dp]
        return Local_result, Global_result, None, val_hv

    def Praffl(self, num_rounds=10, local_epochs=30, learning_rate=0.005, optimizer="adam"):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_loss, train_accuracy, train_acc_loss, train_fair_loss = [], [], [], []
        Results = []
        start_time = time.time()
        weights = self.model.state_dict()
        ind = 0
        ref_point = np.array([1] * 2)
        HV_ind = HV(ref_point=ref_point)
        Hyper_net_client = {}
        collected_pref = {i: [] for i in range(self.num_clients)}
        val_hv = []
        for round_ in tqdm(range(num_rounds)):
            global_weights, global_losses, local_weights = [], [], []
            if self.prn:
                print(f'\n | Global Training Round : {round_ + 1} |\n')
            self.model.train()
            m = max(1,
                    int(self.fraction_clients * self.num_clients))  # the number of clients to be chosen in each round_
            idxs_users = np.random.choice(range(self.num_clients), m, replace=False)
            if round_==0:
                z_min = [torch.tensor([torch.inf, torch.inf]).to(DEVICE) for _ in range(len(idxs_users))]
                z_max = [torch.tensor([-torch.inf, -torch.inf]).to(DEVICE) for _ in range(len(idxs_users))]
            for idx in idxs_users:
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx],
                                     batch_size=self.batch_size, option="unconstrained", seed=self.seed,
                                     prn=self.train_prn, Z=self.Z, test=False)
                if round_ > 0:
                    self.model.head.load_state_dict(Hyper_net_client[idx])
                w, head, loss, z_min, z_max, collected_pref_idx = local_model.Peasonfair_update(
                    model=copy.deepcopy(self.model), global_round=round_,
                    learning_rate=learning_rate, local_epochs=local_epochs, collected_pref=collected_pref[idx],
                    optimizer=optimizer, idxs_user=idx, z_min=z_min, z_max=z_max, Z=self.Z, pref_bs=self.pref_bs, local_times=self.local_times)
                global_weights.append(copy.deepcopy(w))
                global_losses.append(copy.deepcopy(loss))
                Hyper_net_client[idx] = copy.deepcopy(head)
                collected_pref[idx] = collected_pref_idx
            # update global weights
            weights = average_weights(global_weights, self.clients_idx, idxs_users)
            self.model.base.load_state_dict(weights)
            loss_avg = sum(global_losses) / len(global_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc, global_list_acc = [], []
            fair_loss_valid, global_fair_loss_valid, acc_loss_avlid, global_acc_loss_avlid = [], [], [], []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            train_time = time.time() - start_time
            test_rays = torch.tensor(das_dennis(100, 2), dtype=torch.float32).to(DEVICE)
            self.model.eval()
            val_hv_value = 0
            for c in range(m):
                Obj_array = []
                ################################
                n_yz = [{} for _ in range(101)]
                for n in range(len(n_yz)):
                    for y in [0, 1]:
                        for z in range(self.Z):
                            n_yz[n][(y, z)] = 0
                ###############################
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c],
                                     batch_size=self.batch_size, option="unconstrained", seed=self.seed,
                                     prn=self.train_prn, Z=self.Z, test=False)
                # validation dataset inference
                self.model.head.load_state_dict(Hyper_net_client[c])
                acc, loss, n_yz_c, acc_loss, fair_loss, _, test_rays = local_model.inference_pf(model=self.model, pref=test_rays)
                list_acc = np.array(acc)
                for num in range(len(test_rays)):
                    for yz in n_yz[0]:
                        n_yz[num][yz] += n_yz_c[num][yz]
                    Obj_array.append([1 - list_acc[num], self.disparity(n_yz[num])])

                val_hv_value += HV_ind(np.array(Obj_array))

            print("*******Local HV", val_hv_value / 3,'********')
            val_hv.extend([val_hv_value / 3])
            # if self.trial:
            if ind < val_hv_value / 3:
                Best_model = copy.deepcopy(self.model)
                print(f'Local HV is improved to {val_hv_value / 3}')
                ind = val_hv_value / 3
                print(f'Current Local best HV is:{ind}')
                Best_Hyper_for_client = copy.deepcopy(Hyper_net_client)
        # pref = torch.tensor(das_dennis(10, 2), dtype=torch.float32).to(DEVICE)
        # test_acc, rd = self.test_inference_pf(model=Best_model, pref=pref)
        #
        #
        if self.prn:
            print('\n Total Run Time: {0:0.4f} sec'.format(time.time() - start_time))

        # test dataset inference
        test_rays = torch.tensor(das_dennis(100, 2), dtype=torch.float32).to(DEVICE)
        ######################################################################Local Testing###################################################################
        test_hv_value = 0
        Obj_array = []
        for c in range(m):
            ################################
            n_yz = [{} for _ in range(101)]
            for n in range(len(n_yz)):
                for y in [0, 1]:
                    for z in range(self.Z):
                        n_yz[n][(y, z)] = 0
            Obj_array_client = []
            ################################
            try:
                local_model = Client(dataset=self.test_dataset, idxs=self.test_clients_idx[c] - len(self.train_dataset),
                                     batch_size=self.batch_size, option="unconstrained", seed=self.seed,
                                     prn=self.train_prn, Z=self.Z, test=True, local=True)
            except:
                local_model = Client(dataset=self.test_dataset, idxs=self.test_clients_idx[c],
                                     batch_size=self.batch_size, option="unconstrained", seed=self.seed,
                                     prn=self.train_prn, Z=self.Z, test=True, local=True)
            Best_model.head.load_state_dict(Best_Hyper_for_client[c])
            acc, loss, n_yz_c, acc_loss, fair_loss, _, test_rays = local_model.inference_pf(model=Best_model, pref=test_rays)
            list_acc = np.array(acc)
            for num in range(len(test_rays)):
                for yz in n_yz[0]:
                    n_yz[num][yz] += n_yz_c[num][yz]
                Obj_array_client.append([1 - list_acc[num], self.disparity(n_yz[num])])
            Obj_array.append(Obj_array_client)
            test_hv_value += HV_ind(np.array(Obj_array_client))
        print(f'Test Local HV value:{test_hv_value / m}')
        #########################################################################################################################################################
        ######################################################################Global Testing###################################################################
        global_test_hv_value = 0
        global_Obj_array = []
        for c in range(m):
            ###############################
            global_Obj_array_client = []
            global_n_yz = [{} for _ in range(101)]
            for n in range(len(global_n_yz)):
                for y in [0, 1]:
                    for z in range(self.Z):
                        global_n_yz[n][(y, z)] = 0
            ###############################

            local_model = Client(dataset=self.test_dataset, idxs=self.test_clients_idx,
                                 batch_size=self.batch_size, option="unconstrained", seed=self.seed,
                                 prn=self.train_prn, Z=self.Z, test=True, local=False)
            Best_model.head.load_state_dict(Best_Hyper_for_client[c])
            acc, loss, n_yz_c, acc_loss, fair_loss, _, test_rays = local_model.inference_pf(model=Best_model, pref=test_rays)
            global_list_acc = np.array(acc)
            for num in range(len(test_rays)):
                for yz in n_yz[0]:
                    global_n_yz[num][yz] += n_yz_c[num][yz]
                global_Obj_array_client.append([1 - global_list_acc[num], self.disparity(global_n_yz[num])])
            global_test_hv_value += HV_ind(np.array(global_Obj_array_client))
            global_Obj_array.append(global_Obj_array_client)

        print(f'Test Global HV value:{global_test_hv_value / m}')
        ####################################################################################################################################
        # if self.ret:
        #     return test_acc, rd, self.model
        return Obj_array, global_Obj_array, collected_pref, val_hv

    def FedFB(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, optimizer = 'adam', alpha = 0.01, bits = False):
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        best_hv = 0
        ref_point = np.array([1, 1])
        ind = HV(ref_point=ref_point)
        # Training
        train_loss, train_accuracy, train_fair, train_accloss, train_fairloss = [], [], [], [], []
        Results = []
        val_hv = []
        start_time = time.time()
        weights = self.model.state_dict()
        if self.select_round: best_fairness = float('inf')

        # the number of samples whose label is y and sensitive attribute is z
        m_yz, lbd = {}, {}
        for y in [0,1]:
            for z in range(self.Z):
                m_yz[(y,z)] = ((self.train_dataset.y == y) & (self.train_dataset.sen == z)).sum()

        for y in [0,1]:
            for z in range(self.Z):
                lbd[(y,z)] = (m_yz[(1,z)] + m_yz[(0,z)])/len(self.train_dataset)

        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses, nc = [], [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            self.model.train()

            for idx in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset,
                                            idxs=self.clients_idx[idx], batch_size = self.batch_size,
                                        option = "FB-Variant1",
                                        seed = self.seed, prn = self.train_prn, Z = self.Z)

                w, loss, nc_ = local_model.fb2_update(
                                model=copy.deepcopy(self.model), global_round=round_,
                                    learning_rate = learning_rate, local_epochs = local_epochs,
                                    optimizer = optimizer, m_yz = m_yz, lbd = lbd)
                nc.append(nc_)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            weights = weighted_average_weights(local_weights, nc, sum(nc))
            self.model.load_state_dict(weights)

            loss_avg = sum(local_losses) / len(local_losses)

            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            list_fair = []
            list_accloss = []
            list_fairloss = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            n_yz, f_z = {}, {}
            for z in range(self.Z):
                for y in [0,1]:
                    n_yz[(y,z)] = 0

            for z in range(1, self.Z):
                f_z[z] = 0
            train_time = time.time() - start_time
            self.model.eval()
            for c in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset,
                                            idxs=self.clients_idx[c], batch_size = self.batch_size, option = "FB-Variant1",
                                            seed = self.seed, prn = self.train_prn, Z = self.Z)
                # validation dataset inference
                acc, loss, n_yz_c, acc_loss, fair_loss, f_z_c = local_model.inference(model = self.model, train = True, bits = bits, truem_yz= m_yz)
                list_acc.append(acc)
                list_fair.append(self.disparity(n_yz_c))
                list_accloss.append(acc_loss)
                list_fairloss.append(fair_loss)
                for yz in n_yz:
                    n_yz[yz] += n_yz_c[yz]

                for z in range(1, self.Z):
                    f_z[z] += f_z_c[z] + m_yz[(0,0)]/(m_yz[(0,0)] + m_yz[(1,0)]) - m_yz[(0,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

                if self.prn: print("Client %d: accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                    c+1, acc_loss, fair_loss, self.metric, self.disparity(n_yz_c)))

            for z in range(self.Z):
                if z == 0:
                    lbd[(0,z)] -= alpha / (round_ + 1) ** .5 * sum([f_z[z] for z in range(1, self.Z)])
                    lbd[(0,z)] = lbd[(0,z)].item()
                    lbd[(0,z)] = max(0, min(lbd[(0,z)], 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset)))
                    lbd[(1,z)] = 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset) - lbd[(0,z)]
                else:
                    lbd[(0,z)] += alpha / (round_ + 1) ** .5 * f_z[z]
                    lbd[(0,z)] = lbd[(0,z)].item()
                    lbd[(0,z)] = max(0, min(lbd[(0,z)], 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset)))
                    lbd[(1,z)] = 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset) - lbd[(0,z)]
            result = []
            train_accuracy.append(sum(list_acc)/len(list_acc))
            train_fair.append(sum(list_fair) / len(list_fair))
            train_accloss.append(sum(list_accloss) / len(list_accloss))
            train_fairloss.append(sum(list_fairloss) / len(list_fairloss))
            # print global training loss after every 'i' rounds
            if self.prn:

                print(f' \nAvg Training Stats after {round_+1} global rounds:')
                print("Training loss: %.2f | Training accuracy: %.2f%% |Training %s: %.8f" % (
                    np.mean(np.array(train_loss)),
                    100*train_accuracy[-1], self.metric, self.disparity(n_yz)))
                result.append([1-train_accuracy[-1],self.disparity(n_yz), train_accloss[-1], train_fairloss[-1],train_time])
            if self.trial:
                with tune.checkpoint_dir(round_) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(self.model.state_dict(), path)

                tune.report(loss = loss, accuracy = train_accuracy[-1], disp = self.disparity(n_yz), iteration = round_+1)
            client_obj = [1-train_accuracy[-1], self.disparity(n_yz)]
            val_hv.append(client_obj)
            if ind(np.array(client_obj)) > best_hv:
                best_hv = ind(np.array(client_obj))
                test_model = copy.deepcopy(self.model.state_dict())
            print('Vailidation | Best HV is ', best_hv)
            Results.append(result)
        # Test inference after completion of training

        self.model.load_state_dict(test_model)

        # Local Test inference after completion of training
        Local_hv = 0
        Local_result = []
        for c in range(self.num_clients):
            try:
                test_acc, n_yz = self.test_inference(model=self.model, test_dataset=self.test_dataset,
                                                     idxs=self.test_clients_idx[c] - len(self.train_dataset),
                                                     local=True)
            except:
                test_acc, n_yz = self.test_inference(model=self.model, test_dataset=self.test_dataset,
                                                     idxs=self.test_clients_idx[c], local=True)
            rd_dp = self.disparity(n_yz)
            Local_hv += test_acc * (1 - rd_dp)
            Local_result.append([1 - test_acc, rd_dp])

        print(f'Local Test dataset Client HV: {Local_hv / 3}')

        test_acc, n_yz = self.test_inference(model=self.model, test_dataset=self.test_dataset,idxs=self.test_clients_idx,
                                                     local=False)


        rd = self.disparity(n_yz)
        print(f'Global Test dataset Client HV: {test_acc*(1-rd)}')
        if self.prn:
            print(f' \n Results after {num_rounds} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
            print("|---- Test HV: {:.2f}".format(ind(np.array([1 - test_acc, rd]))))
            # Compute fairness metric
            print("|---- Test "+ self.metric+": {:.8f}".format(rd))

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))


        return Local_result, [1-test_acc, rd], val_hv, None

    def CFLFB(self, outer_rounds = 10,  inner_epochs = 30, learning_rate = 0.005, optimizer = 'adam', alpha = 0.3):
        # new algorithm for demographic parity, add weights directly, signed gradient-based algorithm
        if self.Z == 2:
            # set seed
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)

            # Training
            train_loss, train_accuracy = [], []
            start_time = time.time()

            # Set optimizer for the local updates
            if optimizer == 'sgd':
                optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate,
                                            momentum=0.5) #
            elif optimizer == 'adam':
                optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate,
                                            weight_decay=1e-4)

            # the number of samples whose label is y and sensitive attribute is z
            m_yz, lbd = {}, {}
            for y in [0,1]:
                for z in range(self.Z):
                    m_yz[(y,z)] = ((self.train_dataset.y == y) & (self.train_dataset.sen == z)).sum()

            for y in [0,1]:
                for z in range(self.Z):
                    lbd[(y,z)] = m_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

            for round_ in tqdm(range(outer_rounds)):
                if self.prn and (round_+1) % 10 == 0: print(f'\n | Global Training Round : {round_+1} |\n')

                self.model.train()
                batch_loss = []
                for _ in range(inner_epochs):
                    for _, (features, labels, sensitive) in enumerate(self.trainloader):
                        features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                        sensitive = sensitive.to(DEVICE)
                        _, logits = self.model(features)

                        v = torch.randn(len(labels)).type(torch.DoubleTensor)

                        group_idx = {}

                        for y, z in lbd:
                            group_idx[(y,z)] = torch.where((labels == y) & (sensitive == z))[0]
                            v[group_idx[(y,z)]] = lbd[(y,z)] * sum([m_yz[(y,z)] for z in range(self.Z)]) / m_yz[(y,z)]
                        loss = weighted_loss(logits, labels, v)

                        optimizer.zero_grad()
                        if not np.isnan(loss.item()): loss.backward()
                        optimizer.step()
                        batch_loss.append(loss.item())

                loss_avg = sum(batch_loss)/len(batch_loss)
                train_loss.append(loss_avg)

                # Calculate avg training accuracy over all clients at every round
                list_acc = []
                # the number of samples which are assigned to class y and belong to the sensitive group z
                n_yz, loss_yz = {}, {}
                for y in [0,1]:
                    for z in range(self.Z):
                        n_yz[(y,z)] = 0
                        loss_yz[(y,z)] = 0

                self.model.eval()
                    # validation dataset inference
                acc, loss, n_yz, acc_loss, fair_loss, loss_yz = self.inference_m(model = self.model, option = 'FairBatch')
                list_acc.append(acc)


                if self.prn and (round_+1) % 10 == 0: print("Accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                    acc_loss, fair_loss, self.metric, self.disparity(n_yz)))

                # update the lambda according to the paper -> see Section A.1 of FairBatch
                # works well! The real batch size would be slightly different from the setting
                for y, z in loss_yz:
                    loss_yz[(y,z)] = loss_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

                y0_diff = loss_yz[(0,0)] - loss_yz[(0,1)]
                y1_diff = loss_yz[(1,0)] - loss_yz[(1,1)]
                if y0_diff > y1_diff:
                    lbd[(0,0)] -= alpha / (round_+1)
                    lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
                    lbd[(1,0)] = 1 - lbd[(0,0)]
                    lbd[(0,1)] += alpha / (round_+1)
                    lbd[(0,1)] = min(max(0, lbd[(0,1)]), 1)
                    lbd[(1,1)] = 1 - lbd[(0,1)]
                else:
                    lbd[(0,0)] += alpha / (round_+1)
                    lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
                    lbd[(0,1)] = 1 - lbd[(0,0)]
                    lbd[(1,0)] -= alpha / (round_+1)
                    lbd[(1,0)] = min(max(0, lbd[(1,0)]), 1)
                    lbd[(1,1)] = 1 - lbd[(1,0)]

                train_accuracy.append(sum(list_acc)/len(list_acc))

                # print global training loss after every 'i' rounds
                if self.prn and (round_+1) % 10 == 0:
                    if (round_+1) % self.print_every == 0:
                        print(f' \nAvg Training Stats after {round_+1} global rounds:')
                        print("Training loss: %.2f | Training accuracy: %.2f%% | Training %s: %8" % (
                            np.mean(np.array(train_loss)),
                            100*train_accuracy[-1], self.metric, self.disparity(n_yz)))

                if self.trial:
                    with tune.checkpoint_dir(round_) as checkpoint_dir:
                        path = os.path.join(checkpoint_dir, "checkpoint")
                        torch.save(self.model.state_dict(), path)

                    tune.report(loss = loss, accuracy = train_accuracy[-1], disp = self.disparity(n_yz), iteration = round_+1)


            # Test inference after completion of training
            test_acc, n_yz = self.test_inference(self.model, self.test_dataset)
            rd = self.disparity(n_yz)

            if self.prn:
                print(f' \n Results after {outer_rounds} global rounds of training:')
                print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
                print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

                # Compute fairness metric
                print("|---- Test "+ self.metric+": {:.8f}".format(rd))

                print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

            if self.ret: return test_acc, rd, self.model
        else:
            # new algorithm for demographic parity, add weights directly, signed gradient-based algorithm
            # set seed
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)

            # Training
            train_loss, train_accuracy = [], []
            start_time = time.time()

            # Set optimizer for the local updates
            if optimizer == 'sgd':
                optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate,
                                            momentum=0.5) #
            elif optimizer == 'adam':
                optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate,
                                            weight_decay=1e-4)

            # the number of samples whose label is y and sensitive attribute is z
            m_yz, lbd = {}, {}
            for y in [0,1]:
                for z in range(self.Z):
                    m_yz[(y,z)] = ((self.train_dataset.y == y) & (self.train_dataset.sen == z)).sum()

            for y in [0,1]:
                for z in range(self.Z):
                    lbd[(y,z)] = (m_yz[(1,z)] + m_yz[(0,z)])/len(self.train_dataset)

            for round_ in tqdm(range(outer_rounds)):
                if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

                self.model.train()
                batch_loss = []
                for _, (features, labels, sensitive) in enumerate(self.trainloader):
                    features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                    sensitive = sensitive.to(DEVICE)
                    _, logits = self.model(features)

                    v = torch.ones(len(labels)).type(torch.DoubleTensor)

                    group_idx = {}
                    for y, z in lbd:
                        group_idx[(y,z)] = torch.where((labels == y) & (sensitive == z))[0]
                        v[group_idx[(y,z)]] = lbd[(y,z)] / (m_yz[(1,z)] + m_yz[(0,z)])

                    loss = weighted_loss(logits, labels, v, False)

                    optimizer.zero_grad()
                    if not np.isnan(loss.item()): loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())

                loss_avg = sum(batch_loss) / len(batch_loss)
                train_loss.append(loss_avg)

                # Calculate avg training accuracy over all clients at every round
                list_acc = []
                # the number of samples which are assigned to class y and belong to the sensitive group z
                n_yz, loss_yz = {}, {}
                for y in [0,1]:
                    for z in range(self.Z):
                        n_yz[(y,z)] = 0
                        loss_yz[(y,z)] = 0

                self.model.eval()
                acc, loss, n_yz, acc_loss, fair_loss, loss_yz = self.inference(model = self.model, option = 'FairBatch')
                list_acc.append(acc)

                if self.prn and (round_+1) % 50 == 0: print("Accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                    acc_loss, fair_loss, self.metric, self.disparity(n_yz)))

                for y, z in loss_yz:
                    loss_yz[(y,z)] = loss_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

                for z in range(self.Z):
                    if z == 0:
                        lbd[(0,z)] -= alpha / (round_ + 1) ** .5 * sum([(loss_yz[(0,0)] + loss_yz[(1,0)] - loss_yz[(0,z)] - loss_yz[(1,z)]) for z in range(self.Z)])
                        lbd[(0,z)] = lbd[(0,z)].item()
                        lbd[(0,z)] = max(0, min(lbd[(0,z)], 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset)))
                        lbd[(1,z)] = 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset) - lbd[(0,z)]
                    else:
                        lbd[(0,z)] += alpha / (round_ + 1) ** .5 * (loss_yz[(0,0)] + loss_yz[(1,0)] - loss_yz[(0,z)] - loss_yz[(1,z)])
                        lbd[(0,z)] = lbd[(0,z)].item()
                        lbd[(0,z)] = max(0, min(lbd[(0,z)], 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset)))
                        lbd[(1,z)] = 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset) - lbd[(0,z)]

                train_accuracy.append(sum(list_acc)/len(list_acc))

                # print global training loss after every 'i' rounds
                if self.prn and (round_+1) % 50 == 0:
                    if (round_+1) % self.print_every == 0:
                        print(f' \nAvg Training Stats after {round_+1} global rounds:')
                        print("Training loss: %.2f | Training accuracy: %.2f%% | Training %s: %.8f" % (
                            np.mean(np.array(train_loss)),
                            100*train_accuracy[-1], self.metric, self.disparity(n_yz)))

                if self.trial:
                    with tune.checkpoint_dir(round_) as checkpoint_dir:
                        path = os.path.join(checkpoint_dir, "checkpoint")
                        torch.save(self.model.state_dict(), path)

                    tune.report(loss = loss, accuracy = train_accuracy[-1], disp = self.disparity(n_yz), iteration = round_+1)

            # Test inference after completion of training
            test_acc, n_yz = self.test_inference(self.model, self.test_dataset)
            rd = self.disparity(n_yz)

            if self.prn:
                print(f' \n Results after {outer_rounds} global rounds of training:')
                print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
                print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

                # Compute fairness metric
                print("|---- Test "+ self.metric+": {:.8f}".format(rd))

                print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

            if self.ret: return test_acc, rd, self.model

    def UFLFB(self, num_epochs = 450, learning_rate = (0.005, 0.005, 0.005), alpha = (0.08,0.1,0.1), optimizer = 'adam'):
        models = []
        results = []
        ind = HV(ref_point=[1, 1])

        for c in range(self.num_clients):
            local_model = Client(dataset=self.train_dataset,
                                idxs=self.clients_idx[c], batch_size = self.batch_size, option = "FB-Variant1",
                                seed = self.seed, prn = self.train_prn, Z = self.Z)
            result, model, _ = local_model.uflfb_update(copy.deepcopy(self.model).to(DEVICE), num_epochs, learning_rate[c], optimizer, alpha[c])
            models.append(model)
            results.append(result)

        # # Test inference after completion of training
        # test_acc, n_yz = self.ufl_inference(models, test_dataset=self.test_dataset)
        Local_hv = 0
        Local_result = []
        for c in range(self.num_clients):
            try:
                test_acc, n_yz = self.ufl_inference(models=models, test_dataset=self.test_dataset,
                                                     idxs=self.test_clients_idx[c] - len(self.train_dataset),
                                                     local=True)
            except:
                test_acc, n_yz = self.ufl_inference(models=models, test_dataset=self.test_dataset,
                                                     idxs=self.test_clients_idx[c], local=True)
            rd_dp = self.disparity(n_yz)
            Local_hv += test_acc * (1 - rd_dp)
            Local_result.append([1 - test_acc, rd_dp])

        print(f'Local Test dataset Client HV: {Local_hv / 3}')

        test_acc, n_yz = self.test_inference(model=self.model, test_dataset=self.test_dataset,idxs=self.test_clients_idx,
                                                     local=False)
        rd = self.disparity(n_yz)
        print(f'Global Test dataset Client HV: {test_acc*(1-rd)}')

        return Local_result, [1 - test_acc, rd], None, None

    def FFLFB(self, num_rounds = 15, local_epochs = 30, learning_rate = 0.005, optimizer = 'adam', alpha = (.000001,.0000001,.0000001)):
        # new algorithm for demographic parity, add weights directly, signed gradient-based algorithm
        # set seed
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        ind = HV(ref_point=[1, 1])
        best_hv = 0
        # Training
        train_loss, train_accuracy, train_accloss, train_fairloss, train_fair = [], [], [], [], []
        Results = []
        start_time = time.time()
        weights = self.model.state_dict()
        if self.select_round: best_fairness = float('inf')

        lbd, m_yz, nc = [None for _ in range(self.num_clients)], [None for _ in range(self.num_clients)], [None for _ in range(self.num_clients)]

        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses = [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            self.model.train()

            for idx in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx],
                            batch_size = self.batch_size, option = "FB-Variant1", seed = self.seed, prn = self.train_prn, Z = self.Z)

                w, loss, nc_, lbd_, m_yz_ = local_model.local_fb(
                                model=copy.deepcopy(self.model),
                                    learning_rate = learning_rate, local_epochs = local_epochs,
                                    optimizer = optimizer, alpha = alpha[idx], lbd = lbd[idx], m_yz = m_yz[idx])
                lbd[idx], m_yz[idx], nc[idx] = lbd_, m_yz_, nc_
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            weights = weighted_average_weights(local_weights, nc, sum(nc))
            self.model.load_state_dict(weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc, list_accloss, list_fairloss, list_fair = [],[],[],[]
            # the number of samples which are assigned to class y and belong to the sensitive group z
            n_yz, loss_yz = {}, {}
            for y in [0,1]:
                for z in range(self.Z):
                    n_yz[(y,z)] = 0
                    loss_yz[(y,z)] = 0
            train_time = time.time()-start_time
            self.model.eval()
            for c in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset,
                                            idxs=self.clients_idx[c], batch_size = self.batch_size, option = "FB-Variant1",
                                            seed = self.seed, prn = self.train_prn, Z = self.Z)
                # validation dataset inference
                acc, loss, n_yz_c, acc_loss, fair_loss, loss_yz_c = local_model.inference_m(model = self.model, truem_yz=m_yz)


                for yz in n_yz:
                    n_yz[yz] += n_yz_c[yz]
                    loss_yz[yz] += loss_yz_c[yz]
                list_acc.append(acc)
                list_fair.append(self.disparity(n_yz))
                list_accloss.append(acc_loss)
                list_fairloss.append(fair_loss)
                if self.prn: print("Client %d: accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                    c+1, acc_loss, fair_loss, self.metric, self.disparity(n_yz_c)))

            train_accuracy.append(sum(list_acc) / len(list_acc))
            train_accloss.append(sum(list_accloss) / len(list_accloss))
            train_fairloss.append(sum(list_fairloss) / len(list_fairloss))
            train_fair.append(sum(list_fair) / len(list_fair))
            Results.append([1-train_accuracy[-1], train_fair[-1], train_accloss[-1], train_fairloss[-1], train_time])

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation %s: %.8f" % (
                        np.mean(np.array(train_loss)),
                        100*train_accuracy[-1], self.metric, self.disparity(n_yz)))

            if self.trial:
                with tune.checkpoint_dir(round_) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(self.model.state_dict(), path)

                tune.report(loss = loss, accuracy = train_accuracy[-1], disp = self.disparity(n_yz), iteration = round_+1)
            current_hv = ind(np.array([1-train_accuracy[-1], self.disparity(n_yz)]))
            if current_hv > best_hv:
                best_hv = current_hv
                test_model = copy.deepcopy(self.model.state_dict())

        # Test inference after completion of training
        self.model.load_state_dict(test_model)
        Local_hv = 0
        Local_result = []
        for c in range(self.num_clients):
            try:
                test_acc, n_yz = self.test_inference(model=self.model, test_dataset=self.test_dataset,
                                                     idxs=self.test_clients_idx[c] - len(self.train_dataset),
                                                     local=True)
            except:
                test_acc, n_yz = self.test_inference(model=self.model, test_dataset=self.test_dataset,
                                                     idxs=self.test_clients_idx[c], local=True)
            rd_dp = self.disparity(n_yz)
            Local_hv += test_acc * (1 - rd_dp)
            Local_result.append([1 - test_acc, rd_dp])

        print(f'Local Test dataset Client HV: {Local_hv / 3}')

        test_acc, n_yz = self.test_inference(model=self.model, test_dataset=self.test_dataset,idxs=self.test_clients_idx,
                                                     local=False)
        rd = self.disparity(n_yz)
        print(f'Global Test dataset Client HV: {test_acc*(1-rd)}')

        return Local_result, [1 - test_acc, rd], None, None

    # only support binary sensitive attribute
    # assign a higher weight to clients that have similar local fairness to the global fairness metric
    def FairFed(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, beta = 0.1, alpha = 0.01, optimizer = 'adam'):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        ind = HV(ref_point=[1, 1])
        best_hv = 0
        # Training
        train_loss, train_accuracy, train_accloss, train_fairloss, train_fair = [], [], [], [], []
        Results = []
        start_time = time.time()
        weights = self.model.state_dict()
        val_hv = []
        lbd, m_yz = [None for _ in range(self.num_clients)], [None for _ in range(self.num_clients)]

        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses, nw = [], [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            # get local fairness metric
            list_acc, list_accloss, list_fairloss, list_fair = [],[],[],[]
            n_yz = {}
            for y in [0,1]:
                for z in range(self.Z):
                    n_yz[(y,z)] = 0
            train_time = time.time()-start_time
            self.model.eval()
            for c in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c],
                            batch_size = self.batch_size, option = "unconstrained", seed = self.seed, prn = self.train_prn, Z = self.Z)
                acc, loss, n_yz_c, acc_loss, fair_loss, _ = local_model.inference(model = self.model, train = True)


                for yz in n_yz:
                    n_yz[yz] += n_yz_c[yz]
                list_acc.append(acc)
                list_fair.append(self.disparity(n_yz))
                list_accloss.append(acc_loss)
                list_fairloss.append(fair_loss)

                nw.append(self.disparity(n_yz_c))

                if self.prn:
                    print("Client %d: accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                            c+1, acc_loss, fair_loss, self.metric, self.disparity(n_yz_c)))

            train_accuracy.append(sum(list_acc) / len(list_acc))
            train_accloss.append(sum(list_accloss) / len(list_accloss))
            train_fairloss.append(sum(list_fairloss) / len(list_fairloss))
            train_fair.append(sum(list_fair) / len(list_fair))
            Results.append([1-train_accuracy[-1], train_fair[-1], train_accloss[-1], train_fairloss[-1],train_time])
            objs = np.array([1-train_accuracy[-1], train_fair[-1]])
            print('Current HV is ', ind(objs))
            val_hv.append(ind(objs))
            if ind(objs) > best_hv:
                best_hv = ind(objs)
                test_model = self.model
            for c in range(self.num_clients):
                nw[c] = np.exp(-beta * abs(nw[c] - self.disparity(n_yz))) * len(self.clients_idx[c]) / len(self.train_dataset)

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation %s: %.8f" % (
                        np.mean(np.array(train_loss)),
                        100*train_accuracy[-1], self.metric, self.disparity(n_yz)))

            if self.trial:
                with tune.checkpoint_dir(round_) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(self.model.state_dict(), path)

                tune.report(loss = loss, accuracy = train_accuracy[-1], disp = self.disparity(n_yz), iteration = round_+1)

            self.model.train()

            for idx in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx],
                            batch_size = self.batch_size, option = "FB-Variant1", seed = self.seed, prn = self.train_prn, Z = self.Z)

                w, loss, _, lbd_, m_yz_ = local_model.local_fb(
                                model=copy.deepcopy(self.model),
                                    learning_rate = learning_rate, local_epochs = local_epochs,
                                    optimizer = optimizer, alpha = alpha, lbd = lbd[idx], m_yz = m_yz[idx])
                lbd[idx], m_yz[idx] = lbd_, m_yz_
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            weights = weighted_average_weights(local_weights, nw, sum(nw))
            self.model.load_state_dict(weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

        # Test inference after completion of training
        self.model.load_state_dict(test_model.state_dict())
        Local_hv = 0
        Local_result = []
        for c in range(self.num_clients):
            try:
                test_acc, n_yz = self.test_inference(model=self.model, test_dataset=self.test_dataset,
                                                     idxs=self.test_clients_idx[c] - len(self.train_dataset),
                                                     local=True)
            except:
                test_acc, n_yz = self.test_inference(model=self.model, test_dataset=self.test_dataset,
                                                     idxs=self.test_clients_idx[c], local=True)
            rd_dp = self.disparity(n_yz)
            Local_hv += test_acc * (1 - rd_dp)
            Local_result.append([1 - test_acc, rd_dp])

        print(f'Local Test dataset Client HV: {Local_hv / 3}')

        test_acc, n_yz = self.test_inference(model=self.model, test_dataset=self.test_dataset,idxs=self.test_clients_idx,
                                                     local=False)
        rd = self.disparity(n_yz)
        print(f'Global Test dataset Client HV: {test_acc*(1-rd)}')

        return Local_result, [1 - test_acc, rd], val_hv, None

    def FAFL(self, num_epochs = 450, learning_rate = 0.005, penalty = 2):
        def loss_with_agnostic_fair(logits, targets, sensitive, sen_bar, larg = 1):

            acc_loss = F.cross_entropy(logits, targets, reduction = 'sum')
            fair_loss0 = torch.mul(sensitive - sen_bar, logits.T[0] - torch.mean(logits.T[0]))
            fair_loss0 = torch.mean(torch.mul(fair_loss0, fair_loss0))
            fair_loss1 = torch.mul(sensitive - sen_bar, logits.T[1] - torch.mean(logits.T[1]))
            fair_loss1 = torch.mean(torch.mul(fair_loss1, fair_loss1))
            fair_loss = fair_loss0 + fair_loss1

            return acc_loss, larg*fair_loss, acc_loss+larg*fair_loss
        best_hv = 0
        HV_ind = HV(ref_point=[1,1])
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        start_time = time.time()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        sen_bar = self.train_dataset.sen.mean()
        if self.select_round: best_fairness = float('inf')
        Results = []
        val_hv = []
        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            for _, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                sensitive = sensitive.to(DEVICE)
                _, Theta_X = self.model(features)

                _,_,loss = loss_with_agnostic_fair(Theta_X, labels, sensitive, sen_bar, penalty)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_time = time.time()-start_time
            self.model.eval()
            # validation dataset inference
            acc, loss, n_yz, acc_loss, fair_loss, _ = self.inference(model = self.model)

            if self.prn and (epoch+1) % 10 == 0:
                print("Accuracy loss: %.2f | fairness loss %.2f | Accuracy %.4f | %s = %.8f" % (
                acc_loss, fair_loss, acc, self.metric, self.disparity(n_yz)))


            # print global training loss after every 'i' rounds
            if self.prn and (epoch+1) % 10 == 0:
                if (epoch+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                    print("Training accuracy: %.4f%% | Training %s: %.8f" % (acc*100, self.metric, self.disparity(n_yz)))
                    Results.append([1-acc, self.disparity(n_yz), acc_loss, fair_loss,train_time])
            if self.trial:
                with tune.checkpoint_dir(epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(self.model.state_dict(), path)

                tune.report(loss = loss, accuracy = acc, disp = self.disparity(n_yz), iteration = epoch)
            current_hv = HV_ind(np.array([1-acc, self.disparity(n_yz)]))
            val_hv.append(current_hv)
            if current_hv > best_hv:
                best_hv = current_hv
                test_model = copy.deepcopy(self.model.state_dict())
            print("Validation | Current Best HV is ", best_hv)
        # Test inference after completion of training
        self.model.load_state_dict(test_model)
        Local_hv = 0
        Local_result = []
        for c in range(self.num_clients):
            try:
                test_acc, n_yz = self.test_inference(model=self.model, test_dataset=self.test_dataset,
                                                     idxs=self.test_clients_idx[c] - len(self.train_dataset),
                                                     local=True)
            except:
                test_acc, n_yz = self.test_inference(model=self.model, test_dataset=self.test_dataset,
                                                     idxs=self.test_clients_idx[c], local=True)
            rd_dp = self.disparity(n_yz)
            Local_hv += test_acc * (1 - rd_dp)
            Local_result.append([1 - test_acc, rd_dp])

        print(f'Local Test dataset Client HV: {Local_hv / 3}')

        test_acc, n_yz = self.test_inference(model=self.model, test_dataset=self.test_dataset,idxs=self.test_clients_idx,
                                                     local=False)
        rd = self.disparity(n_yz)
        print(f'Global Test dataset Client HV: {test_acc*(1-rd)}')

        return Local_result, [1 - test_acc, rd], val_hv, None

    def inference(self, option = 'unconstrained', penalty = 100, model = None, validloader = None):
        """
        Returns the inference accuracy,
                                loss,
                                N(sensitive group, pos),
                                N(non-sensitive group, pos),
                                N(sensitive group),
                                N(non-sensitive group),
                                acc_loss,
                                fair_loss
        """

        if model == None: model = self.model
        if validloader == None:
            validloader = self.validloader
        model.eval()
        loss, total, correct, fair_loss, acc_loss, num_batch = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        n_yz, loss_yz = {}, {}
        for y in [0,1]:
            for z in range(self.Z):
                loss_yz[(y,z)] = 0
                n_yz[(y,z)] = 0

        for _, (features, labels, sensitive) in enumerate(validloader):
            features, labels = features.to(DEVICE), labels.type(torch.LongTensor).to(DEVICE)
            sensitive = sensitive.type(torch.LongTensor).to(DEVICE)

            # Inference
            outputs, logits = model(features)
            outputs, logits = outputs.to(DEVICE), logits.to(DEVICE)

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1).to(DEVICE)
            bool_correct = torch.eq(pred_labels, labels)
            correct += torch.sum(bool_correct).item()
            total += len(labels)
            num_batch += 1

            group_boolean_idx = {}

            for yz in n_yz:
                group_boolean_idx[yz] = (labels == yz[0]) & (sensitive == yz[1])
                n_yz[yz] += torch.sum((pred_labels == yz[0]) & (sensitive == yz[1])).item()

                if option == "FairBatch":
                # the objective function have no lagrangian term

                    loss_yz_,_,_ = loss_func("FB_inference", logits[group_boolean_idx[yz]].to(DEVICE),
                                                    labels[group_boolean_idx[yz]].to(DEVICE),
                                         outputs[group_boolean_idx[yz]].to(DEVICE), sensitive[group_boolean_idx[yz]].to(DEVICE),
                                         penalty)
                    loss_yz[yz] += loss_yz_

            batch_loss, batch_acc_loss, batch_fair_loss = loss_func(option, logits,
                                                        labels, outputs, sensitive, penalty)
            loss, acc_loss, fair_loss = (loss + batch_loss.item(),
                                         acc_loss + batch_acc_loss.item(),
                                         fair_loss + batch_fair_loss.item())
        accuracy = correct/total
        if option in ["FairBatch", "FB-Variant1"]:
            return accuracy, loss, n_yz, acc_loss / num_batch, fair_loss / num_batch, loss_yz
        else:
            return accuracy, loss, n_yz, acc_loss / num_batch, fair_loss / num_batch, None

    def test_inference(self, model = None, test_dataset = None, idxs=None, local=True):

        """
        Returns the test accuracy and fairness level.
        """
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        if model == None: model = self.model
        if test_dataset == None:
            test_dataset = self.test_dataset

        model.eval()
        total, correct = 0.0, 0.0
        n_yz = {}
        for y in [0,1]:
            for z in range(self.Z):
                n_yz[(y,z)] = 0

        if local:
            data = DatasetSplit(test_dataset, idxs)
            testloader = DataLoader(data, batch_size=self.batch_size)
        else:
            testloader = DataLoader(test_dataset, batch_size=self.batch_size,
                                    shuffle=False)



        for _, (features, labels, sensitive) in enumerate(testloader):
            features = features.to(DEVICE)
            labels =  labels.to(DEVICE)
            # Inference
            outputs, _ = model(features)

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            bool_correct = torch.eq(pred_labels, labels)
            correct += torch.sum(bool_correct).item()
            total += len(labels)

            for y,z in n_yz:
                n_yz[(y,z)] += torch.sum((sensitive == z) & (pred_labels.detach().cpu() == y)).item()

        accuracy = correct/total

        return accuracy, n_yz

    def test_inference_pf(self, pref, model=None, test_dataset=None):

        """
        Returns the test accuracy and fairness level.
        """
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        if model == None: model = self.model
        if test_dataset == None: test_dataset = self.test_dataset

        model.eval()


        testloader = DataLoader(test_dataset, batch_size=self.batch_size,
                                shuffle=False)

        ACC_list = []
        Fair_list = []
        for ray in pref:
            total, correct = 0.0, 0.0
            n_yz = {}
            for y in [0, 1]:
                for z in range(self.Z):
                    n_yz[(y, z)] = 0
            for _, (features, labels, sensitive) in enumerate(testloader):
                features = features.to(DEVICE)
                labels = labels.to(DEVICE)
                # Inference
                outputs, _ = model(features, ray.reshape(1,-1))

                # Prediction
                _, pred_labels = torch.max(outputs[0], 1)
                pred_labels = pred_labels.view(-1)
                bool_correct = torch.eq(pred_labels, labels)
                correct += torch.sum(bool_correct).item()
                total += len(labels)

                for y, z in n_yz:
                    n_yz[(y, z)] += torch.sum((sensitive == z) & (pred_labels.detach().cpu() == y)).item()

            accuracy = correct / total
            ACC_list.append(accuracy)
            Fair_list.append(self.disparity(n_yz))

        return ACC_list, Fair_list

    def ufl_inference(self, models, test_dataset = None, idxs=None, local=True):
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        if test_dataset == None:
            test_dataset = self.test_dataset

        total, correct = 0.0, 0.0
        n_yz = {}
        for y in [0,1]:
            for z in range(self.Z):
                n_yz[(y,z)] = 0
        if local:
            data = DatasetSplit(test_dataset, idxs)
            testloader = DataLoader(data, batch_size=self.batch_size)
        else:
            testloader = DataLoader(test_dataset, batch_size=self.batch_size,
                                    shuffle=False)

        for model in models:
            model.eval()

        # testloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        for _, (features, labels, sensitive) in enumerate(testloader):
            features = features.to(DEVICE)
            labels =  labels.type(torch.LongTensor).to(DEVICE)
            sensitive = sensitive.to(DEVICE)

            # Inference
            outputs = torch.zeros((len(labels),2)).to(DEVICE)
            for c in range(self.num_clients):
                output, _ = models[c](features)
                output = output/output.sum()
                outputs += output * len(self.clients_idx[c])
            outputs = outputs / np.array(list(map(len, self.clients_idx))).sum()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1).to(DEVICE)
            bool_correct = torch.eq(pred_labels, labels)
            correct += torch.sum(bool_correct).item()
            total += len(labels)

            for y,z in n_yz:
                n_yz[(y,z)] += torch.sum((sensitive == z) & (pred_labels == y)).item()

        accuracy = correct/total

        return accuracy, n_yz

class Client(object):
    def __init__(self, dataset, idxs, batch_size, option, seed = 0, prn = True, penalty = 500, Z = 2, test=False, local=True):
        self.seed = seed
        self.dataset = dataset
        self.idxs = idxs
        self.option = option
        self.prn = prn
        self.Z = Z
        if test:
            if local:
                self.validloader = DatasetSplit(dataset, idxs)
            else:
                self.validloader = dataset
            self.validloader = DataLoader(self.validloader,
                                     batch_size=128, shuffle=False)

        else:
            self.trainloader, self.validloader = self.train_val(dataset, list(idxs), batch_size)
        self.penalty = penalty
        self.disparity = DPDisparity

    def train_val(self, dataset, idxs, batch_size):
        """
        Returns train, validation for a given local training dataset
        and user indexes.
        """
        torch.manual_seed(self.seed)

        # split indexes for train, validation (90, 10)
        idxs_train = idxs[:int(0.9*len(idxs))]
        idxs_val = idxs[int(0.9*len(idxs)):len(idxs)]

        self.train_dataset = DatasetSplit(dataset, idxs_train)
        self.test_dataset = DatasetSplit(dataset, idxs_val)

        trainloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        validloader = DataLoader(self.test_dataset,
                                     batch_size=max(int(len(idxs_val)/10),10), shuffle=False)
        return trainloader, validloader

    def standard_update(self, model, global_round, learning_rate, local_epochs, optimizer):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set optimizer for the local updates
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                        ) # momentum=0.5
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)
        for i in range(local_epochs):
            batch_loss = []
            for batch_idx, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                sensitive = sensitive.to(DEVICE)
                # we need to set the gradients to zero before starting to do backpropragation
                # because PyTorch accumulates the gradients on subsequent backward passes.
                # This is convenient while training RNNs

                probas, logits = model(features)
                loss, _, fairloss = loss_func(self.option, logits, labels, probas, sensitive, self.penalty)
                Total = loss + fairloss
                optimizer.zero_grad()
                Total.backward()
                optimizer.step()

                if self.prn and (100. * batch_idx / len(self.trainloader)) % 50 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                        global_round + 1, i, batch_idx * len(features),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # weight, loss
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def Peasonfair_update(self, model, global_round, learning_rate, local_epochs,collected_pref, optimizer, idxs_user, z_min, z_max, Z, pref_bs, local_times):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set optimizer for the local updates
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.base.parameters(), lr=learning_rate,
                                        ) # momentum=0.5
            optimizer_pf = torch.optim.SGD(model.head.parameters(), lr=learning_rate,
                                        ) # momentum=0.5
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.base.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)
            optimizer_pf = torch.optim.Adam(model.head.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)
        for param in model.base.parameters():
            param.requires_grad = True
        for param in model.head.parameters():
            param.requires_grad = False
        OPTIM = optimizer
        Check = True
        ######training hypernetwork######
        test_pref = np.stack([np.linspace(0, 1, 101), 1 - np.linspace(0, 1, 101)]).T
        test_pref = torch.tensor(test_pref, dtype=torch.float32).to(DEVICE)
        for i in range(local_epochs):
            batch_loss = []
            for batch_idx, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                sensitive = sensitive.to(DEVICE)
                # we need to set the gradients to zero before starting to do backpropragation
                # because PyTorch accumulates the gradients on subsequent backward passes.
                # This is convenient while training RNNs
                pref = np.array([[0.5]*2])

                if i > local_times:
                    pref = np.random.dirichlet([1, 1], pref_bs)
                    collected_pref.append(list(pref))
                    if Check:
                        OPTIM = optimizer_pf
                        for param in model.base.parameters():
                            param.requires_grad = False
                        for param in model.head.parameters():
                            param.requires_grad = True
                        Check = False
                    pref = torch.tensor(pref, dtype=torch.float32).to(DEVICE)
                    probas, logits = model(features, pref=pref)

                    # print(logits.shape)
                    acc_loss, fair_loss = tch_loss(logits, labels, probas, sensitive, pref, i, global_round, self.penalty)
                    objs = torch.stack((((acc_loss) * (1/pref[:,0])),
                                      ((fair_loss) * (1/pref[:,1])) ), dim=1)

                    loss = torch.logsumexp(objs, axis=1).mean()

                    # loss = torch.max(objs, axis=1)[0].mean()

                else:
                    pref = torch.tensor(pref, dtype=torch.float32).to(DEVICE)
                    probas, logits = model(features, pref=pref)
                    # acc_loss, fair_loss = tch_loss(self.option, logits, labels, probas, sensitive, pref, self.penalty)
                    loss, _, _ = loss_func(self.option, logits[0], labels, probas[0], sensitive, self.penalty)

                OPTIM.zero_grad()
                loss.backward()

                OPTIM.step()
                if self.prn and (100. * batch_idx / len(self.trainloader)) % 50 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                        global_round + 1, i, batch_idx * len(features),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # weight, loss
        return model.base.state_dict(), model.head.state_dict(), sum(epoch_loss) / len(epoch_loss), z_min, z_max, collected_pref

    def GL_update(self, model, global_round, learning_rate, Alpha_, scheduler, collected_pref, alpha_optim, local_epochs, optimizer, idxs_user, z_min, z_max,
                  Z, pref_bs, local_times, adaptive_dist):
        # Set mode to train model
        model.train()
        epoch_loss = []
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Set optimizer for the local updates
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.base.parameters(), lr=learning_rate,
                                        ) # momentum=0.5
            optimizer_pf = torch.optim.SGD(model.head.parameters(), lr=learning_rate,
                                        ) # momentum=0.5
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.base.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)
            optimizer_pf = torch.optim.Adam(model.head.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)
        for param in model.base.parameters():
            param.requires_grad = True
        for param in model.head.parameters():
            param.requires_grad = False
        Check = True
        OPTIM = optimizer
        ######training hypernetwork######
        reference_point_ = torch.tensor([1, 1]).to(DEVICE)
        Best_HV = [0 for _ in range(len(self.train_dataset))]
        for i in range(local_epochs):
            batch_loss = []
            for batch_idx, (features, labels, sensitive) in enumerate(self.trainloader):
                dirichlet = torch.distributions.Dirichlet(Alpha_.to(DEVICE))
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                sensitive = sensitive.to(DEVICE)
                if i > local_times:
                    if Check:
                        OPTIM = optimizer_pf
                        for param in model.base.parameters():
                            param.requires_grad = False
                        for param in model.head.parameters():
                            param.requires_grad = True
                        Check = False
                    if adaptive_dist is not None:
                        pref = np.random.dirichlet(list(Alpha_.detach().numpy()), pref_bs)
                    else:
                        pref = np.random.dirichlet([1, 1], pref_bs)
                    collected_pref.append(list(pref))
                    pref = torch.tensor(pref, dtype=torch.float32).to(DEVICE)
                    probas, logits = model(features, pref=pref)

                    # print(logits.shape)
                    acc_loss, fair_loss = tch_loss(logits, labels, probas, sensitive, pref, i, global_round, self.penalty)
                    objs = torch.stack((acc_loss * (1/(pref[:,0])),
                                      fair_loss * (1/(pref[:,1]))), dim=1)
                    loss = torch.logsumexp(objs, axis=1).mean()

                    #################################################################
                    # print(reference_point)
                    with torch.no_grad():
                        Collect_loss = torch.stack((acc_loss, fair_loss)).T

                        reference_point, _ = torch.max(Collect_loss, dim=0)

                        reference_point_, _ = torch.max(torch.cat((reference_point.reshape(1, -1), reference_point_.reshape(1, -1)), dim=0), dim=0)

                        unique_rows, indices = torch.unique(Collect_loss, dim=0, return_inverse=True)
                        original_indices = []
                        other_indices = []
                        for unique_row in unique_rows:
                            # 查找与唯一行相等的所有行的索引
                            idx = torch.where((Collect_loss == unique_row).all(dim=1))[0]
                            if idx.numel() > 0:  # 确保有匹配项
                                similarities = []
                                for i in idx:
                                    # 计算余弦相似度
                                    similarity = F.cosine_similarity((pref[i]).unsqueeze(0), Collect_loss[i].unsqueeze(0))
                                    similarities.append(similarity)
                                highest_index = idx[torch.stack(similarities).argmax()].item()
                                other_indice = idx[idx != highest_index]
                                original_indices.append(highest_index)  # 取第一个匹配项并转换为 Python 的 int
                                for j in other_indice:
                                    other_indices.append(j.item())

                        Pareto_data =  Collect_loss[original_indices]
                        pref = pref[original_indices]

                        HV_value = hypervolume_2d(Pareto_data, reference_point=reference_point_)
                        HVC = hypervolume_contribution_pytorch(Pareto_data, reference_point=reference_point_).to(DEVICE)

                    log_prob = dirichlet.log_prob(pref)
                    grad_hvc = torch.mean((-HVC) * log_prob)
                    alpha_optim.zero_grad()
                    grad_hvc.backward()
                    alpha_optim.step()
                    with torch.no_grad():  # 禁用梯度计算
                        # Alpha_.clamp_(0.5, 5)  # 直接在原地修改 Alpha
                        Alpha_.clamp_(1, 5)

                else:
                    pref = np.array([[0.5] * 2])
                    pref = torch.tensor(pref, dtype=torch.float32).to(DEVICE)
                    probas, logits = model(features, pref=pref)
                    # acc_loss, fair_loss = tch_loss(self.option, logits, labels, probas, sensitive, pref, self.penalty)
                    loss, _, _ = loss_func(self.option, logits[0], labels, probas[0], sensitive, self.penalty)
                #################################################################
                OPTIM.zero_grad()
                loss.backward()
                OPTIM.step()
                if self.prn and (100. * batch_idx / len(self.trainloader)) % 50 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                        global_round + 1, i, batch_idx * len(features),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        # weight, loss
        return model.base.state_dict(), model.head.state_dict(), sum(epoch_loss) / len(epoch_loss), z_min, z_max, collected_pref, Alpha_


    def fb_update(self, model, global_round, learning_rate, local_epochs, optimizer, lbd, m_yz):
        # Set mode to train model
        model.train()
        epoch_loss = []
        nc = 0

        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set optimizer for the local updates
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                        momentum=0.5) #
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)
        for i in range(local_epochs):
            batch_loss = []
            for batch_idx, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.type(torch.LongTensor).to(DEVICE)
                sensitive = sensitive.to(DEVICE)
                _, logits = model(features)

                logits = logits.to(DEVICE)
                v = torch.randn(len(labels)).type(torch.DoubleTensor).to(DEVICE)

                group_idx = {}

                for y, z in lbd:
                    group_idx[(y,z)] = torch.where((labels == y) & (sensitive == z))[0]
                    v[group_idx[(y,z)]] = lbd[(y,z)] * sum([m_yz[(y,z)] for z in range(self.Z)]) / m_yz[(y,z)]
                    nc += v[group_idx[(y,z)]].sum().item()

                # print(logits)
                loss = weighted_loss(logits, labels, v)
                # if global_round == 1: print(loss)

                optimizer.zero_grad()
                if not np.isnan(loss.item()): loss.backward()
                optimizer.step()

                if self.prn and (100. * batch_idx / len(self.trainloader)) % 50 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                        global_round + 1, i, batch_idx * len(features),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # weight, loss
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), nc

    def fb2_update(self, model, global_round, learning_rate, local_epochs, optimizer, lbd, m_yz):
        # Set mode to train model
        model.train()
        epoch_loss = []
        nc = 0

        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set optimizer for the local updates
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                        momentum=0.5) #
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)
        for i in range(local_epochs):
            batch_loss = []
            for batch_idx, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.type(torch.LongTensor).to(DEVICE)
                sensitive = sensitive.to(DEVICE)
                _, logits = model(features)

                v = torch.ones(len(labels)).type(torch.DoubleTensor)

                group_idx = {}
                for y, z in lbd:
                    group_idx[(y,z)] = torch.where((labels == y) & (sensitive == z))[0].cpu()
                    v[group_idx[(y,z)]] = lbd[(y,z)] / (m_yz[(1,z)] + m_yz[(0,z)])
                    nc += v[group_idx[(y,z)]].sum().item()

                loss = weighted_loss(logits, labels, v.to(DEVICE), False)

                optimizer.zero_grad()
                if not np.isnan(loss.item()): loss.backward()
                optimizer.step()

                if self.prn and (100. * batch_idx / len(self.trainloader)) % 50 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                        global_round + 1, i, batch_idx * len(features),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # weight, loss
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), nc

    def uflfb_update(self, model, num_epochs, learning_rate, optimizer, alpha):
        start_time = time.time()
        best_hv = 0

        ref_point = np.array([1, 1])
        ind = HV(ref_point=ref_point)
        val_hv = []
        if self.Z == 2:
            # set seed
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            # Training
            train_loss, train_accuracy, train_accloss, train_fairloss, train_fair = [], [], [], [], []
            Results = []
            # Set optimizer for the local updates
            if optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                            momentum=0.5) #
            elif optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                            weight_decay=1e-4)

            # the number of samples whose label is y and sensitive attribute is z
            m_yz, lbd = {}, {}
            for y in [0,1]:
                for z in range(self.Z):
                    m_yz[(y,z)] = ((self.train_dataset.y == y) & (self.train_dataset.sen == z)).sum()

            for y in [0,1]:
                for z in range(self.Z):
                    lbd[(y,z)] = m_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

            for round_ in tqdm(range(num_epochs)):
                if self.prn and (round_+1) % 50 == 0:
                    print(f'\n | Global Training Round : {round_+1} |\n')

                model.train()
                batch_loss = []
                for _, (features, labels, sensitive) in enumerate(self.trainloader):
                    features, labels = features.to(DEVICE), labels.type(torch.LongTensor).to(DEVICE)
                    sensitive = sensitive.to(DEVICE)
                    _, logits = model(features)

                    v = torch.randn(len(labels)).type(torch.DoubleTensor).to(DEVICE)

                    group_idx = {}

                    for y, z in lbd:
                        group_idx[(y,z)] = torch.where((labels == y) & (sensitive == z))[0]
                        v[group_idx[(y,z)]] = lbd[(y,z)] * sum([m_yz[(y,z)] for z in range(self.Z)]) / m_yz[(y,z)]
                    loss = weighted_loss(logits.to(DEVICE), labels, v)

                    optimizer.zero_grad()
                    if not np.isnan(loss.item()): loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())

                loss_avg = sum(batch_loss)/len(batch_loss)
                train_loss.append(loss_avg)

                # Calculate avg training accuracy over all clients at every round
                list_acc, list_accloss, list_fairloss, list_fair = [],[],[],[]
                # the number of samples which are assigned to class y and belong to the sensitive group z
                n_yz, loss_yz = {}, {}
                for y in [0,1]:
                    for z in range(self.Z):
                        n_yz[(y,z)] = 0
                        loss_yz[(y,z)] = 0
                train_time = time.time()-start_time
                model.eval()
                    # validation dataset inference
                acc, loss, n_yz, acc_loss, fair_loss, loss_yz = self.inference_m(model = model, truem_yz=m_yz)
                list_acc.append(acc)
                list_fair.append(self.disparity(n_yz))
                list_accloss.append(acc_loss)
                list_fairloss.append(fair_loss)
                if self.prn:
                    print("Accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                    acc_loss, fair_loss, "DP Disparity", self.disparity(n_yz)))

                # update the lambda according to the paper -> see Section A.1 of FairBatch
                # works well! The real batch size would be slightly different from the setting
                for y, z in loss_yz:
                    loss_yz[(y,z)] = loss_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

                y0_diff = loss_yz[(0,0)] - loss_yz[(0,1)]
                y1_diff = loss_yz[(1,0)] - loss_yz[(1,1)]
                if y0_diff > y1_diff:
                    lbd[(0,0)] -= alpha / (round_+1)
                    lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
                    lbd[(1,0)] = 1 - lbd[(0,0)]
                    lbd[(0,1)] += alpha / (round_+1)
                    lbd[(0,1)] = min(max(0, lbd[(0,1)]), 1)
                    lbd[(1,1)] = 1 - lbd[(0,1)]
                else:
                    lbd[(0,0)] += alpha / (round_+1)
                    lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
                    lbd[(0,1)] = 1 - lbd[(0,0)]
                    lbd[(1,0)] -= alpha / (round_+1)
                    lbd[(1,0)] = min(max(0, lbd[(1,0)]), 1)
                    lbd[(1,1)] = 1 - lbd[(1,0)]

                train_accuracy.append(sum(list_acc)/len(list_acc))
                train_accloss.append(sum(list_accloss)/len(list_accloss))
                train_fairloss.append(sum(list_fairloss) / len(list_fairloss))
                train_fair.append(sum(list_fair)/len(list_fair))
                Results.append([1-train_accuracy[-1], train_fair[-1], train_accloss[-1], train_fairloss[-1], train_time])
                objs = np.array([1-train_accuracy[-1], train_fair[-1]])
                val_hv.append(ind(objs))
                if ind(objs) > best_hv:
                    best_hv = ind(objs)
                    best_model = copy.deepcopy(model)
                # print global training loss after every 'i' rounds
                # if (round_+1) % 10 == 0:
                #     print(f' \nAvg Training Stats after {round_+1} global rounds:')
                #     print("Training loss: %.2f | Training accuracy: %.2f%% | Training %s: %.4f" % (
                #         np.mean(np.array(train_loss)),
                #         100*train_accuracy[-1], "DP Disparity", self.disparity(n_yz)))

        else:
            # new algorithm for demographic parity, add weights directly, signed gradient-based algorithm
            # set seed
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)

            # Training
            train_loss, train_accuracy, train_accloss, train_fairloss, train_fair = [], [], [], [], []
            Results = []

            # Set optimizer for the local updates
            if optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                            momentum=0.5) #
            elif optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                            weight_decay=1e-4)

            # the number of samples whose label is y and sensitive attribute is z
            m_yz, lbd = {}, {}
            for y in [0,1]:
                for z in range(self.Z):
                    m_yz[(y,z)] = ((self.train_dataset.y == y) & (self.train_dataset.sen == z)).sum()

            for y in [0,1]:
                for z in range(self.Z):
                    lbd[(y,z)] = (m_yz[(1,z)] + m_yz[(0,z)])/len(self.train_dataset)

            for round_ in tqdm(range(num_epochs)):
                if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

                model.train()
                batch_loss = []
                for _, (features, labels, sensitive) in enumerate(self.trainloader):
                    features, labels = features.to(DEVICE), labels.to(DEVICE)
                    sensitive = sensitive.to(DEVICE)
                    _, logits = model(features)

                    v = torch.ones(len(labels)).type(torch.DoubleTensor)

                    group_idx = {}
                    for y, z in lbd:
                        group_idx[(y,z)] = torch.where((labels == y) & (sensitive == z))[0]
                        v[group_idx[(y,z)]] = lbd[(y,z)] / (m_yz[(1,z)] + m_yz[(0,z)])

                    loss = weighted_loss(logits, labels, v.to(DEVICE), False)

                    optimizer.zero_grad()
                    if not np.isnan(loss.item()): loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())

                loss_avg = sum(batch_loss) / len(batch_loss)
                train_loss.append(loss_avg)

                # Calculate avg training accuracy over all clients at every round
                list_acc, list_accloss, list_fairloss, list_fair = [],[],[],[]
                # the number of samples which are assigned to class y and belong to the sensitive group z
                n_yz, loss_yz = {}, {}
                for y in [0,1]:
                    for z in range(self.Z):
                        n_yz[(y,z)] = 0
                        loss_yz[(y,z)] = 0
                train_time = time.time()-start_time
                model.eval()
                acc, loss, n_yz, acc_loss, fair_loss, loss_yz = self.inference_m(model = model, train = True)
                list_acc.append(acc)
                list_fair.append(self.disparity(n_yz))
                list_accloss.append(acc_loss)
                list_fairloss.append(fair_loss)

                if self.prn and (round_+1) % 10 == 0: print("Accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                    acc_loss, fair_loss, "DP Disparity", self.disparity(n_yz)))

                for y, z in loss_yz:
                    loss_yz[(y,z)] = loss_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

                for z in range(self.Z):
                    if z == 0:
                        lbd[(0,z)] -= alpha / (round_ + 1) ** .5 * sum([(loss_yz[(0,0)] + loss_yz[(1,0)] - loss_yz[(0,z)] - loss_yz[(1,z)]) for z in range(self.Z)])
                        lbd[(0,z)] = lbd[(0,z)].item()
                        lbd[(0,z)] = max(0, min(lbd[(0,z)], 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset)))
                        lbd[(1,z)] = 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset) - lbd[(0,z)]
                    else:
                        lbd[(0,z)] += alpha / (round_ + 1) ** .5 * (loss_yz[(0,0)] + loss_yz[(1,0)] - loss_yz[(0,z)] - loss_yz[(1,z)])
                        lbd[(0,z)] = lbd[(0,z)].item()
                        lbd[(0,z)] = max(0, min(lbd[(0,z)], 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset)))
                        lbd[(1,z)] = 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset) - lbd[(0,z)]

                train_accuracy.append(sum(list_acc)/len(list_acc))
                train_accloss.append(sum(list_accloss)/len(list_accloss))
                train_fairloss.append(sum(list_fairloss) / len(list_fairloss))
                train_fair.append(sum(list_fair)/len(list_fair))
                Results.append([1-train_accuracy[-1], train_fair[-1], train_accloss[-1], train_fairloss[-1],train_time])
                objs = np.array([1-train_accuracy[-1], train_fair[-1]])
                val_hv.append(ind(objs))
                if ind(objs) > best_hv:
                    best_hv = ind(objs)
                    best_model = copy.deepcopy(model)

        return Results, best_model, val_hv

    def local_fb(self, model, learning_rate, local_epochs, optimizer, alpha, lbd = None, m_yz = None):
        if self.Z == 2:
            # Set mode to train model
            epoch_loss = []
            nc = 0

            # set seed
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)

            # Set optimizer for the local updates
            if optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                            momentum=0.5) #
            elif optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                            weight_decay=1e-4)

            if lbd == None:
                m_yz, lbd = {}, {}
                for y in [0,1]:
                    for z in range(self.Z):
                        m_yz[(y,z)] = ((self.dataset.y == y) & (self.dataset.sen == z)).sum()

                for y in [0,1]:
                    for z in range(self.Z):
                        lbd[(y,z)] = m_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

            for epoch in range(local_epochs):
                model.train()
                batch_loss = []
                for _, (features, labels, sensitive) in enumerate(self.trainloader):
                    features, labels = features.to(DEVICE), labels.type(torch.LongTensor).to(DEVICE)
                    sensitive = sensitive.to(DEVICE)
                    _, logits = model(features)

                    v = torch.ones(len(labels)).type(torch.DoubleTensor)

                    group_idx = {}
                    for y, z in lbd:

                        group_idx[(y,z)] = (torch.where((labels == y) & (sensitive == z))[0]).cpu()
                        v[group_idx[(y,z)]] = lbd[(y,z)] / (m_yz[(1,z)] + m_yz[(0,z)])
                        nc += v[group_idx[(y,z)]].sum().item()

                    loss = weighted_loss(logits, labels, v.to(DEVICE), False)

                    optimizer.zero_grad()
                    if not np.isnan(loss.item()): loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

                model.eval()
                # validation dataset inference
                _, _, _, _, _, loss_yz = self.inference_m(model=model, train=True, truem_yz=m_yz)

                for y, z in loss_yz:
                    loss_yz[(y,z)] = loss_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])


                    y0_diff = loss_yz[(0,0)] - loss_yz[(0,1)]
                    y1_diff = loss_yz[(1,0)] - loss_yz[(1,1)]
                    if y0_diff > y1_diff:
                        lbd[(0,0)] -= alpha / (epoch+1)
                        lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
                        lbd[(1,0)] = 1 - lbd[(0,0)]
                        lbd[(0,1)] += alpha / (epoch+1)
                        lbd[(0,1)] = min(max(0, lbd[(0,1)]), 1)
                        lbd[(1,1)] = 1 - lbd[(0,1)]
                    else:
                        lbd[(0,0)] += alpha / (epoch+1)
                        lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
                        lbd[(0,1)] = 1 - lbd[(0,0)]
                        lbd[(1,0)] -= alpha / (epoch+1)
                        lbd[(1,0)] = min(max(0, lbd[(1,0)]), 1)
                        lbd[(1,1)] = 1 - lbd[(1,0)]
            # weight, loss
            return model.state_dict(), sum(epoch_loss) / len(epoch_loss), nc, lbd, m_yz

        else:
            epoch_loss = []
            nc = 0

            # set seed
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)

            # Set optimizer for the local updates
            if optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                            momentum=0.5) #
            elif optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                            weight_decay=1e-4)

            if lbd == None:
                m_yz, lbd = {}, {}
                for y in [0,1]:
                    for z in range(self.Z):
                        m_yz[(y,z)] = ((self.dataset.y == y) & (self.dataset.sen == z)).sum()

                for y in [0,1]:
                    for z in range(self.Z):
                        lbd[(y,z)] = m_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

            for i in range(local_epochs):
                batch_loss = []
                for batch_idx, (features, labels, sensitive) in enumerate(self.trainloader):
                    features, labels = features.to(DEVICE), labels.type(torch.LongTensor).to(DEVICE)
                    sensitive = sensitive.to(DEVICE)
                    _, logits = model(features)

                    v = torch.ones(len(labels)).type(torch.DoubleTensor)

                    group_idx = {}
                    for y, z in lbd:
                        group_idx[(y,z)] = torch.where((labels == y) & (sensitive == z))[0].cpu()
                        v[group_idx[(y,z)]] = lbd[(y,z)] / (m_yz[(1,z)] + m_yz[(0,z)])
                        nc += v[group_idx[(y,z)]].sum().item()

                    loss = weighted_loss(logits, labels, v.to(DEVICE), False)

                    optimizer.zero_grad()
                    if not np.isnan(loss.item()): loss.backward()
                    optimizer.step()

                    if self.prn and (100. * batch_idx / len(self.trainloader)) % 50 == 0:
                        print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                            global_round + 1, i, batch_idx * len(features),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader), loss.item()))
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

            model.eval()
            # validation dataset inference
            _, _, _, _, _, loss_yz = self.inference_m(model = model, train = True, truem_yz=m_yz)
            for y, z in loss_yz:
                loss_yz[(y,z)] = loss_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

            for z in range(self.Z):
                if z == 0:
                    lbd[(0,z)] -= alpha ** .5 * sum([(loss_yz[(0,0)] + loss_yz[(1,0)] - loss_yz[(0,z)] - loss_yz[(1,z)]) for z in range(self.Z)])
                    lbd[(0,z)] = lbd[(0,z)].item()
                    lbd[(0,z)] = max(0, min(lbd[(0,z)], 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset)))
                    lbd[(1,z)] = 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset) - lbd[(0,z)]
                else:
                    lbd[(0,z)] += alpha ** .5 * (loss_yz[(0,0)] + loss_yz[(1,0)] - loss_yz[(0,z)] - loss_yz[(1,z)])
                    lbd[(0,z)] = lbd[(0,z)].item()
                    lbd[(0,z)] = max(0, min(lbd[(0,z)], 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset)))
                    lbd[(1,z)] = 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset) - lbd[(0,z)]

            # weight, loss
            return model.state_dict(), sum(epoch_loss) / len(epoch_loss), nc, lbd, m_yz

    def inference_m(self, model, train = False, bits = False, truem_yz = None):
        """
        Returns the inference accuracy,
                                loss,
                                N(sensitive group, pos),
                                N(non-sensitive group, pos),
                                N(sensitive group),
                                N(non-sensitive group),
                                acc_loss,
                                fair_loss
        """

        model.eval()
        loss, total, correct, fair_loss, acc_loss, num_batch = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        n_yz, loss_yz, m_yz, f_z = {}, {}, {}, {}
        for y in [0,1]:
            for z in range(self.Z):
                loss_yz[(y,z)] = 0
                n_yz[(y,z)] = 0
                m_yz[(y,z)] = 0

        dataset = self.validloader if not train else self.trainloader
        for _, (features, labels, sensitive) in enumerate(dataset):
            features, labels = features.to(DEVICE), labels.type(torch.LongTensor).to(DEVICE)
            sensitive = sensitive.type(torch.LongTensor).to(DEVICE)

            # Inference
            outputs, logits = model(features)
            outputs, logits = outputs.to(DEVICE), logits.to(DEVICE)

            # Prediction

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1).to(DEVICE)
            bool_correct = torch.eq(pred_labels, labels)
            correct += torch.sum(bool_correct).item()
            total += len(labels)
            num_batch += 1

            group_boolean_idx = {}

            for yz in n_yz:
                group_boolean_idx[yz] = (labels == yz[0]) & (sensitive == yz[1])
                n_yz[yz] += torch.sum((pred_labels == yz[0]) & (sensitive == yz[1])).item()
                m_yz[yz] += torch.sum((labels == yz[0]) & (sensitive == yz[1])).item()

                if self.option in["FairBatch", "FB-Variant1"]:
                # the objective function have no lagrangian term

                    loss_yz_,_,_ = loss_func("standard", logits[group_boolean_idx[yz]].to(DEVICE),
                                                    labels[group_boolean_idx[yz]].to(DEVICE),
                                         outputs[group_boolean_idx[yz]].to(DEVICE), sensitive[group_boolean_idx[yz]].to(DEVICE),
                                         self.penalty)
                    loss_yz[yz] += loss_yz_

            batch_loss, batch_acc_loss, batch_fair_loss = loss_func(self.option, logits,
                                                        labels, outputs, sensitive, self.penalty)
            loss, acc_loss, fair_loss = (loss + batch_loss.item(),
                                         acc_loss + batch_acc_loss.item(),
                                         fair_loss + batch_fair_loss.item())
        accuracy = correct/total
        if self.option in ["FairBatch", "FB-Variant1"]:
            # for z in range(1, self.Z):
            #     f_z[z] = - loss_yz[(0,0)]/(truem_yz[(0,0)] + truem_yz[(1,0)]) + loss_yz[(1,0)]/(truem_yz[(0,0)] + truem_yz[(1,0)]) + loss_yz[(0,z)]/(truem_yz[(0,z)] + truem_yz[(1,z)]) - loss_yz[(1,z)]/(truem_yz[(0,z)] + truem_yz[(1,z)])
            # if bits:
            #     bins = np.linspace(-2, 2, 2**bits // (self.Z - 1))
            #     for z in range(1, self.Z):
            #         f_z[z] = bins[np.digitize(f_z[z].item(), bins)-1]
            return accuracy, loss, n_yz, acc_loss / num_batch, fair_loss / num_batch, loss_yz
        else:
            return accuracy, loss, n_yz, acc_loss / num_batch, fair_loss / num_batch, None

    def inference(self, model, train=False, bits=False, truem_yz=None):
        """
        Returns the inference accuracy,
                                loss,
                                N(sensitive group, pos),
                                N(non-sensitive group, pos),
                                N(sensitive group),
                                N(non-sensitive group),
                                acc_loss,
                                fair_loss
        """

        model.eval()
        loss, total, correct, fair_loss, acc_loss, num_batch = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        n_yz, loss_yz, m_yz, f_z = {}, {}, {}, {}
        for y in [0, 1]:
            for z in range(self.Z):
                loss_yz[(y, z)] = 0
                n_yz[(y, z)] = 0
                m_yz[(y, z)] = 0

        dataset = self.validloader if not train else self.trainloader
        for _, (features, labels, sensitive) in enumerate(dataset):
            features, labels = features.to(DEVICE), labels.type(torch.LongTensor).to(DEVICE)
            sensitive = sensitive.type(torch.LongTensor).to(DEVICE)

            # Inference
            outputs, logits = model(features)
            outputs, logits = outputs.to(DEVICE), logits.to(DEVICE)

            # Prediction

            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1).to(DEVICE)
            bool_correct = torch.eq(pred_labels, labels)
            correct += torch.sum(bool_correct).item()
            total += len(labels)
            num_batch += 1

            group_boolean_idx = {}

            for yz in n_yz:
                group_boolean_idx[yz] = (labels == yz[0]) & (sensitive == yz[1])
                n_yz[yz] += torch.sum((pred_labels == yz[0]) & (sensitive == yz[1])).item()
                m_yz[yz] += torch.sum((labels == yz[0]) & (sensitive == yz[1])).item()

                if self.option in ["FairBatch", "FB-Variant1"]:
                    # the objective function have no lagrangian term

                    loss_yz_, _, _ = loss_func("standard", logits[group_boolean_idx[yz]].to(DEVICE),
                                               labels[group_boolean_idx[yz]].to(DEVICE),
                                               outputs[group_boolean_idx[yz]].to(DEVICE),
                                               sensitive[group_boolean_idx[yz]].to(DEVICE),
                                               self.penalty)
                    loss_yz[yz] += loss_yz_

            batch_loss, batch_acc_loss, batch_fair_loss = loss_func(self.option, logits,
                                                                    labels, outputs, sensitive, self.penalty)
            loss, acc_loss, fair_loss = (loss + batch_loss.item(),
                                         acc_loss + batch_acc_loss.item(),
                                         fair_loss + batch_fair_loss.item())
        accuracy = correct / total
        if self.option in ["FairBatch", "FB-Variant1"]:
            for z in range(1, self.Z):
                f_z[z] = - loss_yz[(0,0)]/(truem_yz[(0,0)] + truem_yz[(1,0)]) + loss_yz[(1,0)]/(truem_yz[(0,0)] + truem_yz[(1,0)]) + loss_yz[(0,z)]/(truem_yz[(0,z)] + truem_yz[(1,z)]) - loss_yz[(1,z)]/(truem_yz[(0,z)] + truem_yz[(1,z)])
            if bits:
                bins = np.linspace(-2, 2, 2**bits // (self.Z - 1))
                for z in range(1, self.Z):
                    f_z[z] = bins[np.digitize(f_z[z].item(), bins)-1]
            return accuracy, loss, n_yz, acc_loss / num_batch, fair_loss / num_batch, f_z
        else:
            return accuracy, loss, n_yz, acc_loss / num_batch, fair_loss / num_batch, None


    def inference_pf(self, model, pref, train=False, bits=False, truem_yz=None, merged_model=None):
        """
        Returns the inference accuracy,
                                loss,
                                N(sensitive group, pos),
                                N(non-sensitive group, pos),
                                N(sensitive group),
                                N(non-sensitive group),
                                acc_loss,
                                fair_loss
        """

        model.eval()
        accuracy_, loss_, n_yz_, acc_loss_, fair_loss_, f_z_  = [], [], [], [], [], []
        dataset = self.validloader if not train else self.trainloader
        # pref = torch.tensor(das_dennis(100, 2), dtype=torch.float32).to(DEVICE)
        for ray in pref:
            loss, total, correct, fair_loss, acc_loss, num_batch = 0.0, 0.0, 0.0, 0.0, 0.0, 0
            n_yz, loss_yz, m_yz, f_z = {}, {}, {}, {}
            for y in [0, 1]:
                for z in range(self.Z):
                    loss_yz[(y, z)] = 0
                    n_yz[(y, z)] = 0
                    m_yz[(y, z)] = 0
            for _, (features, labels, sensitive) in enumerate(dataset):
                features, labels = features.to(DEVICE), labels.type(torch.LongTensor).to(DEVICE)
                sensitive = sensitive.type(torch.LongTensor).to(DEVICE)

                if merged_model is None:
                    # Inference
                    outputs, logits = model(features, ray.reshape(1,-1))

                    outputs, logits = outputs[0].to(DEVICE), logits[0].to(DEVICE)
                else:
                    outputs, logits = merged_model(features, ray.reshape(1, -1))
                    outputs, logits = outputs[0].to(DEVICE), logits[0].to(DEVICE)
                # Prediction

                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1).to(DEVICE)
                bool_correct = torch.eq(pred_labels, labels)
                correct += torch.sum(bool_correct).item()
                total += len(labels)
                num_batch += 1

                group_boolean_idx = {}

                for yz in n_yz:
                    group_boolean_idx[yz] = (labels == yz[0]) & (sensitive == yz[1])
                    n_yz[yz] += torch.sum((pred_labels == yz[0]) & (sensitive == yz[1])).item()
                    m_yz[yz] += torch.sum((labels == yz[0]) & (sensitive == yz[1])).item()

                    if self.option in ["FairBatch", "FB-Variant1"]:
                        # the objective function have no lagrangian term

                        loss_yz_, _, _ = loss_func("standard", logits[group_boolean_idx[yz]].to(DEVICE),
                                                   labels[group_boolean_idx[yz]].to(DEVICE),
                                                   outputs[group_boolean_idx[yz]].to(DEVICE),
                                                   sensitive[group_boolean_idx[yz]].to(DEVICE),
                                                   self.penalty)
                        loss_yz[yz] += loss_yz_

                batch_loss, batch_acc_loss, batch_fair_loss = loss_func(self.option, logits,
                                                                        labels, outputs, sensitive, self.penalty)
                loss, acc_loss, fair_loss = (loss + batch_loss.item(),
                                             acc_loss + batch_acc_loss.item(),
                                             fair_loss + batch_fair_loss.item())
            accuracy = correct / total
            accuracy_.append(accuracy)
            loss_.append(loss)
            n_yz_.append(n_yz)
            acc_loss_.append(acc_loss/num_batch)
            fair_loss_.append(fair_loss/ num_batch)

        if self.option in ["FairBatch", "FB-Variant1"]:
            for z in range(1, self.Z):
                f_z[z] = - loss_yz[(0,0)]/(truem_yz[(0,0)] + truem_yz[(1,0)]) + loss_yz[(1,0)]/(truem_yz[(0,0)] + truem_yz[(1,0)]) + loss_yz[(0,z)]/(truem_yz[(0,z)] + truem_yz[(1,z)]) - loss_yz[(1,z)]/(truem_yz[(0,z)] + truem_yz[(1,z)])
            if bits:
                bins = np.linspace(-2, 2, 2**bits // (self.Z - 1))
                for z in range(1, self.Z):
                    f_z[z] = bins[np.digitize(f_z[z].item(), bins)-1]
            return accuracy_, loss_, n_yz_, acc_loss_, fair_loss_, f_z, pref
        else:
            return accuracy_, loss_, n_yz_, acc_loss_, fair_loss_, None, pref

