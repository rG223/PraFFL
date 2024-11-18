from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import pandas as pd
from scipy.stats import multivariate_normal
import torch, random, copy, os

################## MODEL SETTING ########################
device_idx = torch.cuda.current_device()
# if torch.cuda.is_available():
#     DEVICE = torch.device('cuda')
# else:
#     DEVICE = torch.device('cpu')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


#########################################################

class LoadData(Dataset):
    def __init__(self, df, pred_var, sen_var):
        self.y = df[pred_var].values
        self.x = df.drop(pred_var, axis=1).values
        self.sen = df[sen_var].values

    def __getitem__(self, index):

        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.int64)
        self.sen = self.sen.astype(np.float32)

        return torch.tensor(self.x[index]), torch.tensor(self.y[index]), torch.tensor(self.sen[index])

    def __len__(self):
        return self.y.shape[0]


class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.x = self.dataset.x[self.idxs]
        self.y = self.dataset.y[self.idxs]
        self.sen = self.dataset.sen[self.idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        feature, label, sensitive = self.dataset[self.idxs[item]]
        return feature, label, sensitive
        # return self.x[item], self.y[item], self.sen[item]


class logReg(torch.nn.Module):
    """
    Logistic regression model.
    """

    def __init__(self, num_features, num_classes, seed=123):
        torch.manual_seed(seed)

        super().__init__()
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(num_features, num_classes)

    def forward(self, x):
        logits = self.linear(x.float())
        probas = torch.sigmoid(logits)
        return probas.type(torch.FloatTensor), logits


class mlp(torch.nn.Module):
    """
    Logistic regression model.
    """

    def __init__(self, num_features, num_classes, seed=123):
        torch.manual_seed(seed)

        super().__init__()
        self.num_classes = num_classes
        self.linear1 = torch.nn.Linear(num_features, 4)
        self.linear2 = torch.nn.Linear(4, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.linear1(x.float())
        out = self.relu(out)
        out = self.linear2(out)
        probas = torch.sigmoid(out)

        return probas, out

class PF_MLP(torch.nn.Module):
    """
    Logistic regression model.
    """

    def __init__(self, num_features, num_classes, seed=123):
        torch.manual_seed(seed)

        super().__init__()
        self.num_classes = num_classes
        self.global_model=torch.nn.Sequential(torch.nn.Linear(num_features, 4),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(4, 4),
                                        torch.nn.ReLU()
                                    )

    def forward(self, x):
        out = self.global_model(x.float())
        return out

class Hypernet(torch.nn.Module):
    def __init__(self, n_obj, num_classes, seed=123):
        torch.manual_seed(seed)
        super(Hypernet, self).__init__()

        self.ray_mlp = torch.nn.Sequential(
            torch.nn.Linear(n_obj, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 4),
        )
        setattr(self, "fc1", torch.nn.Linear(4, 4 * 4))
        setattr(self, "bias1", torch.nn.Linear(4, 4))
        setattr(self, "fc2", torch.nn.Linear(4, num_classes * 4))
        setattr(self, "bias2", torch.nn.Linear(4, 1))
    def forward(self, ray, x, idx=None):
        features = self.ray_mlp(ray)
        Probs = {}
        Out = {}

        for r in range(len(ray)):

            x1 = F.linear(x, weight=getattr(self, "fc1")(features[r]).reshape(-1, x.shape[-1]),
                         bias=getattr(self, "bias1")(features[r]).flatten())
            out = F.linear(x1, weight=getattr(self, "fc2")(features[r]).reshape(-1, x1.shape[-1]),
                           bias=getattr(self, "bias2")(features[r]).flatten())

            probas = torch.sigmoid(out)
            if idx is None:
                Probs[r] = probas
                Out[r] = out
            else:
                Probs[idx] = probas
                Out[idx] = out
        return Probs, Out

class W_generation(torch.nn.Module):

    def __init__(self, num_clients, base, head, hyper, seed=123):
        torch.manual_seed(seed)
        super().__init__()
        self.num_clients = num_clients
        self.base = base
        self.head = hyper

        self.head_model = head
        self.weight_model=torch.nn.Sequential(torch.nn.Linear(2, 16),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(16, self.num_clients), torch.nn.Sigmoid())
        self.Relu = torch.nn.ReLU()
    def forward(self, x, pref):
        feature = self.base(x.float())
        Probs = {}
        Out = {}
        for ray_idx, ray in enumerate(pref):
            merged_params = {name: torch.zeros_like(param) for name, param in self.head_model.named_parameters()}
            weight = self.weight_model(ray.reshape(1, -1))
            weight = weight / weight.sum(dim=1, keepdim=True)  # 归一化权重
            for idx in range(len(self.head)):
                for name, param in self.head[idx].items():
                    merged_params[name] += weight[0][idx] * param.clone()  # 使用 clone 确保没有原地操作
            ray1 = F.linear(ray, weight=merged_params['ray_mlp.0.weight'],
                          bias=merged_params['ray_mlp.0.bias'])
            ray1 = self.Relu(ray1)
            ray2 = F.linear(ray1, weight=merged_params['ray_mlp.2.weight'],
                          bias=merged_params['ray_mlp.2.bias'])

            linear1_weight = F.linear(ray2, weight=merged_params['fc1.weight'],
                          bias=merged_params['fc1.bias'])
            linear1_bias = F.linear(ray2, weight=merged_params['bias1.weight'],
                          bias=merged_params['bias1.bias'])
            linear2_weight = F.linear(ray2, weight=merged_params['fc2.weight'],
                          bias=merged_params['fc2.bias'])
            linear2_bias = F.linear(ray2, weight=merged_params['bias2.weight'],
                          bias=merged_params['bias2.bias'])

            linear_value1 = F.linear(feature, weight=linear1_weight.reshape(-1, feature.shape[-1]),
                                      bias = linear1_bias.flatten())

            out = F.linear(linear_value1, weight=linear2_weight.reshape(-1, linear_value1.shape[-1]),
                                      bias = linear2_bias.flatten())

            probas = torch.sigmoid(out)
            Probs[ray_idx] = probas
            Out[ray_idx] = out

        return Probs, Out


# split an original model into a base and a head
class BaseHeadSplit(torch.nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head

    def forward(self, x, pref, client_idx=None):
        out = self.base(x)
        probas, out = self.head(pref,out)
        return probas, out

def logit_compute(probas):
    return torch.log(probas / (1 - probas))


def riskDifference(n_yz, absolute=True):
    """
    Given a dictionary of number of samples in different groups, compute the risk difference.
    |P(Group1, pos) - P(Group2, pos)| = |N(Group1, pos)/N(Group1) - N(Group2, pos)/N(Group2)|
    """
    n_z1 = max(n_yz[(1, 1)] + n_yz[(0, 1)], 1)
    n_z0 = max(n_yz[(0, 0)] + n_yz[(1, 0)], 1)
    if absolute:
        return abs(n_yz[(1, 1)] / n_z1 - n_yz[(1, 0)] / n_z0)
    else:
        return n_yz[(1, 1)] / n_z1 - n_yz[(1, 0)] / n_z0


def pRule(n_yz):
    """
    Compute the p rule level.
    min(P(Group1, pos)/P(Group2, pos), P(Group2, pos)/P(Group1, pos))
    """
    return min(n_yz[(1, 1)] / n_yz[(1, 0)], n_yz[(1, 0)] / n_yz[(1, 1)])


def DPDisparity(n_yz, each_z=False):
    """
    Same metric as FairBatch. Compute the demographic disparity.
    max(|P(pos | Group1) - P(pos)|, |P(pos | Group2) - P(pos)|)
    """
    z_set = sorted(list(set([z for _, z in n_yz.keys()])))
    p_y1_n, p_y1_d, n_z = 0, 0, []
    for z in z_set:
        p_y1_n += n_yz[(1, z)]
        n_z.append(max(n_yz[(1, z)] + n_yz[(0, z)], 1))
        for y in [0, 1]:
            p_y1_d += n_yz[(y, z)]
    p_y1 = p_y1_n / p_y1_d

    if not each_z:
        return max([abs(n_yz[(1, z)] / n_z[z] - p_y1) for z in z_set])
    else:
        return [n_yz[(1, z)] / n_z[z] - p_y1 for z in z_set]


def is_dominated(a, b):
    """检查解 a 是否被解 b 支配"""
    return (a <= b).all() and (a < b).any()


def hypervolume_2d(solution_set, reference_point):
    """
    计算二维解集的 Pareto 最优解的超体积（hypervolume）。

    参数：
    - solution_set: (n_solutions, n_objectives) 维的 PyTorch 张量，表示解集
    - reference_point: (n_objectives,) 维的 PyTorch 张量，表示参考点

    返回：
    - Hypervolume: Pareto 解集的超体积值
    """
    # 初始化 mask，用于标记 Pareto 最优解
    pareto_mask = torch.ones(solution_set.shape[0], dtype=torch.bool)

    # Step 1: 识别 Pareto 最优解
    for i in range(solution_set.shape[0]):
        for j in range(solution_set.shape[0]):
            if i != j and is_dominated(solution_set[j], solution_set[i]):
                pareto_mask[i] = False
                break

    pareto_solutions = solution_set[pareto_mask]

    # Step 2: 根据 Pareto 解集计算 Hypervolume
    sorted_set = pareto_solutions[pareto_solutions[:, 0].argsort(descending=True)]
    total_hv = 0.0
    previous_value = reference_point

    for sol in sorted_set:
        width = previous_value[0] - sol[0]  # x 轴宽度
        height = reference_point[1] - sol[1]  # y 轴高度
        area = width * height
        total_hv += area
        previous_value = sol

    return total_hv


def hypervolume_contribution_pytorch(solution_set, reference_point):
    contributions = torch.zeros(len(solution_set))

    # 计算当前解集的超体积

    current_hv = hypervolume_2d(solution_set, reference_point)
    for i in range(len(solution_set)):
        modified_set = torch.cat((solution_set[:i], solution_set[i + 1:]), dim=0)
        modified_hv = hypervolume_2d(modified_set, reference_point)
        # 计算当前解的贡献
        # if current_hv == modified_hv:
        #     contributions[i] = - torch.abs((solution_set[i][0]-reference_point[0]) * (solution_set[i][1]-reference_point[1])) / 8
        # else:
        contributions[i] = current_hv - modified_hv
    return contributions

def EODisparity(n_eyz, each_z=False):
    """
    Equal opportunity disparity: max_z{|P(yhat=1|z=z,y=1)-P(yhat=1|y=1)|}

    Parameter:
    n_eyz: dictionary. #(yhat=e,y=y,z=z)
    """
    z_set = list(set([z for _, _, z in n_eyz.keys()]))
    if not each_z:
        eod = 0
        p11 = sum([n_eyz[(1, 1, z)] for z in z_set]) / sum([n_eyz[(1, 1, z)] + n_eyz[(0, 1, z)] for z in z_set])
        for z in z_set:
            try:
                eod_z = abs(n_eyz[(1, 1, z)] / (n_eyz[(0, 1, z)] + n_eyz[(1, 1, z)]) - p11)
            except ZeroDivisionError:
                if n_eyz[(1, 1, z)] == 0:
                    eod_z = 0
                else:
                    eod_z = 1
            if eod < eod_z:
                eod = eod_z
        return eod
    else:
        eod = []
        p11 = sum([n_eyz[(1, 1, z)] for z in z_set]) / sum([n_eyz[(1, 1, z)] + n_eyz[(0, 1, z)] for z in z_set])
        for z in z_set:
            try:
                eod_z = n_eyz[(1, 1, z)] / (n_eyz[(0, 1, z)] + n_eyz[(1, 1, z)]) - p11
            except ZeroDivisionError:
                if n_eyz[(1, 1, z)] == 0:
                    eod_z = 0
                else:
                    eod_z = 1
            eod.append(eod_z)
        return eod


def RepresentationDisparity(loss_z):
    return max(loss_z) - min(loss_z)


def accVariance(acc_z):
    return np.std(acc_z)


# def mutual_information(n_yz, u = 0):
#     # u = 0 : demographic parity

def average_weights(w, clients_idx, idx_users):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    num_samples = 0
    for i in range(1, len(w)):
        num_samples += len(clients_idx[idx_users[i]])
        for key in w_avg.keys():
            w_avg[key] += w[i][key] * len(clients_idx[idx_users[i]])

    for key in w_avg.keys():
        w_avg[key] = torch.div(w_avg[key], num_samples)
    return w_avg

def merge_weights(w, merge_weight):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for i in range(1, len(w)):
        for key in w_avg.keys():
            w_avg[key] += w[i][key] * merge_weight[i]
    return w_avg

def weighted_average_weights(w, nc, n):
    w_avg = copy.deepcopy(w[0])
    for i in range(1, len(w)):
        for key in w_avg.keys():
            w_avg[key] += w[i][key] * nc[i]

    for key in w_avg.keys():
        w_avg[key] = torch.div(w_avg[key], n)
    return w_avg


def loss_func(option, logits, targets, outputs, sensitive, larg=1):
    """
    Loss function.
    """

    acc_loss = F.cross_entropy(logits, targets, reduction='sum')
    fair_loss0 = torch.mul(sensitive - sensitive.type(torch.FloatTensor).mean(), logits.T[0] - torch.mean(logits.T[0]))
    fair_loss0 = torch.mean(torch.mul(fair_loss0, fair_loss0))
    fair_loss1 = torch.mul(sensitive - sensitive.type(torch.FloatTensor).mean(), logits.T[1] - torch.mean(logits.T[1]))
    fair_loss1 = torch.mean(torch.mul(fair_loss1, fair_loss1))
    fair_loss = fair_loss0 + fair_loss1

    if option == 'local zafar':
        return acc_loss + larg * fair_loss, acc_loss, larg * fair_loss
    elif option == 'FB_inference':
        # acc_loss = torch.sum(torch.nn.BCELoss(reduction = 'none')((outputs.T[1]+1)/2, torch.ones(logits.shape[0])))
        acc_loss = F.cross_entropy(logits, torch.ones(logits.shape[0]).type(torch.LongTensor).to(DEVICE),
                                   reduction='sum')
        return acc_loss, acc_loss, fair_loss
    else:
        return acc_loss, acc_loss, larg * fair_loss

def average_model_state_dicts(models_state_dicts):
    # 假设 models_state_dicts 是一个包含多个模型 state_dict 的列表
    average_state_dict = {}

    # 初始化所有参数的总和
    for key in models_state_dicts[0].keys():
        average_state_dict[key] = torch.zeros_like(models_state_dicts[0][key])

    # 累加每个模型的参数
    for idx in range(len(models_state_dicts)):
        for key in models_state_dicts[idx].keys():
            average_state_dict[key] += models_state_dicts[idx][key]

    # 计算平均值
    num_models = len(models_state_dicts)
    for key in average_state_dict.keys():
        average_state_dict[key] /= num_models

    return average_state_dict

def manual_log_prob_gradient(alpha, x):
    # 计算梯度
    grad = torch.zeros((len(x), len(alpha))).to(x.device)
    grad += torch.log(x)  # 第一项
    grad += torch.special.digamma(alpha)  # 第二项
    grad -= torch.special.digamma(torch.sum(alpha))  # 第三项

    return grad

def tch_loss(logits, targets, outputs, sensitive, pref_vec, epoch, round, larg=1):
    """
    Loss function.

    Same metric as FairBatch. Compute the demographic disparity.
    max(|P(pos | Group1) - P(pos)|, |P(pos | Group2) - P(pos)|)

    """


    Acc_Loss = []
    Fair_Loss = []

    for i in range(len(logits)):
        # cross_entropy = F.cross_entropy(logits[i], targets, reduction='none')
        # acc_loss = cross_entropy.mean()
        #
        # fair_term1 = (cross_entropy[sensitive == 0]).mean()
        # fair_term0 = (cross_entropy[sensitive == 1]).mean()
        # DP = torch.abs(fair_term0-fair_term1)
        # fair_loss = torch.log(torch.exp(-DP)+torch.exp(DP))
        acc_loss = F.cross_entropy(logits[i], targets, reduction='mean')
        Acc_Loss.append(acc_loss)

        fair_loss0 = torch.mul(sensitive - sensitive.type(torch.FloatTensor).mean(), logits[i].T[0] - torch.mean(logits[i].T[0]))
        fair_loss0 = torch.mean(torch.mul(fair_loss0, fair_loss0))
        fair_loss1 = torch.mul(sensitive - sensitive.type(torch.FloatTensor).mean(), logits[i].T[1] - torch.mean(logits[i].T[1]))
        fair_loss1 = torch.mean(torch.mul(fair_loss1, fair_loss1))
        fair_loss = fair_loss0 + fair_loss1


        Fair_Loss.append(fair_loss)

    Acc_Loss = torch.stack(Acc_Loss)
    Fair_Loss = torch.stack(Fair_Loss)

    return Acc_Loss, Fair_Loss


def eo_loss(logits, targets, sensitive, larg, mean_z1=None, left=None, option='local fc'):
    acc_loss = F.cross_entropy(logits, targets, reduction='sum')
    y1_idx = torch.where(targets == 1)
    if option == 'unconstrained':
        return acc_loss
    if left:
        fair_loss = torch.mean(
            torch.mul(sensitive[y1_idx] - mean_z1, logits.T[0][y1_idx] - torch.mean(logits.T[0][y1_idx])))
        return acc_loss - larg * fair_loss
    elif left == False:
        fair_loss = torch.mean(
            torch.mul(sensitive[y1_idx] - mean_z1, logits.T[0][y1_idx] - torch.mean(logits.T[0][y1_idx])))
        return acc_loss + larg * fair_loss
    else:
        fair_loss0 = torch.mul(sensitive[y1_idx] - sensitive.type(torch.FloatTensor).mean(),
                               logits.T[0][y1_idx] - torch.mean(logits.T[0][y1_idx]))
        fair_loss0 = torch.mean(torch.mul(fair_loss0, fair_loss0))
        fair_loss1 = torch.mul(sensitive[y1_idx] - sensitive.type(torch.FloatTensor).mean(),
                               logits.T[1][y1_idx] - torch.mean(logits.T[1][y1_idx]))
        fair_loss1 = torch.mean(torch.mul(fair_loss1, fair_loss1))
        fair_loss = fair_loss0 + fair_loss1
        return acc_loss + larg * fair_loss


def zafar_loss(logits, targets, outputs, sensitive, larg, mean_z, left):
    acc_loss = F.cross_entropy(logits, targets, reduction='sum')
    fair_loss = torch.mean(torch.mul(sensitive - mean_z, logits.T[0] - torch.mean(logits.T[0])))
    if left:
        return acc_loss - larg * fair_loss
    else:
        return acc_loss + larg * fair_loss


def weighted_loss(logits, targets, weights, mean=True):
    acc_loss = F.cross_entropy(logits, targets, reduction='none')
    if mean:
        weights_sum = weights.sum().item()
        acc_loss = torch.sum(acc_loss * weights / weights_sum)
    else:
        acc_loss = torch.sum(acc_loss * weights)
    return acc_loss


def al_loss(logits, targets, adv_logits, adv_targets):
    acc_loss = F.cross_entropy(logits, targets, reduction='sum')
    adv_loss = F.cross_entropy(adv_logits, adv_targets)
    return acc_loss, adv_loss


def mtl_loss(logits, targets, penalty, global_model, model):
    penalty_term = torch.tensor(0., requires_grad=True).to(DEVICE)
    for v, w in zip(model.parameters(), global_model.parameters()):
        penalty_term = penalty_term + torch.norm(v - w) ** 2
    # penalty_term = torch.nodem(v-global_weights, v-global_weights)
    loss = F.cross_entropy(logits, targets, reduction='sum') + penalty / 2 * penalty_term
    return loss


# copied from https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    # m = n / arrays[0].size
    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            # for j in xrange(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
    return out


## Synthetic data generation ##
########################
####### setting ########
########################
X_DIST = {0: {"mean": (-2, -2), "cov": np.array([[10, 1], [1, 3]])},
          1: {"mean": (2, 2), "cov": np.array([[5, 1], [1, 5]])}}


def X_PRIME(x):
    return (x[0] * np.cos(np.pi / 4) - x[1] * np.sin(np.pi / 4),
            x[0] * np.sin(np.pi / 4) + x[1] * np.cos(np.pi / 4))


def Z_MEAN(x):
    """
    Given x, the probability of z = 1.
    """
    x_transform = X_PRIME(x)
    return multivariate_normal.pdf(x_transform, mean=X_DIST[1]["mean"], cov=X_DIST[1]["cov"]) / (
            multivariate_normal.pdf(x_transform, mean=X_DIST[1]["mean"], cov=X_DIST[1]["cov"]) +
            multivariate_normal.pdf(x_transform, mean=X_DIST[0]["mean"], cov=X_DIST[0]["cov"])
    )


def dataSample(train_samples=3000, test_samples=500,
               y_mean=0.6, Z=2):
    num_samples = train_samples + test_samples
    ys = np.random.binomial(n=1, p=y_mean, size=num_samples)

    xs, zs = [], []

    if Z == 2:
        for y in ys:
            x = np.random.multivariate_normal(mean=X_DIST[y]["mean"], cov=X_DIST[y]["cov"], size=1)[0]
            z = np.random.binomial(n=1, p=Z_MEAN(x), size=1)[0]
            xs.append(x)
            zs.append(z)
    elif Z == 3:
        for y in ys:
            x = np.random.multivariate_normal(mean=X_DIST[y]["mean"], cov=X_DIST[y]["cov"], size=1)[0]
            # Z = 3: 0.7 y = 1, 0.3 y = 1 + 0.3 y = 0, 0.7 y = 0
            py1 = multivariate_normal.pdf(x, mean=X_DIST[1]["mean"], cov=X_DIST[1]["cov"])
            py0 = multivariate_normal.pdf(x, mean=X_DIST[0]["mean"], cov=X_DIST[0]["cov"])
            p = np.array([0.7 * py1, 0.3 * py1 + 0.3 * py0, 0.7 * py0]) / (py1 + py0)
            z = np.random.choice([0, 1, 2], size=1, p=p)[0]
            xs.append(x)
            zs.append(z)
    elif Z==10:
        for y in ys:
            x = np.random.multivariate_normal(mean=X_DIST[y]["mean"], cov=X_DIST[y]["cov"], size=1)[0]
            # Z = 3: 0.7 y = 1, 0.3 y = 1 + 0.3 y = 0, 0.7 y = 0
            py1 = multivariate_normal.pdf(x, mean=X_DIST[1]["mean"], cov=X_DIST[1]["cov"])
            py0 = multivariate_normal.pdf(x, mean=X_DIST[0]["mean"], cov=X_DIST[0]["cov"])
            p = np.array([0.1 for i in range(10)])
            z = np.random.choice([i for i in range(10)], size=1, p=p)[0]
            xs.append(x)
            zs.append(z)
    elif Z==20:
        for y in ys:
            x = np.random.multivariate_normal(mean=X_DIST[y]["mean"], cov=X_DIST[y]["cov"], size=1)[0]
            # Z = 3: 0.7 y = 1, 0.3 y = 1 + 0.3 y = 0, 0.7 y = 0
            py1 = multivariate_normal.pdf(x, mean=X_DIST[1]["mean"], cov=X_DIST[1]["cov"])
            py0 = multivariate_normal.pdf(x, mean=X_DIST[0]["mean"], cov=X_DIST[0]["cov"])
            p = np.array([0.05 for i in range(20)])
            z = np.random.choice([i for i in range(20)], size=1, p=p)[0]
            xs.append(x)
            zs.append(z)
    elif Z==50:
        for y in ys:
            x = np.random.multivariate_normal(mean=X_DIST[y]["mean"], cov=X_DIST[y]["cov"], size=1)[0]
            # Z = 3: 0.7 y = 1, 0.3 y = 1 + 0.3 y = 0, 0.7 y = 0
            py1 = multivariate_normal.pdf(x, mean=X_DIST[1]["mean"], cov=X_DIST[1]["cov"])
            py0 = multivariate_normal.pdf(x, mean=X_DIST[0]["mean"], cov=X_DIST[0]["cov"])
            p = np.array([0.02 for i in range(50)])
            z = np.random.choice([i for i in range(50)], size=1, p=p)[0]
            xs.append(x)
            zs.append(z)
    elif Z==100:
        for y in ys:
            x = np.random.multivariate_normal(mean=X_DIST[y]["mean"], cov=X_DIST[y]["cov"], size=1)[0]
            # Z = 3: 0.7 y = 1, 0.3 y = 1 + 0.3 y = 0, 0.7 y = 0
            py1 = multivariate_normal.pdf(x, mean=X_DIST[1]["mean"], cov=X_DIST[1]["cov"])
            py0 = multivariate_normal.pdf(x, mean=X_DIST[0]["mean"], cov=X_DIST[0]["cov"])
            p = np.array([0.01 for i in range(100)])
            z = np.random.choice([i for i in range(100)], size=1, p=p)[0]
            xs.append(x)
            zs.append(z)
    elif Z==300:
        for y in ys:
            x = np.random.multivariate_normal(mean=X_DIST[y]["mean"], cov=X_DIST[y]["cov"], size=1)[0]
            # Z = 3: 0.7 y = 1, 0.3 y = 1 + 0.3 y = 0, 0.7 y = 0
            py1 = multivariate_normal.pdf(x, mean=X_DIST[1]["mean"], cov=X_DIST[1]["cov"])
            py0 = multivariate_normal.pdf(x, mean=X_DIST[0]["mean"], cov=X_DIST[0]["cov"])
            p = np.array([0.01/3 for i in range(300)])
            z = np.random.choice([i for i in range(300)], size=1, p=p)[0]
            xs.append(x)
            zs.append(z)
    data = pd.DataFrame(zip(np.array(xs).T[0], np.array(xs).T[1], ys, zs), columns=["x1", "x2", "y", "z"])
    # data = data.sample(frac=1).reset_index(drop=True)
    train_data = data[:train_samples]
    test_data = data[train_samples:]
    return train_data, test_data


def process_csv(dir_name, filename, label_name, favorable_class, sensitive_attributes, privileged_classes,
                categorical_attributes, continuous_attributes, features_to_keep, na_values=[], header='infer',
                columns=None):
    """
    process the adult file: scale, one-hot encode
    only support binary sensitive attributes -> [gender, race] -> 4 sensitive groups
    """

    df = pd.read_csv(os.path.join(dir_name, filename), delimiter=',', header=header, na_values=na_values)
    if header == None: df.columns = columns
    df = df[features_to_keep]

    # apply one-hot encoding to convert the categorical attributes into vectors
    df = pd.get_dummies(df, columns=categorical_attributes)

    # normalize numerical attributes to the range within [0, 1]
    def scale(vec):
        minimum = min(vec)
        maximum = max(vec)
        return (vec - minimum) / (maximum - minimum)

    df[continuous_attributes] = df[continuous_attributes].apply(scale, axis=0)
    df.loc[df[label_name] != favorable_class, label_name] = 'SwapSwapSwap'
    df.loc[df[label_name] == favorable_class, label_name] = 1
    df.loc[df[label_name] == 'SwapSwapSwap', label_name] = 0
    df[label_name] = df[label_name].astype('category').cat.codes
    if len(sensitive_attributes) > 1:
        if privileged_classes != None:
            for i in range(len(sensitive_attributes)):
                df.loc[df[sensitive_attributes[i]] != privileged_classes[i], sensitive_attributes[i]] = 0
                df.loc[df[sensitive_attributes[i]] == privileged_classes[i], sensitive_attributes[i]] = 1
        df['z'] = list(zip(*[df[c] for c in sensitive_attributes]))
        df['z'] = df['z'].astype('category').cat.codes
    else:
        df['z'] = df[sensitive_attributes[0]].astype('category').cat.codes
    df = df.drop(columns=sensitive_attributes)
    return df


def nsfData(q=(0.99, 0.01), theta=(0.38 / 0.99, -0.5), train_samples=3000, test_samples=300, seed=123):
    np.random.seed(seed)
    random.seed(seed)
    clients_idx = []
    train_data, test_data = [], []

    for c in range(2):
        a = np.random.binomial(n=1, p=q[c], size=train_samples // 2 + test_samples // 2)

        def prod_x(a):
            if a:
                return np.random.binomial(n=1, p=1 / 2 + theta[c], size=1)[0]
            else:
                return np.random.binomial(n=1, p=1 / 2, size=1)[0]

        prod_x_v = np.vectorize(prod_x)
        x = prod_x_v(a)
        y = copy.deepcopy(x)
        data = pd.DataFrame(zip(x, a, y), columns=["x", "a", "y"])
        train_data.append(data[:train_samples // 2])
        test_data.append(data[train_samples // 2:])
    train_data = pd.concat(train_data).reset_index(drop=True)
    test_data = pd.concat(test_data).reset_index(drop=True)
    train_data = train_data.sample(frac=1)
    test_data = test_data.sample(frac=1).reset_index(drop=True)
    clients_idx.append(np.where(train_data.index < train_samples // 2)[0])
    clients_idx.append(np.where(train_data.index >= train_samples // 2)[0])
    train_data = train_data.reset_index(drop=True)
    train_dataset = LoadData(train_data, "y", "a")
    test_dataset = LoadData(test_data, "y", "a")
    return [train_dataset, test_dataset, clients_idx]


def ufldataset(train_samples=3000, test_samples=300, seed=123):
    np.random.seed(seed)
    random.seed(seed)
    clients_idx = []
    train_data, test_data = [], []

    # client 0
    a = np.random.binomial(n=1, p=.5, size=train_samples // 2 + test_samples // 2)

    def prod_x(a):
        if a:
            return np.random.normal(0, 2, size=1)[0]
        else:
            return np.random.normal(2, 2, size=1)[0]

    prod_x_v = np.vectorize(prod_x)
    x = prod_x_v(a)

    def prod_y(x):
        return np.random.binomial(n=1, p=1 / (1 + np.exp(-x)), size=1)[0]

    prod_y_v = np.vectorize(prod_y)
    y = prod_y_v(x)

    data = pd.DataFrame(zip(x, a, y), columns=["x", "a", "y"])
    train_data.append(data[:train_samples // 2])
    test_data.append(data[train_samples // 2:])

    # client 1
    a = np.random.binomial(n=1, p=.5, size=train_samples // 2 + test_samples // 2)

    def prod_x(a):
        if a:
            return np.random.normal(0, 0.5, size=1)[0]
        else:
            return np.random.normal(-2, 0.5, size=1)[0]

    prod_x_v = np.vectorize(prod_x)
    x = prod_x_v(a)

    prod_y_v = np.vectorize(prod_y)
    y = prod_y_v(x)

    data = pd.DataFrame(zip(x, a, y), columns=["x", "a", "y"])
    train_data.append(data[:train_samples // 2])
    test_data.append(data[train_samples // 2:])

    train_data = pd.concat(train_data).reset_index(drop=True)
    test_data = pd.concat(test_data).reset_index(drop=True)
    train_data = train_data.sample(frac=1)
    test_data = test_data.sample(frac=1).reset_index(drop=True)
    clients_idx.append(np.where(train_data.index < train_samples // 2)[0])
    clients_idx.append(np.where(train_data.index >= train_samples // 2)[0])
    train_data = train_data.reset_index(drop=True)
    train_dataset = LoadData(train_data, "y", "a")
    test_dataset = LoadData(test_data, "y", "a")
    return [train_dataset, test_dataset, clients_idx]


def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)


def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)

def compare(p1, p2):
    # return 0同层 1 p1支配p2  -1 p2支配p1
    D = len(p1)
    p1_dominate_p2 = True  # p1 更小
    p2_dominate_p1 = True
    for i in range(D):
        if p1[i] > p2[i]:
            p1_dominate_p2 = False
        if p1[i] < p2[i]:
            p2_dominate_p1 = False

    if p1_dominate_p2 == p2_dominate_p1:
        return 0
    return 1 if p1_dominate_p2 else -1

def fast_non_dominated_sort(P):
    P_size = len(P)
    n = torch.full(size=(P_size,), fill_value=0, dtype=torch.long)  # 被支配数
    S = []  # 支配的成员
    f = []  # 0 开始每层包含的成员编号们
    rank = torch.full(size=(P_size,), fill_value=-1, dtype=torch.long)  # 所处等级

    f_0 = []
    for p in range(P_size):
        n_p = 0
        S_p = []
        for q in range(P_size):
            if p == q:
                continue
            cmp = compare(P[p], P[q])
            if cmp == 1:
                S_p.append(q)
            elif cmp == -1:  # 被支配
                n_p += 1
        S.append(S_p)
        n[p] = n_p
        if n_p == 0:
            rank[p] = 0
            f_0.append(p)
    f.append(f_0)

    i = 0
    while len(f[i]) != 0:  # 可能还有i+1层
        Q = []
        for p in f[i]:  # i层中每个个体
            for q in S[p]:  # 被p支配的个体
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        f.append(Q)
    rank += 1
    return rank, f
