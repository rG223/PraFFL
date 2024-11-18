import numpy as np
from utils import *
import torch

# synthetic
def dataSplit(train_data, test_data, client_split = ((.5, .2), (.3, .4), (.2, .4)), Z = 2):
    if Z == 2:
        z1_idx = train_data[train_data.z == 1].index
        z0_idx = train_data[train_data.z == 0].index
        test_z1_idx = test_data[test_data.z == 1].index
        test_z0_idx = test_data[test_data.z == 0].index

        client1_idx = np.concatenate((z1_idx[:int(client_split[0][0]*len(z1_idx))], z0_idx[:int(client_split[0][1]*len(z0_idx))]))
        client2_idx = np.concatenate((z1_idx[int(client_split[0][0]*len(z1_idx)):int((client_split[0][0] + client_split[1][0])*len(z1_idx))],
                                      z0_idx[int(client_split[0][1]*len(z0_idx)):int((client_split[0][1] + client_split[1][1])*len(z0_idx))]))
        client3_idx = np.concatenate((z1_idx[int((client_split[0][0] + client_split[1][0])*len(z1_idx)):], z0_idx[int((client_split[0][1] + client_split[1][1])*len(z0_idx)):]))

        test_client1_idx = np.concatenate(
            (test_z1_idx[:int(client_split[0][0] * len(test_z1_idx))], test_z0_idx[:int(client_split[0][1] * len(test_z0_idx))]))
        test_client2_idx = np.concatenate(
            (test_z1_idx[int(client_split[0][0] * len(test_z1_idx)):int((client_split[0][0] + client_split[1][0]) * len(test_z1_idx))],
             test_z0_idx[
             int(client_split[0][1] * len(test_z0_idx)):int((client_split[0][1] + client_split[1][1]) * len(test_z0_idx))]))
        test_client3_idx = np.concatenate((test_z1_idx[int((client_split[0][0] + client_split[1][0]) * len(test_z1_idx)):],
                                      test_z0_idx[int((client_split[0][1] + client_split[1][1]) * len(test_z0_idx)):]))

        random.shuffle(client1_idx)
        random.shuffle(client2_idx)
        random.shuffle(client3_idx)

        clients_idx = [client1_idx, client2_idx, client3_idx]
        test_clients_idx = [test_client1_idx, test_client2_idx, test_client3_idx]

    elif Z == 3:
        z_idx, z_len = [], []
        test_z_idx, test_z_len = [], []
        for z in range(3):
            z_idx.append(train_data[train_data.z == z].index)
            z_len.append(len(z_idx[z]))

            test_z_idx.append(test_data[test_data.z == z].index)
            test_z_len.append(len(test_z_idx[z]))

        clients_idx = []
        test_clients_idx = []

        a, b = np.zeros(3), np.zeros(3)
        ta, tb = np.zeros(3), np.zeros(3)
        for c in range(4):
            if c > 0:
                a += np.array(client_split[c-1]) * z_len
                ta += np.array(client_split[c - 1]) * test_z_len
            b += np.array(client_split[c]) * z_len
            tb += np.array(client_split[c]) * test_z_len
            clients_idx.append(np.concatenate((z_idx[0][int(a[0]):int(b[0])],
                                               z_idx[1][int(a[1]):int(b[1])],
                                               z_idx[2][int(a[2]):int(b[2])])))
            test_clients_idx.append(np.concatenate((test_z_idx[0][int(ta[0]):int(tb[0])],
                                               test_z_idx[1][int(ta[1]):int(tb[1])],
                                               test_z_idx[2][int(ta[2]):int(tb[2])])))
            random.shuffle(clients_idx[c])
        
    elif Z == 10:
        z_idx, z_len = [], []
        test_z_idx, test_z_len = [], []
        for z in range(10):
            z_idx.append(train_data[train_data.z == z].index)
            z_len.append(len(z_idx[z]))

            test_z_idx.append(test_data[test_data.z == z].index)
            test_z_len.append(len(test_z_idx[z]))

        clients_idx = []
        test_clients_idx = []
        a, b = np.zeros(10), np.zeros(10)
        ta, tb = np.zeros(10), np.zeros(10)
        for c in range(11):
            if c > 0:
                a += np.array(client_split[c-1]) * z_len
                ta += np.array(client_split[c - 1]) * test_z_len

            b += np.array(client_split[c]) * z_len
            tb += np.array(client_split[c]) * test_z_len
            clients_idx.append(np.concatenate((z_idx[0][int(a[0]):int(b[0])],
                                               z_idx[1][int(a[1]):int(b[1])],
                                               z_idx[2][int(a[2]):int(b[2])],
                                               z_idx[3][int(a[3]):int(b[3])],
                                               z_idx[4][int(a[4]):int(b[4])],
                                               z_idx[5][int(a[5]):int(b[5])],
                                               z_idx[6][int(a[6]):int(b[6])],
                                               z_idx[7][int(a[7]):int(b[7])],
                                               z_idx[8][int(a[8]):int(b[8])],
                                               z_idx[9][int(a[9]):int(b[9])])))

            test_clients_idx.append(np.concatenate((test_z_idx[0][int(ta[0]):int(tb[0])],
                                                    test_z_idx[1][int(ta[1]):int(tb[1])],
                                                    test_z_idx[2][int(ta[2]):int(tb[2])],
                                                    test_z_idx[3][int(ta[3]):int(tb[3])],
                                                    test_z_idx[4][int(ta[4]):int(tb[4])],
                                                    test_z_idx[5][int(ta[5]):int(tb[5])],
                                                    test_z_idx[6][int(ta[6]):int(tb[6])],
                                                    test_z_idx[7][int(ta[7]):int(tb[7])],
                                                    test_z_idx[8][int(ta[8]):int(tb[8])],
                                                    test_z_idx[9][int(ta[9]):int(tb[9])])))
            random.shuffle(clients_idx[c])


    elif Z == 20:
        z_idx, z_len = [], []
        test_z_idx, test_z_len = [], []
        for z in range(20):
            z_idx.append(train_data[train_data.z == z].index)
            z_len.append(len(z_idx[z]))

            test_z_idx.append(test_data[test_data.z == z].index)
            test_z_len.append(len(test_z_idx[z]))

        clients_idx = []
        test_clients_idx = []
        a, b = np.zeros(20), np.zeros(20)
        ta, tb = np.zeros(20), np.zeros(20)
        for c in range(21):
            if c > 0:
                a += np.array(client_split[c-1]) * z_len
                ta += np.array(client_split[c - 1]) * test_z_len

            b += np.array(client_split[c]) * z_len
            tb += np.array(client_split[c]) * test_z_len

            clients_idx.append(np.concatenate(
                [z_idx[i][int(a[i]):int(b[i])] for i in range(len(z_idx))]
            ))

            test_clients_idx.append(np.concatenate(
                [test_z_idx[i][int(ta[i]):int(tb[i])] for i in range(len(test_z_idx))]
            ))
            random.shuffle(clients_idx[c])
    elif Z== 50:
        z_idx, z_len = [], []
        test_z_idx, test_z_len = [], []
        for z in range(50):
            z_idx.append(train_data[train_data.z == z].index)
            z_len.append(len(z_idx[z]))

            test_z_idx.append(test_data[test_data.z == z].index)
            test_z_len.append(len(test_z_idx[z]))

        clients_idx = []
        test_clients_idx = []
        a, b = np.zeros(50), np.zeros(50)
        ta, tb = np.zeros(50), np.zeros(50)
        for c in range(51):
            if c > 0:
                a += np.array(client_split[c-1]) * z_len
                ta += np.array(client_split[c - 1]) * test_z_len

            b += np.array(client_split[c]) * z_len
            tb += np.array(client_split[c]) * test_z_len

            clients_idx.append(np.concatenate(
                [z_idx[i][int(a[i]):int(b[i])] for i in range(len(z_idx))]
            ))

            test_clients_idx.append(np.concatenate(
                [test_z_idx[i][int(ta[i]):int(tb[i])] for i in range(len(test_z_idx))]
            ))
            random.shuffle(clients_idx[c])
    elif Z== 100:
        z_idx, z_len = [], []
        test_z_idx, test_z_len = [], []
        for z in range(101):
            z_idx.append(train_data[train_data.z == z].index)
            z_len.append(len(z_idx[z]))

            test_z_idx.append(test_data[test_data.z == z].index)
            test_z_len.append(len(test_z_idx[z]))

        clients_idx = []
        test_clients_idx = []
        a, b = np.zeros(100), np.zeros(100)
        ta, tb = np.zeros(100), np.zeros(100)
        for c in range(101):
            if c > 0:
                a += np.array(client_split[c-1]) * z_len
                ta += np.array(client_split[c - 1]) * test_z_len

            b += np.array(client_split[c]) * z_len
            tb += np.array(client_split[c]) * test_z_len

            clients_idx.append(np.concatenate(
                [z_idx[i][int(a[i]):int(b[i])] for i in range(len(z_idx))]
            ))

            test_clients_idx.append(np.concatenate(
                [test_z_idx[i][int(ta[i]):int(tb[i])] for i in range(len(test_z_idx))]
            ))
            random.shuffle(clients_idx[c])

    elif Z== 300:
        z_idx, z_len = [], []
        test_z_idx, test_z_len = [], []
        for z in range(301):
            z_idx.append(train_data[train_data.z == z].index)
            z_len.append(len(z_idx[z]))

            test_z_idx.append(test_data[test_data.z == z].index)
            test_z_len.append(len(test_z_idx[z]))

        clients_idx = []
        test_clients_idx = []
        a, b = np.zeros(300), np.zeros(300)
        ta, tb = np.zeros(50), np.zeros(300)
        for c in range(301):
            if c > 0:
                a += np.array(client_split[c-1]) * z_len
                ta += np.array(client_split[c - 1]) * test_z_len

            b += np.array(client_split[c]) * z_len
            tb += np.array(client_split[c]) * test_z_len

            clients_idx.append(np.concatenate(
                [z_idx[i][int(a[i]):int(b[i])] for i in range(len(z_idx))]
            ))

            test_clients_idx.append(np.concatenate(
                [test_z_idx[i][int(ta[i]):int(tb[i])] for i in range(len(test_z_idx))]
            ))
            random.shuffle(clients_idx[c])
    train_dataset = LoadData(train_data, "y", "z")
    test_dataset = LoadData(test_data, "y", "z")


    synthetic_info = [train_dataset, test_dataset, clients_idx, test_clients_idx]
    return synthetic_info

def dataGenerate(seed = 432, train_samples = 3000, test_samples = 500, 
                y_mean = 0.6, client_split = ((.5, .2), (.3, .4), (.2, .4)), Z = 2):
    """
    Generate dataset consisting of two sensitive groups.
    """
    ########################
    # Z = 2:
    # 3 clients: 
    #           client 1: %50 z = 1, %20 z = 0
    #           client 2: %30 z = 1, %40 z = 0
    #           client 3: %20 z = 1, %40 z = 0
    ########################
    # 4 clients:
    #           client 1: 50% z = 0, 10% z = 1, 20% z = 2
    #           client 2: 30% z = 0, 30% z = 1, 30% z = 2
    #           client 3: 10% z = 0, 30% z = 1, 30% z = 2
    #           client 4: 10% z = 0, 30% z = 1, 20% z = 2
    ########################
    np.random.seed(seed)
    random.seed(seed)
        
    train_data, test_data = dataSample(train_samples, test_samples, y_mean, Z)
    return dataSplit(train_data, test_data, client_split, Z)

Heterogeneity_Level = {0.1: ((0.474, 0.09), (0.09, 0.818), (0.436, 0.092)),
                       0.2: ((0.44, 0.17), (0.11, 0.67), (0.45, 0.16)),
                       # 0.5: ((0.4, 0.25), (0.25, 0.5), (0.35, 0.25)),
                       5: ((0.49924409, 0.33939015), (0.27875569, 0.42313098), (0.22200021, 0.23747887)),
                       1000: ((0.33855557, 0.33149328), (0.32945964, 0.33188017), (0.33198478, 0.33662654))}
# 0.1 ((9.44124347e-01, 6.18674676e-06), (1.52101650e-07, 9.99993786e-01), (5.58755004e-02, 2.72863279e-08))
# 0.2 ((0.00784015, 0.76961015), (0.5493687, 0.01504449), (0.44279115, 0.21534536))
# 0.5 ((0.03903129, 0.0203803), (0.95934784, 0.89899236), (0.00162087, 0.08062734))
# 5 ((0.49924409, 0.33939015), (0.27875569, 0.42313098), (0.22200021, 0.23747887))
# 5000 ((0.33855557, 0.33149328), (0.32945964, 0.33188017), (0.33198478, 0.33662654))

# synthetic_info = dataGenerate(seed = 123, test_samples = 1500, train_samples = 3500, client_split=Heterogeneity_Level[5])
# synthetic_info = dataGenerate(seed = 123, test_samples = 1500, train_samples = 3500, client_split=((0.5,0.2),(0.3,0.4),(0.2, 0.4)))

synthetic_info = dataGenerate(seed = 123, test_samples = 2000, train_samples = 5000, client_split=((0.08,0.1),(0.1,0.12),(0.15, 0.05),(0.15, 0.05),(0.12, 0.08)
                                                                                                   ,(0.12, 0.1),(0.1, 0.08),(0.05, 0.15),(0.05, 0.15),(0.08, 0.12)))

# A = [1/50]*10 + [0.5/50]*20 + [1.5/50]*20
# B = [1.5/50]*20 + [1/50]*10 + [0.5/50]*20
# combined_tuple = tuple(zip(A, B))
# synthetic_info = dataGenerate(seed = 123, test_samples = 5000, train_samples = 15000, client_split=combined_tuple)

# A = [1/100]*20 + [0.4/100]*40 + [1.6/100]*40
# B = [1.6/100]*40 + [1/100]*20 + [0.4/100]*40
# combined_tuple = tuple(zip(A, B))
# synthetic_info = dataGenerate(seed = 123, test_samples = 8000, train_samples = 20000, client_split=combined_tuple)


# A = [1/300]*60 + [0.5/300]*120 + [1.5/300]*120
# B = [1.5/300]*120 + [1/300]*60 + [0.5/300]*120
# combined_tuple = tuple(zip(A, B))
# synthetic_info = dataGenerate(seed = 123, test_samples = 8000, train_samples = 25000, client_split=combined_tuple)


# synthetic_info = dataGenerate(seed = 123, test_samples = 1500, train_samples = 10000, client_split=((0.05,0.2),(0.05,0.3),(0.05, 0.2),(0.05, 0.05),(0.05, 0.05)
#                                                                                                    ,(0.02, 0.05),(0.03, 0.05),(0.2, 0.05),(0.3, 0.03),(0.2, 0.02)))

# A = np.arange(0, 50 * 1/1225, 1/1225)
# B = 0.04 - A
# combined_tuple = tuple(zip(A, B))
# synthetic_info = dataGenerate(seed = 123, test_samples = 1600, train_samples = 16000, client_split=combined_tuple)

# A = np.arange(0, 100 * 1/4950, 1/4950)
# B = 0.02 - A
# combined_tuple = tuple(zip(A, B))
# synthetic_info = dataGenerate(seed = 123, test_samples = 1700, train_samples = 17000, client_split=combined_tuple)

# A = np.arange(0, 300 * 1/44850, 1/44850)
# B = 0.006667 - A
# combined_tuple = tuple(zip(A, B))
# synthetic_info = dataGenerate(seed = 123, test_samples = 1800, train_samples = 18000, client_split=combined_tuple)
# Adult
sensitive_attributes = ['sex']
categorical_attributes = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
continuous_attributes = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
features_to_keep = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss','hours-per-week', 
            'native-country', 'salary']
label_name = 'salary'

adult = process_csv('adult', 'adult.data', label_name, ' >50K', sensitive_attributes, [' Female'], categorical_attributes, continuous_attributes, features_to_keep, na_values = [], header = None, columns = features_to_keep)
test = process_csv('adult', 'adult.test', label_name, ' >50K.', sensitive_attributes, [' Female'], categorical_attributes, continuous_attributes, features_to_keep, na_values = [], header = None, columns = features_to_keep) # the distribution is very different from training distribution
test['native-country_ Holand-Netherlands'] = 0
test = test[adult.columns]

np.random.seed(1)
adult_private_idx = adult[adult['workclass_ Private'] == 1].index
adult_others_idx = adult[adult['workclass_ Private'] == 0].index
test_adult_private_idx = test[test['workclass_ Private'] == 1].index
test_adult_others_idx = test[test['workclass_ Private'] == 0].index
adult_mean_sensitive = adult['z'].mean()

client1_idx = np.concatenate((np.random.choice(adult_private_idx, int(.8*len(adult_private_idx)), replace = False),
                                np.random.choice(adult_others_idx, int(.2*len(adult_others_idx)), replace = False)))
client2_idx = np.array(list(set(adult.index) - set(client1_idx)))
test_client1_idx = np.concatenate((np.random.choice(test_adult_private_idx, int(.8*len(test_adult_private_idx)), replace = False),
                                np.random.choice(test_adult_others_idx, int(.2*len(test_adult_others_idx)), replace = False)))
test_client2_idx = np.array(list(set(test.index) - set(test_client1_idx)))

adult_clients_idx = [client1_idx, client2_idx]
test_adult_clients_idx = [test_client1_idx, test_client2_idx]

adult_num_features = len(adult.columns)-1
adult_test = LoadData(test, 'salary', 'z')
adult_train = LoadData(adult, 'salary', 'z')
torch.manual_seed(0)
adult_info = [adult_train, adult_test, adult_clients_idx, test_adult_clients_idx]

# COMPAS
sensitive_attributes = ['sex', 'race']
categorical_attributes = ['age_cat', 'c_charge_degree', 'c_charge_desc']
continuous_attributes = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count']
features_to_keep = ['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
        'priors_count', 'c_charge_degree', 'c_charge_desc','two_year_recid']
label_name = 'two_year_recid'

compas = process_csv('compas', 'compas-scores-two-years.csv', label_name, 0, sensitive_attributes, ['Female', 'African-American'], categorical_attributes, continuous_attributes, features_to_keep)
train = compas.iloc[:int(len(compas)*.7)]
test = compas.iloc[int(len(compas)*.7):]

np.random.seed(1)
torch.manual_seed(0)
client1_idx = train[train.age > 0.1].index 
client2_idx = train[train.age <= 0.1].index
test_client1_idx = test[test.age > 0.1].index
test_client2_idx = test[test.age <= 0.1].index

compas_mean_sensitive = train['z'].mean()
compas_z = len(set(compas.z))

clients_idx = [client1_idx, client2_idx]
test_clients_idx = [test_client1_idx, test_client2_idx]

compas_num_features = len(compas.columns) - 1
compas_train = LoadData(train, label_name, 'z')
compas_test = LoadData(test, label_name, 'z')

compas_info = [compas_train, compas_test, clients_idx, test_clients_idx]

# communities
np.random.seed(1)
torch.manual_seed(0)

sensitive_attributes = ['racePctWhite', 'racepctblack', 'racePctAsian', 'racePctHisp']
categorical_attributes = []
df = pd.read_csv(os.path.join('communities', 'communities_process.csv'))
features_to_keep = list(set(df.columns) - {'communityname'})
continuous_attributes = list(set(features_to_keep) - {'racePctWhite', 'racepctblack', 'racePctAsian', 'racePctHisp', 'state'})
label_name = 'ViolentCrimesPerPop'

communities = process_csv('communities', 'communities_process.csv', label_name, 1, sensitive_attributes, None, categorical_attributes, continuous_attributes, features_to_keep)
communities = communities.sample(frac=1).reset_index(drop=True)
train = communities.iloc[:int(len(communities)*.7)]
test = communities.iloc[int(len(communities)*.7):]

state_high_idx = np.where(train.state > 20)[0]
state_low_idx = np.where(train.state <= 20)[0]

client1_idx = train[train.state > 20].index
client2_idx = train[train.state <= 20].index

test_client1_idx = test[test.state > 20].index
test_client2_idx = test[test.state <= 20].index

train = train.drop(columns = ['state'])
test = test.drop(columns = ['state'])
communities_mean_sensitive = train['z'].mean()
communities_z = len(set(communities.z))

clients_idx = [client1_idx, client2_idx]
test_clients_idx = [test_client1_idx, test_client2_idx]

communities_num_features = len(train.columns) - 1
communities_train = LoadData(train, label_name, 'z')
communities_test = LoadData(test, label_name, 'z')

communities_info = [communities_train, communities_test, clients_idx, test_clients_idx]

# Bank
######################################################
### Pre-processing code (leave here for reference) ###
######################################################
# import pandas as pd
# import numpy as np
# import os
# from utils import LoadData

# df = pd.read_csv(os.path.join('bank', 'bank-full.csv'), sep = ';')
# q1 = df.age.quantile(q = 0.2)
# q1_idx = np.where(df.age <= q1)[0]
# q2 = df.age.quantile(q = 0.4)
# q2_idx = np.where((q1 < df.age) & (df.age <= q2))[0]
# q3 = df.age.quantile(q = 0.6)
# q3_idx = np.where((q2 < df.age) & (df.age <= q3))[0]
# q4 = df.age.quantile(q = 0.8)
# q4_idx = np.where((q3 < df.age) & (df.age <= q4))[0]
# q5_idx = np.where(df.age > q4)[0]
# df.loc[q1_idx, 'age'] = 0
# df.loc[q2_idx, 'age'] = 1
# df.loc[q3_idx, 'age'] = 2
# df.loc[q4_idx, 'age'] = 3
# df.loc[q5_idx, 'age'] = 4
# df.to_csv(os.path.join('bank', 'bank_cat_age.csv'))
######################################################

np.random.seed(1)
torch.manual_seed(0)
sensitive_attributes = ['age']
categorical_attributes = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
continuous_attributes = ['balance', 'duration', 'campaign', 'pdays', 'previous']
features_to_keep = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome', 
                    'balance', 'duration', 'campaign', 'pdays', 'previous', 'y']
label_name = 'y'

bank = process_csv('bank', 'bank_cat_age.csv', label_name, 'yes', sensitive_attributes, None, categorical_attributes, continuous_attributes, features_to_keep, na_values = [])
bank = bank.sample(frac=1).reset_index(drop=True)
train = bank.iloc[:int(len(bank)*.7)]
test = bank.iloc[int(len(bank)*.7):]

loan_idx = np.where(train.loan_no == 1)[0]
loan_no_idx = np.where(train.loan_no == 0)[0]
test_loan_idx = np.where(test.loan_no == 1)[0]
test_loan_no_idx = np.where(test.loan_no == 0)[0]

client1_idx = np.concatenate((loan_idx[:int(len(loan_idx)*.5)], loan_no_idx[:int(len(loan_no_idx)*.2)]))
client2_idx = np.concatenate((loan_idx[int(len(loan_idx)*.5):int(len(loan_idx)*.6)], loan_no_idx[int(len(loan_no_idx)*.2):int(len(loan_no_idx)*.8)]))
client3_idx = np.concatenate((loan_idx[int(len(loan_idx)*.6):], loan_no_idx[int(len(loan_no_idx)*.8):]))
test_client1_idx = np.concatenate((test_loan_idx[:int(len(test_loan_idx)*.5)], test_loan_no_idx[:int(len(test_loan_no_idx)*.2)]))
test_client2_idx = np.concatenate((test_loan_idx[int(len(test_loan_idx)*.5):int(len(test_loan_idx)*.6)], test_loan_no_idx[int(len(test_loan_no_idx)*.2):int(len(test_loan_no_idx)*.8)]))
test_client3_idx = np.concatenate((test_loan_idx[int(len(test_loan_idx)*.6):], test_loan_no_idx[int(len(test_loan_no_idx)*.8):]))

np.random.shuffle(client1_idx)
np.random.shuffle(client2_idx)
np.random.shuffle(client3_idx)

bank_mean_sensitive = train['z'].mean()
bank_z = len(set(bank.z))

clients_idx = [client1_idx, client2_idx, client3_idx]
test_clients_idx = [test_client1_idx, test_client2_idx, test_client3_idx]
bank_num_features = len(bank.columns) - 1
bank_train = LoadData(train, label_name, 'z')
bank_test = LoadData(test, label_name, 'z')

bank_info = [bank_train, bank_test, clients_idx,test_clients_idx]