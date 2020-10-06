from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np


class BankDataset(Dataset):
    def __init__(self, data_path):
        data = pd.read_csv(data_path, sep=';')
        print("Dataset shape: ", data.shape)
        print("Check for NaNs: ", data.isnull().values.any())
        self.len = data.shape[0]

        # DEFINE DATA COLUMN TYPES
        categorical_columns = ['age', 'job', 'marital',
                               'education', 'default', 'housing',
                               'loan', 'contact', 'month',
                               'day_of_week', 'poutcome', 'y']

        numerical_columns = ['duration', 'campaign', 'pdays',
                             'previous', 'emp.var.rate', 'cons.price.idx',
                             'cons.conf.idx', 'euribor3m', 'nr.employed']

        # CATEGORICAL TENSORS
        for category in categorical_columns:
            data[category] = data[category].astype('category')

        age = data['age'].cat.codes.values
        job = data['job'].cat.codes.values
        marital = data['marital'].cat.codes.values
        education = data['education'].cat.codes.values
        default = data['default'].cat.codes.values
        housing = data['housing'].cat.codes.values
        loan = data['loan'].cat.codes.values
        contact = data['contact'].cat.codes.values
        month = data['month'].cat.codes.values
        day_of_week = data['day_of_week'].cat.codes.values
        poutcome = data['poutcome'].cat.codes.values

        categorical_data = np.stack([age, job, marital,
                                     education, default, housing,
                                     loan, contact, month,
                                     day_of_week, poutcome], 1)
        categorical_data = torch.tensor(categorical_data, dtype=torch.int64)


        # NUMERICAL TENSORS
        numerical_data = np.stack([data[col].values for col in numerical_columns], 1)
        numerical_data = torch.tensor(numerical_data, dtype=torch.float64)



        # FINAL TENSORs
        self.X = torch.cat((categorical_data, numerical_data), 1)
        self.Y = torch.tensor(data['y'].cat.codes.values.flatten(), dtype=torch.int64)


    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len


def make_weights_for_balanced_classes(df, nclasses):
    count = [0] * nclasses
    for item in df:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(df)
    for idx, val in enumerate(df):
        weight[idx] = weight_per_class[val[1]]
    return weight
