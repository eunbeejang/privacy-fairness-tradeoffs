from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import numpy as np


class BankDataset(Dataset):
    def __init__(self):
        data = pd.read_csv('./bank-data/bank-additional-full.csv', sep=';')
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


dataset = BankDataset()


# Creating data indices for training and validation splits:
batch_size = 16
test_split = .2
random_seed= 42

validation_split = .2
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(test_split * dataset_size))
np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
