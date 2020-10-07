from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class BankDataset(Dataset):
    def __init__(self, data):

        print("Dataset shape: ", data.shape)
        print("Check for NaNs: ", data.isnull().values.any())
        self.len = data.shape[0]

        # DEFINE DATA COLUMN TYPES
        categorical_columns = ['age', 'job', 'marital',
                               'education', 'default', 'housing',
                               'loan', 'contact', 'month',
                               'day_of_week', 'poutcome']

        numerical_columns = ['duration', 'campaign', 'pdays',
                             'previous', 'emp.var.rate', 'cons.price.idx',
                             'cons.conf.idx', 'euribor3m', 'nr.employed']

        # CATEGORICAL TENSORS
        for category in categorical_columns:
            data[category] = data[category].astype('category')
        data['y'] = data['y'].astype('category')
        num_classes = len(data['y'].cat.categories)

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
        Y = data['y'].cat.codes.values


        categorical_data = np.stack([age, job, marital,
                                     education, default, housing,
                                     loan, contact, month,
                                     day_of_week, poutcome], 1)
        self.categorical_data = torch.tensor(categorical_data, dtype=torch.int64)


        # NUMERICAL TENSORS
        numerical_data = np.stack([data[col].values for col in numerical_columns], 1)
        self.numerical_data = torch.tensor(numerical_data, dtype=torch.float64)



        # FINAL TENSORs
        #self.X = torch.cat((categorical_data, numerical_data), 1)

        self.Y = torch.tensor(Y.flatten(), dtype=torch.int64)

        categorical_column_sizes = [len(data[column].cat.categories) for column in categorical_columns]
        self.categorical_embedding_sizes = [(col_size, min(50, (col_size + 1) // 2)) for col_size in
                                       categorical_column_sizes]
        self.num_numerical_cols = self.numerical_data.shape[1]

#        self.emb_size = categorical_embedding_sizes
#        self.emb_size.append((num_numerical_cols, 1))
#        print(self.emb_size)

    def __getitem__(self, idx):
        return self.categorical_data[idx], self.numerical_data[idx], self.Y[idx]

    def __len__(self):
        return self.len



data_path = "./bank-data/bank-additional-full.csv"
data = pd.read_csv(data_path, sep=';')
data = data.sample(frac=1).reset_index(drop=True)
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

train_data = BankDataset(train_df)
test_data = BankDataset(test_df)

cat_emb_size = train_data.categorical_embedding_sizes
num_conts = train_data.num_numerical_cols

train_loader = DataLoader(dataset=train_data,
                          sampler=WeightedRandomSampler([.7,.3], 128, replacement=True),
                          drop_last = False)

test_loader = DataLoader(dataset=test_data,
                         batch_size=128,
                         drop_last = False)